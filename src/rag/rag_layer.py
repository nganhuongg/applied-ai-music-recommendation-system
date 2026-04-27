"""
src/rag/rag_layer.py

The recommendation pipeline — called at every playlist refresh.

Retrieval-first RAG pipeline:

    User types intent: "late night jazz piano, melancholy, no lyrics"
              ↓
    [RAG step]  TF-IDF search over 81k song descriptions
                → top-N song IDs whose descriptions best match the intent
              ↓
    [ML step]   rank(candidates, user_profile) → score by personal taste
              ↓
    [Groq]      Write one sentence per song explaining why it fits
              ↓
    Top-K songs with personalised explanations

Why TF-IDF for retrieval:
    The 81k catalog cannot fit in any LLM context window.  TF-IDF search is
    the ONLY way to narrow to relevant songs from a free-text query.  No API
    calls, no rate limits — the pre-built index loads in ~1 second.

Why Groq for explanations:
    Groq provides fast, free LLM inference (Llama models).  The explanation
    task is simple enough that a fast model handles it well.

Fallback chain:
    1. No TF-IDF index built yet → profile-based retrieval (retriever.py)
    2. Groq call fails → return songs with empty explanations (ML ranking stands)
    3. No intent typed → profile-based retrieval, no explanations

Public API:
    recommend(user_intent, user_profile, history_summary, api_key, k=10)
        -> List[Dict]  [{song_id, title, artist, score, explanation}]

    retrieve_songs(user_intent, top_n=100)
        -> List[str]   song_ids ordered by TF-IDF similarity to the intent
"""

import csv
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
DATA_DIR       = REPO_ROOT / "data"

TFIDF_MATRIX_PATH = EMBEDDINGS_DIR / "knowledge_base_tfidf.npz"
VECTORIZER_PATH   = EMBEDDINGS_DIR / "knowledge_base_vectorizer.pkl"
DOCS_PATH         = EMBEDDINGS_DIR / "knowledge_base_docs.json"
SONGS_CSV         = DATA_DIR / "songs_full.csv"

# ── constants ──────────────────────────────────────────────────────────────────
GROQ_CHAT_URL      = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL         = "llama-3.3-70b-versatile"   # best quality on Groq free tier
RAG_CANDIDATE_COUNT = 100    # songs retrieved from the catalog per intent query
DEFAULT_K           = 10     # final songs returned to the user
MAX_CHAT_RETRIES    = 3
CHAT_RETRY_DELAY    = 5.0    # seconds between retries on 429 / 5xx

# ── module-level caches ────────────────────────────────────────────────────────
# The TF-IDF matrix is stored as raw CSR arrays (no scipy) so it can be
# multiplied with a dense query vector using plain numpy (see _csr_dot).
_tfidf_data:    Optional[np.ndarray] = None   # non-zero values
_tfidf_indices: Optional[np.ndarray] = None   # column indices of non-zeros
_tfidf_indptr:  Optional[np.ndarray] = None   # row pointer array
_tfidf_shape:   Optional[tuple]      = None   # (n_songs, vocab_size)
_vectorizer      = None                        # fitted TfidfVectorizer
_kb_docs:        Optional[List[Dict]] = None   # parallel list of doc metadata
_kb_load_attempted: bool              = False

_songs_by_id:    Optional[Dict[str, Dict]] = None
_songs_load_attempted: bool                = False


# ── loaders ────────────────────────────────────────────────────────────────────

def _load_knowledge_base() -> None:
    """
    Load the TF-IDF index (as raw numpy CSR arrays), vectorizer, and doc list
    into memory once per process.  No scipy required.

    If the files don't exist, the cache stays None and recommend() falls back to
    profile-based retrieval.
    """
    global _tfidf_data, _tfidf_indices, _tfidf_indptr, _tfidf_shape
    global _vectorizer, _kb_docs, _kb_load_attempted
    if _kb_load_attempted:
        return
    _kb_load_attempted = True

    if not TFIDF_MATRIX_PATH.exists() or not VECTORIZER_PATH.exists() or not DOCS_PATH.exists():
        print("[rag_layer] TF-IDF index not found — run build_knowledge_base.py first.")
        print("[rag_layer] Falling back to profile-based retrieval.")
        return

    print("[rag_layer] Loading TF-IDF index ...")
    loaded = np.load(str(TFIDF_MATRIX_PATH))
    _tfidf_data    = loaded["data"].astype(np.float32)
    _tfidf_indices = loaded["indices"].astype(np.int32)
    _tfidf_indptr  = loaded["indptr"].astype(np.int32)
    _tfidf_shape   = tuple(int(x) for x in loaded["shape"])

    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    with open(DOCS_PATH, encoding="utf-8") as f:
        _kb_docs = json.load(f)

    print(f"[rag_layer] Index loaded: {_tfidf_shape[0]:,} songs, "
          f"vocab={_tfidf_shape[1]:,}.")


def _load_songs() -> None:
    """Build song_id → metadata dict from songs_full.csv, once per process."""
    global _songs_by_id, _songs_load_attempted
    if _songs_load_attempted:
        return
    _songs_load_attempted = True

    if not SONGS_CSV.exists():
        return

    _songs_by_id = {}
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = row.get("song_id", "").strip()
            if sid:
                _songs_by_id[sid] = {
                    "title":       row.get("title", "Unknown").strip(),
                    "artist_name": row.get("artist_name", "Unknown").strip().split(";")[0].strip(),
                }


# ── TF-IDF helpers (no scipy) ─────────────────────────────────────────────────

def _tfidf_query_vector(query_text: str) -> np.ndarray:
    """
    Convert a query string to a dense TF-IDF vector without calling
    vectorizer.transform() (which returns a scipy sparse matrix).

    Replicates TfidfVectorizer(sublinear_tf=True, norm='l2'):
        tf  = 1 + log(count)   for each term that appears
        weight = tf * idf[term]
        then L2-normalise the whole vector
    """
    tokens = _vectorizer.build_analyzer()(query_text)
    counts: dict = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    vocab = _vectorizer.vocabulary_
    idf   = _vectorizer.idf_
    vec   = np.zeros(len(vocab), dtype=np.float32)

    for term, count in counts.items():
        if term in vocab:
            tf = 1.0 + np.log(float(count))
            vec[vocab[term]] = tf * idf[vocab[term]]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _csr_dot(query_vec: np.ndarray) -> np.ndarray:
    """
    Multiply the cached CSR sparse matrix by a dense query vector.

    For each document row i, compute dot(row_i, query_vec).  Because both the
    matrix rows and the query vector are L2-normalised, this equals cosine sim.

    Uses np.bincount instead of a Python loop — runs in compiled numpy C code
    over ~5-8 M non-zeros in ~50 ms even without scipy.
    """
    n_rows   = _tfidf_shape[0]
    row_ids  = np.repeat(
        np.arange(n_rows, dtype=np.int32),
        np.diff(_tfidf_indptr),
    )
    contrib  = _tfidf_data * query_vec[_tfidf_indices]
    return np.bincount(row_ids, weights=contrib, minlength=n_rows).astype(np.float32)


# ── TF-IDF retrieval ───────────────────────────────────────────────────────────

def retrieve_songs(
    user_intent: str,
    top_n: int = RAG_CANDIDATE_COUNT,
) -> List[str]:
    """
    Search the TF-IDF index and return the top_n most relevant song_ids.

    How TF-IDF search works:
        1. The vectorizer converts user_intent into a sparse vector where each
           dimension is a vocabulary term weighted by TF-IDF.
        2. Cosine similarity between the query vector and each song's description
           vector is computed as a sparse dot product (both are L2-normalised).
        3. Songs with the highest scores are returned.

    Because the TF-IDF matrix rows are L2-normalised (norm='l2' in the vectorizer),
    the dot product between the query vector and a row equals their cosine similarity
    — exactly the same math as dense embedding search, just with a sparse matrix.

    Args:
        user_intent: Free-text, e.g. "late night jazz piano, melancholy, no lyrics".
        top_n:       Number of candidate songs to return.

    Returns:
        List of song_ids ordered best-match first.
        Empty list if the index hasn't been built yet.
    """
    _load_knowledge_base()
    if _tfidf_data is None or _vectorizer is None or _kb_docs is None:
        return []

    query_vec = _tfidf_query_vector(user_intent)  # dense [vocab_size]
    scores    = _csr_dot(query_vec)               # dense [n_songs]

    top_indices = scores.argsort()[::-1][:top_n]

    song_ids = []
    for idx in top_indices:
        doc = _kb_docs[int(idx)]
        sid = doc.get("song_id", "")
        if sid:
            song_ids.append(sid)

    return song_ids


# ── Groq explanation call ──────────────────────────────────────────────────────

def _build_explanation_prompt(
    top_songs: List[Dict],
    user_intent: str,
    history_summary: str,
    song_meta: Dict[str, Dict],
    k: int,
) -> str:
    """
    Build a focused prompt asking Groq only to write explanations.

    Groq's role is narrow: the ML ranker already ranked the songs, the TF-IDF
    already matched them to the intent.  Groq only adds natural-language prose
    explaining why each song fits, citing the user's own words where possible.
    """
    songs_for_prompt = []
    for i, song in enumerate(top_songs[:k], 1):
        sid  = song.get("song_id", "")
        meta = song_meta.get(sid, {})
        songs_for_prompt.append({
            "rank":     i,
            "song_id":  sid,
            "title":    meta.get("title", "Unknown"),
            "artist":   meta.get("artist_name", "Unknown"),
            "ml_score": round(float(song.get("score", 0.0)), 4),
        })

    songs_json = json.dumps(songs_for_prompt, ensure_ascii=False)

    return f"""You are a music recommendation assistant writing brief explanations for songs.

User's request: "{user_intent}"
User's listening history: {history_summary}

These {k} songs were selected by combining keyword search (matching the user's request)
with a personalised ML ranker (matching the user's taste from their history):
{songs_json}

For each song, write ONE sentence (max 20 words) explaining why it fits the user's request.
Be specific — reference the user's own words or the genre/mood where possible.
Do not make up facts about songs you don't know.

Return JSON only:
{{"explanations": [{{"song_id": "...", "explanation": "..."}}]}}"""


def _call_groq_explain(prompt: str, api_key: str) -> Optional[str]:
    """
    POST to Groq chat completions and return the raw JSON response text.

    Groq uses an OpenAI-compatible API format.
    response_format: json_object forces clean JSON output.

    Returns None on repeated failure so the caller returns songs without
    explanations rather than crashing.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model":           GROQ_MODEL,
        "messages":        [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature":     0.3,   # low temperature = more consistent, less creative
    }

    for attempt in range(1, MAX_CHAT_RETRIES + 1):
        try:
            response = requests.post(
                GROQ_CHAT_URL, headers=headers, json=body, timeout=30
            )
        except requests.RequestException as exc:
            if attempt == MAX_CHAT_RETRIES:
                print(f"[rag_layer] Groq request failed: {exc}")
                return None
            time.sleep(CHAT_RETRY_DELAY)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            print(f"[rag_layer] HTTP {response.status_code}, "
                  f"retry {attempt}/{MAX_CHAT_RETRIES} ...")
            time.sleep(CHAT_RETRY_DELAY * attempt)
            continue

        if response.status_code != 200:
            print(f"[rag_layer] Groq error {response.status_code}: "
                  f"{response.text[:200]}")
            return None

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            print(f"[rag_layer] Unexpected Groq response structure: {exc}")
            return None

    return None


# ── output assembly ────────────────────────────────────────────────────────────

def _attach_metadata(
    ranked_songs: List[Dict],
    song_meta: Dict[str, Dict],
    explanations: Optional[Dict[str, str]],
    k: int,
) -> List[Dict]:
    """Join ML-ranked songs with title/artist metadata and Groq explanations."""
    output = []
    for song in ranked_songs[:k]:
        sid  = song.get("song_id", "")
        meta = song_meta.get(sid, {})
        output.append({
            "song_id":     sid,
            "title":       meta.get("title", "Unknown"),
            "artist":      meta.get("artist_name", "Unknown"),
            "score":       float(song.get("score", 0.0)),
            "explanation": (explanations or {}).get(sid, ""),
        })
    return output


# ── public API ─────────────────────────────────────────────────────────────────

def recommend(
    user_intent: str,
    user_profile: Dict,
    history_summary: str,
    api_key: str,
    k: int = DEFAULT_K,
) -> List[Dict]:
    """
    Full recommendation pipeline — called at every playlist refresh.

    With intent:
        1. TF-IDF search over 81k catalog → top-N song IDs      [RAG]
        2. ML-rank candidates by user's taste profile            [ranker]
        3. Groq writes one-sentence explanations for top-K       [LLM]

    Without intent (empty string):
        1. Profile-based candidate pool (freshness-aware)        [retriever]
        2. ML-rank → top-K                                       [ranker]
        (No explanations — no query to explain)

    Args:
        user_intent:     Free-text typed by the user.  Empty string triggers
                         the profile-based fallback.
        user_profile:    Dict from data/current_user.json.
        history_summary: e.g. "Liked 12 songs. Mostly chill, ambient."
        api_key:         GROQ_API_KEY from .env.
        k:               Songs to return (default 10).

    Returns:
        [{song_id, title, artist, score, explanation}]
        Falls back gracefully — always returns something or [].
    """
    # Deferred imports: load heavy modules only when this function is first called.
    from retrieval.retriever import build_candidate_pool
    from ranking.ranker import rank

    _load_knowledge_base()
    _load_songs()
    song_meta = _songs_by_id or {}

    # ── step 1: candidate songs ───────────────────────────────────────────────
    if user_intent.strip() and _tfidf_data is not None:
        # TF-IDF search over the full 81k catalog
        candidates = retrieve_songs(user_intent.strip(), top_n=RAG_CANDIDATE_COUNT)
        using_rag  = bool(candidates)
        if not candidates:
            print("[rag_layer] TF-IDF returned empty — falling back to profile retrieval.")
            candidates = build_candidate_pool(user_profile)
            using_rag  = False
    else:
        # No intent or index not built → profile-based retrieval
        candidates = build_candidate_pool(user_profile)
        using_rag  = False

    if not candidates:
        print("[rag_layer] Candidate pool is empty.")
        return []

    # ── step 2: ML ranking ────────────────────────────────────────────────────
    ranked = rank(candidates, user_profile, user_profile.get("user_id", ""))
    if not ranked:
        print("[rag_layer] ML ranker returned empty.")
        return []

    # ── step 3: Groq explanations (only when intent was given) ────────────────
    explanations: Optional[Dict[str, str]] = None

    if using_rag and user_intent.strip() and api_key:
        prompt   = _build_explanation_prompt(ranked, user_intent, history_summary, song_meta, k)
        raw_json = _call_groq_explain(prompt, api_key)

        if raw_json:
            try:
                parsed    = json.loads(raw_json)
                expl_list = parsed.get("explanations", [])
                explanations = {
                    item["song_id"]: item.get("explanation", "")
                    for item in expl_list
                    if isinstance(item, dict) and "song_id" in item
                }
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"[rag_layer] Explanation parse error: {exc}")

    # ── step 4: assemble output ───────────────────────────────────────────────
    return _attach_metadata(ranked, song_meta, explanations, k)
