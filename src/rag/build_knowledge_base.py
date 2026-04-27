"""
src/rag/build_knowledge_base.py

Run ONCE offline to build the TF-IDF search index over all 81,343 songs.
Re-run whenever songs_full.csv changes.

What this script does:
  1. Reads all songs from songs_full.csv
  2. Generates a rich text description for each song from its audio features
  3. Fits a TF-IDF vectorizer and transforms all descriptions
  4. Saves three files that rag_layer.py loads at inference time:
       embeddings/knowledge_base_tfidf.npz     — sparse TF-IDF matrix [81343 × vocab]
       embeddings/knowledge_base_vectorizer.pkl — fitted TF-IDF vectorizer
       embeddings/knowledge_base_docs.json      — parallel list of {song_id, title, artist, text}

Usage:
    python src/rag/build_knowledge_base.py

No API key required — TF-IDF runs entirely on your machine with sklearn.

Why TF-IDF instead of a dense embedding API:
    Gemini / other embedding APIs are rate-limited and require network calls.
    TF-IDF (Term Frequency–Inverse Document Frequency) is a proven text retrieval
    technique that builds in seconds and has no rate limits.

    How it works: for a query like "calm jazz piano for studying", TF-IDF finds
    songs whose descriptions contain rare matching terms (jazz, calm, studying,
    piano) — terms that appear in many documents get down-weighted automatically
    so common words don't dominate the score.

    With descriptions written to include genre, mood, tempo labels, era, and use-case
    phrases, TF-IDF correctly surfaces jazz songs for "jazz", calm songs for "calm",
    and instrumental songs for "no lyrics" — without any API call.

Why this is still real RAG:
    The 81k song catalog is too large to fit in any LLM context window.  TF-IDF
    search over the pre-built index is the ONLY way to narrow 81k songs to a
    relevant shortlist from a free-text query.  The retrieval step is essential.
"""

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
DATA_DIR       = REPO_ROOT / "data"
SONGS_CSV      = DATA_DIR / "songs_full.csv"

# The TF-IDF sparse matrix is stored as raw CSR arrays (no scipy needed):
#   knowledge_base_tfidf.npz  ← np.savez with keys: data, indices, indptr, shape
TFIDF_MATRIX_OUT  = EMBEDDINGS_DIR / "knowledge_base_tfidf.npz"
VECTORIZER_OUT    = EMBEDDINGS_DIR / "knowledge_base_vectorizer.pkl"
DOCS_OUT          = EMBEDDINGS_DIR / "knowledge_base_docs.json"


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    try:
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0


def _describe_energy(energy: float) -> str:
    if energy >= 0.8:  return "very high energy"
    if energy >= 0.6:  return "high energy"
    if energy >= 0.4:  return "moderate energy"
    if energy >= 0.2:  return "low energy"
    return "very calm"


def _describe_valence(valence: float) -> str:
    if valence >= 0.7:  return "very upbeat"
    if valence >= 0.5:  return "positive"
    if valence >= 0.3:  return "neutral"
    return "melancholy"


def _describe_popularity(pop_norm: float) -> str:
    if pop_norm >= 0.8:  return "very popular"
    if pop_norm >= 0.6:  return "popular"
    if pop_norm >= 0.4:  return "moderately popular"
    return "niche underground"


def _suggest_use_cases(
    energy: float,
    instrumentalness: float,
    valence: float,
    tempo: float,
    speechiness: float,
) -> str:
    cases = []
    if energy >= 0.7 and tempo >= 120:
        cases.append("working out exercise gym")
    if energy <= 0.5 and instrumentalness >= 0.4 and speechiness < 0.1:
        cases.append("studying focus concentration")
    if valence >= 0.6 and energy >= 0.5:
        cases.append("uplifting happy cheerful")
    if valence <= 0.35 and energy <= 0.5:
        cases.append("melancholy sad reflection emotional")
    if energy <= 0.35 and instrumentalness >= 0.5:
        cases.append("sleeping relaxing meditation calm")
    if energy >= 0.6 and valence >= 0.5:
        cases.append("party dancing social")
    if not cases:
        cases.append("general listening")
    return " ".join(cases)


def _release_era(release_date: str) -> str:
    try:
        year = int(release_date[:4])
        decade = (year // 10) * 10
        return f"{decade}s"
    except (ValueError, TypeError, IndexError):
        return ""


# ── data loading ───────────────────────────────────────────────────────────────

def _load_songs() -> List[Dict]:
    songs = []
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            songs.append(row)
    return songs


# ── description generator ──────────────────────────────────────────────────────

def generate_song_descriptions(songs: List[Dict]) -> List[Dict]:
    """
    Generate a rich text description for every song in the catalog.

    The description is designed for TF-IDF keyword search.  It packs in the
    words a user is most likely to type: genre names, mood words, era, tempo
    descriptions, use-case phrases, and whether the track has no lyrics.

    Example output:
        "Clair de Lune" by Claude Debussy — classical, 1890s. Very calm (0.03),
        56 BPM, melancholy (valence 0.16). Mostly instrumental no lyrics.
        Niche underground. Good for: studying focus concentration sleeping
        relaxing meditation calm.

    Doc format returned:
        {"song_id": "...", "title": "...", "artist": "...", "text": "..."}
    """
    docs = []
    for song in songs:
        sid = song.get("song_id", "").strip()
        if not sid:
            continue

        title    = song.get("title", "Unknown").strip()
        artist   = song.get("artist_name", "Unknown").strip().split(";")[0].strip()
        genre    = song.get("genre", "").strip()
        mood     = song.get("mood", "").strip()
        release  = song.get("release_date", "").strip()

        energy           = _safe_float(song.get("energy"))
        tempo            = _safe_float(song.get("tempo_bpm"))
        valence          = _safe_float(song.get("valence"))
        instrumentalness = _safe_float(song.get("instrumentalness"))
        speechiness      = _safe_float(song.get("speechiness"))
        pop_norm         = _safe_float(song.get("popularity_norm"))

        header = f'"{title}" by {artist}'
        if genre:
            header += f" — {genre}"
        if mood:
            header += f" {mood}"
        era = _release_era(release)
        if era:
            header += f" {era}"

        energy_desc  = _describe_energy(energy)
        valence_desc = _describe_valence(valence)
        pop_desc     = _describe_popularity(pop_norm)
        use_cases    = _suggest_use_cases(
            energy, instrumentalness, valence, tempo, speechiness
        )

        details = (
            f"{energy_desc} {tempo:.0f} BPM "
            f"{valence_desc} valence {valence:.2f}"
        )

        # "no lyrics" / "instrumental" are high-value search terms — include both
        if instrumentalness >= 0.5:
            details += " instrumental no lyrics"

        text = f"{header}. {details}. {pop_desc}. Good for {use_cases}."

        docs.append({
            "song_id": sid,
            "title":   title,
            "artist":  artist,
            "text":    text,
        })

    return docs


# ── TF-IDF build ───────────────────────────────────────────────────────────────

def build_tfidf_index(docs: List[Dict]) -> tuple:
    """
    Fit a TF-IDF vectorizer on all song descriptions and return the matrix.

    TF-IDF parameters:
        max_features=30_000  — vocabulary cap; covers all meaningful music terms
        ngram_range=(1, 2)   — unigrams + bigrams ("no lyrics", "working out")
        sublinear_tf=True    — log-scale term frequency; reduces effect of very
                               common terms appearing many times in one document
        norm='l2'            — L2-normalise each row so dot product = cosine sim

    Returns:
        (vectorizer, tfidf_matrix)
        tfidf_matrix is scipy CSR sparse matrix of shape [N_songs × vocab_size]
    """
    texts = [doc["text"] for doc in docs]

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",          # each row is unit-length → dot product = cosine sim
    )

    print(f"  Fitting TF-IDF on {len(texts):,} descriptions ...")
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"  Matrix shape: {tfidf_matrix.shape}  "
          f"(songs × vocab)  nnz={tfidf_matrix.nnz:,}")

    return vectorizer, tfidf_matrix


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Building RAG song catalog (TF-IDF, all 81k songs)")
    print("=" * 60)
    print("\nNo API key required — runs entirely on your machine.\n")

    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    # 1. load songs
    print("[1/4] Loading songs_full.csv ...")
    songs = _load_songs()
    print(f"  {len(songs):,} songs loaded.")

    # 2. generate descriptions
    print(f"\n[2/4] Generating text descriptions ...")
    docs = generate_song_descriptions(songs)
    print(f"  {len(docs):,} descriptions generated.")

    # 3. build TF-IDF index
    print(f"\n[3/4] Building TF-IDF index ...")
    vectorizer, tfidf_matrix = build_tfidf_index(docs)

    # 4. save
    print(f"\n[4/4] Saving to {EMBEDDINGS_DIR}/ ...")

    # Store the CSR sparse matrix as three plain numpy arrays so scipy is not
    # needed at load time.  np.savez packs multiple arrays into one .npz file.
    np.savez(
        str(TFIDF_MATRIX_OUT),
        data    = tfidf_matrix.data,
        indices = tfidf_matrix.indices,
        indptr  = tfidf_matrix.indptr,
        shape   = np.array(tfidf_matrix.shape, dtype=np.int64),
    )

    with open(VECTORIZER_OUT, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(DOCS_OUT, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"  knowledge_base_tfidf.npz      : {tfidf_matrix.shape} (CSR arrays)")
    print(f"  knowledge_base_vectorizer.pkl : fitted on {len(docs):,} songs")
    print(f"  knowledge_base_docs.json      : {len(docs):,} songs")
    print("\nDone. Run the app — the RAG layer loads these files automatically.")


if __name__ == "__main__":
    main()
