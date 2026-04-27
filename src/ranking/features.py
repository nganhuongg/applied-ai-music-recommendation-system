"""
src/ranking/features.py

Compute a fixed-length feature vector for every song in the candidate pool.

The LightGBM ranker (ranker.py) takes this output as its training/inference input.
Each element in the returned list is one (user, song) pair represented as a dict.

Feature keys in each row dict:
  song_id              identifier (not a model input — kept for joining results)
  content_similarity   cosine(user_embedding, song_embedding)
  artist_similarity    cosine(user_artist_vector, artist_embedding); 0.0 if unavailable
  freshness            exp(-DECAY_LAMBDA * age_in_days)  — smooth exponential decay
  popularity           popularity_norm from songs_full.csv  (already in [0, 1])
  collaborative_score  mean cosine(candidate_embedding, liked_song_embedding)
  diversity_penalty    count of songs from same artist ranked above this one

All per-song cosine operations are vectorised with numpy (no Python loops per song).

If pandas is available the caller can convert with:
    import pandas as pd
    df = pd.DataFrame(compute_features(...))
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR         = REPO_ROOT / "data"
EMBEDDINGS_DIR   = REPO_ROOT / "embeddings"
SONGS_CSV        = DATA_DIR / "songs_full.csv"
INTERACTIONS_CSV = DATA_DIR / "interactions.csv"

# Exponential decay constant for the freshness feature (PLAN.md §5).
# At DECAY_LAMBDA=0.01:
#   100-day-old song → exp(-1.0)  ≈ 0.37
#   365-day-old song → exp(-3.65) ≈ 0.026
DECAY_LAMBDA = 0.01

# Column names in the order the feature matrix should be built for the ranker
FEATURE_COLUMNS = [
    "content_similarity",
    "artist_similarity",
    "freshness",
    "popularity",
    "collaborative_score",
    "diversity_penalty",
]


# ── data loading ──────────────────────────────────────────────────────────────

def _load_songs_by_id() -> Dict[str, Dict]:
    """Load songs_full.csv into a dict keyed by song_id for O(1) lookup."""
    songs = {}
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            songs[row["song_id"]] = row
    return songs


def _load_embeddings() -> Tuple[
    Optional[np.ndarray],   # song_embeddings   [N_songs   x D]
    Optional[np.ndarray],   # artist_embeddings [N_artists x D]
    Optional[Dict],         # shared id↔row-index mapping
]:
    """Load precomputed embeddings. Returns (None, None, None) if not yet built."""
    index_path    = EMBEDDINGS_DIR / "embedding_index.json"
    song_emb_path = EMBEDDINGS_DIR / "song_embeddings.npy"
    art_emb_path  = EMBEDDINGS_DIR / "artist_embeddings.npy"

    if not index_path.exists():
        return None, None, None

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    song_emb = np.load(str(song_emb_path)) if song_emb_path.exists() else None
    art_emb  = np.load(str(art_emb_path))  if art_emb_path.exists()  else None

    return song_emb, art_emb, index


def _load_user_embedding() -> Optional[np.ndarray]:
    """Load the behaviour-based user embedding saved by user_embedding.py."""
    path = DATA_DIR / "user_embedding.npy"
    return np.load(str(path)) if path.exists() else None


def _load_liked_song_ids(user_id: str) -> List[str]:
    """Return song_ids the given user liked (label=1) from interactions.csv."""
    if not INTERACTIONS_CSV.exists():
        return []

    liked = []
    with open(INTERACTIONS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("user_id") == user_id and row.get("label") == "1":
                liked.append(row["song_id"])
    return liked


# ── cosine similarity helpers ─────────────────────────────────────────────────

def _cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity: query (D,) vs every row of matrix (N x D).

    Returns shape (N,) in [-1, 1].  Zero-norm rows produce 0.0 (not NaN).
    """
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return np.zeros(len(matrix), dtype=np.float32)

    row_norms  = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)

    dots = matrix @ query                           # shape (N,)
    return (dots / (query_norm * safe_norms)).astype(np.float32)


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity: A (M x D), B (K x D) → result (M x K).

    result[i, j] = cosine(A[i], B[j]).

    Used for collaborative_score: A = candidate embeddings, B = liked embeddings.
    The result[i] row holds the similarity of candidate i to every liked song;
    taking the row-mean gives one collaborative score per candidate.
    """
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    A_unit = np.divide(A, A_norm, where=A_norm != 0.0, out=np.zeros_like(A))
    B_unit = np.divide(B, B_norm, where=B_norm != 0.0, out=np.zeros_like(B))

    return (A_unit @ B_unit.T).astype(np.float32)   # (M x K)


# ── individual feature computations ──────────────────────────────────────────

def _compute_content_similarity(
    candidate_indices: List[int],
    song_embeddings: np.ndarray,
    user_embedding: np.ndarray,
) -> np.ndarray:
    """Cosine(user_embedding, song_embedding) for each candidate."""
    sub_matrix = song_embeddings[candidate_indices]     # (N x D)
    return _cosine_batch(user_embedding, sub_matrix)


def _compute_artist_similarity(
    candidate_songs: List[Dict],
    artist_embeddings: np.ndarray,
    embedding_index: Dict,
    user_profile: Dict,
) -> np.ndarray:
    """
    Cosine(user_artist_vector, artist_embedding) per candidate.

    Returns all-zeros when user_profile has no artist_vector or when
    artist embeddings are missing — the ranker treats 0.0 as "no signal".

    Unlike content_similarity (one vector vs a sub-matrix), artist similarity
    requires a per-song lookup because each song points to a different artist
    row.  The loop here is over at most 200 candidates, not 81k songs.
    """
    n = len(candidate_songs)
    if "artist_vector" not in user_profile:
        return np.zeros(n, dtype=np.float32)

    user_artist_vec    = np.array(user_profile["artist_vector"], dtype=np.float32)
    artist_id_to_index = embedding_index.get("artist_id_to_index", {})

    scores = np.zeros(n, dtype=np.float32)
    for i, song in enumerate(candidate_songs):
        art_id  = song.get("artist_id", "")
        art_idx = artist_id_to_index.get(art_id)
        if art_idx is not None:
            art_vec    = artist_embeddings[int(art_idx)][np.newaxis, :]  # (1 x D)
            scores[i]  = float(_cosine_batch(user_artist_vec, art_vec)[0])
    return scores


def _compute_freshness(candidate_songs: List[Dict]) -> np.ndarray:
    """
    Exponential decay freshness: exp(-DECAY_LAMBDA * age_in_days).

    A song released today scores 1.0; one released 100 days ago scores ~0.37.
    Songs with missing / unparseable dates receive 0.0 (treated as very old).
    """
    now    = datetime.now()
    scores = np.zeros(len(candidate_songs), dtype=np.float32)
    for i, song in enumerate(candidate_songs):
        date_str = song.get("release_date", "").strip()
        try:
            release      = datetime.strptime(date_str, "%Y-%m-%d")
            age_days     = max(0, (now - release).days)
            scores[i]    = float(np.exp(-DECAY_LAMBDA * age_days))
        except ValueError:
            scores[i] = 0.0
    return scores


def _compute_popularity(candidate_songs: List[Dict]) -> np.ndarray:
    """
    Normalised popularity from songs_full.csv, already in [0, 1].
    Missing values default to 0.0.
    """
    scores = np.zeros(len(candidate_songs), dtype=np.float32)
    for i, song in enumerate(candidate_songs):
        try:
            scores[i] = float(song.get("popularity_norm", 0.0) or 0.0)
        except (ValueError, TypeError):
            scores[i] = 0.0
    return scores


def _compute_collaborative_score(
    candidate_indices: List[int],
    liked_indices: List[int],
    song_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Mean cosine similarity between each candidate song and all songs the user liked.

    Intuition: if a candidate sounds like things the user already liked,
    it should rank higher — this is item-item collaborative filtering.

    The computation is:
        score[i] = mean_j( cosine(candidate[i], liked[j]) )
                 = row-mean of (candidate_embeddings @ liked_embeddings.T)

    One matrix multiply covers all (candidate, liked) pairs at once, so this
    is O(N_candidates × D × N_liked) in compiled numpy, not Python loops.

    Returns all-zeros when the user has no liked songs yet (cold-start).
    """
    if not liked_indices:
        return np.zeros(len(candidate_indices), dtype=np.float32)

    candidate_emb = song_embeddings[candidate_indices]  # (N x D)
    liked_emb     = song_embeddings[liked_indices]      # (K x D)

    sim_matrix = _cosine_matrix(candidate_emb, liked_emb)  # (N x K)
    return sim_matrix.mean(axis=1)                          # (N,)


def _compute_diversity_penalty(candidate_songs: List[Dict]) -> np.ndarray:
    """
    Count of songs from the same artist that appear earlier in the candidate list.

    A score of 0 means this is the first song from this artist.
    A score of 2 means two songs by the same artist already appear above this one.
    The ranker learns to down-weight high-penalty songs to reduce repetition.
    """
    artist_count: Dict[str, int] = {}
    scores = np.zeros(len(candidate_songs), dtype=np.float32)
    for i, song in enumerate(candidate_songs):
        artist_id = song.get("artist_id", "")
        count     = artist_count.get(artist_id, 0)
        scores[i] = float(count)
        artist_count[artist_id] = count + 1
    return scores


# ── public API ────────────────────────────────────────────────────────────────

def compute_features(
    candidate_song_ids: List[str],
    user_profile: Dict,
    user_id: str,
) -> List[Dict]:
    """
    Build the feature rows for the candidate pool.

    Args:
        candidate_song_ids: Output of retriever.build_candidate_pool() — ordered list.
        user_profile:       User's profile dict (from current_user.json).
        user_id:            User's ID string, used to look up their liked songs.

    Returns:
        List of dicts, one per candidate song, with keys:
            song_id, content_similarity, artist_similarity, freshness,
            popularity, collaborative_score, diversity_penalty

        Songs not found in songs_full.csv are silently dropped.
        Returns an empty list when candidate_song_ids is empty.

    Converting to a DataFrame (when pandas is available):
        import pandas as pd
        df = pd.DataFrame(compute_features(...))

    Extracting a numpy feature matrix for LightGBM:
        rows    = compute_features(...)
        X       = feature_matrix(rows)   # see helper below
        song_ids = [r["song_id"] for r in rows]
    """
    if not candidate_song_ids:
        return []

    songs_by_id = _load_songs_by_id()

    # Filter to songs present in the CSV; preserve candidate order
    candidate_songs: List[Dict] = []
    valid_song_ids:  List[str]  = []
    for sid in candidate_song_ids:
        if sid in songs_by_id:
            candidate_songs.append(songs_by_id[sid])
            valid_song_ids.append(sid)

    if not candidate_songs:
        return []

    song_embeddings, artist_embeddings, embedding_index = _load_embeddings()
    user_embedding = _load_user_embedding()
    liked_song_ids = _load_liked_song_ids(user_id)

    song_id_to_index: Dict = {}
    if embedding_index is not None:
        song_id_to_index = embedding_index.get("song_id_to_index", {})

    # For embedding features, only songs with a known index can be scored.
    # For non-embedding features (freshness, popularity, diversity_penalty),
    # all valid candidate songs are used — so rows are never dropped just
    # because precompute.py hasn't been run yet.
    candidate_indices = [
        int(song_id_to_index[sid]) for sid in valid_song_ids if sid in song_id_to_index
    ]
    indexed_mask = [sid in song_id_to_index for sid in valid_song_ids]

    liked_indices = [
        int(song_id_to_index[sid])
        for sid in liked_song_ids
        if sid in song_id_to_index
    ]

    n = len(valid_song_ids)
    zeros = np.zeros(n, dtype=np.float32)

    # content_similarity — requires user embedding + song embeddings
    if song_embeddings is not None and user_embedding is not None and candidate_indices:
        # _compute_content_similarity returns scores only for indexed songs;
        # map them back to the full valid_song_ids list.
        indexed_scores = _compute_content_similarity(
            candidate_indices, song_embeddings, user_embedding
        )
        content_sim = zeros.copy()
        idx = 0
        for i, has_index in enumerate(indexed_mask):
            if has_index:
                content_sim[i] = indexed_scores[idx]
                idx += 1
    else:
        content_sim = zeros.copy()

    # artist_similarity — requires artist embeddings + artist_vector in profile
    if artist_embeddings is not None and embedding_index is not None:
        artist_sim = _compute_artist_similarity(
            candidate_songs, artist_embeddings, embedding_index, user_profile
        )
    else:
        artist_sim = zeros.copy()

    freshness         = _compute_freshness(candidate_songs)
    popularity        = _compute_popularity(candidate_songs)

    # collaborative_score — requires song embeddings + at least one liked song
    if song_embeddings is not None and candidate_indices:
        indexed_collab = _compute_collaborative_score(
            candidate_indices, liked_indices, song_embeddings
        )
        collab_score = zeros.copy()
        idx = 0
        for i, has_index in enumerate(indexed_mask):
            if has_index:
                collab_score[i] = indexed_collab[idx]
                idx += 1
    else:
        collab_score = zeros.copy()

    diversity_penalty = _compute_diversity_penalty(candidate_songs)

    # Assemble one dict per song
    rows = []
    for i, sid in enumerate(valid_song_ids):
        rows.append({
            "song_id":             sid,
            "content_similarity":  float(content_sim[i]),
            "artist_similarity":   float(artist_sim[i]),
            "freshness":           float(freshness[i]),
            "popularity":          float(popularity[i]),
            "collaborative_score": float(collab_score[i]),
            "diversity_penalty":   float(diversity_penalty[i]),
        })

    return rows


def feature_matrix(rows: List[Dict]) -> np.ndarray:
    """
    Convert the list of feature dicts to a numpy array for LightGBM.

    Returns shape (N, len(FEATURE_COLUMNS)).
    Column order matches FEATURE_COLUMNS — use that list as column names
    when logging or inspecting feature importances.
    """
    if not rows:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)

    return np.array(
        [[row[col] for col in FEATURE_COLUMNS] for row in rows],
        dtype=np.float32,
    )
