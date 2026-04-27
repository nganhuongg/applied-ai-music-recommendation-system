"""
src/retrieval/retriever.py

Build a candidate pool of fresh, relevant songs for each session refresh.

Two retrieval strategies are combined (union):
  1. Embedding similarity  — cosine(user_embedding, song_embedding) over fresh songs
  2. Artist-based          — songs from favorite / similar artists in the fresh window

"Fresh" means released within FRESHNESS_WINDOW_DAYS of today.  Because the
songs_full.csv dataset mostly pre-dates 2024, the window is relaxed through
FALLBACK_WINDOWS until at least MIN_FRESH_SONGS are available.

All per-song cosine similarity is vectorised over numpy arrays (no Python
loop per song), so the fresh pool of thousands of songs is scored in one
matrix–vector multiply.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR       = REPO_ROOT / "data"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
SONGS_CSV      = DATA_DIR / "songs_full.csv"

# Primary freshness window (days).  Songs outside this window are excluded.
FRESHNESS_WINDOW_DAYS = 90

# If fewer than this many songs pass the freshness filter, relax the window
# through FALLBACK_WINDOWS until enough candidates are found.
MIN_FRESH_SONGS = 50

# Progressive fallback windows tried when the primary window is too narrow.
# The final sentinel (9999) means "use all songs regardless of date".
FALLBACK_WINDOWS = [180, 365, 730, 9999]

# Returned pool will have at most this many song_ids.
TARGET_MAX = 200

# How many of the most-similar artists to pull songs from (embedding path).
TOP_SIMILAR_ARTISTS = 10


# ── data loading ──────────────────────────────────────────────────────────────

def _load_songs() -> List[Dict]:
    """Load songs_full.csv as a list of plain dicts (one per song)."""
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _load_embeddings() -> Tuple[
    Optional[np.ndarray],  # song_embeddings  [N_songs   x D]
    Optional[np.ndarray],  # artist_embeddings [N_artists x D]
    Optional[Dict],        # embedding_index  (shared id↔row maps for both)
]:
    """
    Load precomputed embeddings and the shared id↔row-index mapping.

    Both embedding files and the index live in embeddings/.  The index JSON
    carries four keys: song_id_to_index, index_to_song_id,
    artist_id_to_index, index_to_artist_id.

    Returns (None, None, None) if the index file is missing (i.e. precompute.py
    has not been run yet) — callers fall back to name-match mode.
    """
    index_path    = EMBEDDINGS_DIR / "embedding_index.json"
    song_emb_path = EMBEDDINGS_DIR / "song_embeddings.npy"
    art_emb_path  = EMBEDDINGS_DIR / "artist_embeddings.npy"

    if not index_path.exists():
        return None, None, None

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    song_embeddings   = np.load(str(song_emb_path)) if song_emb_path.exists() else None
    artist_embeddings = np.load(str(art_emb_path))  if art_emb_path.exists()  else None

    return song_embeddings, artist_embeddings, index


def _load_user_embedding() -> Optional[np.ndarray]:
    """Load the user's behaviour embedding saved by user_embedding.py."""
    path = DATA_DIR / "user_embedding.npy"
    return np.load(str(path)) if path.exists() else None


# ── freshness filtering ───────────────────────────────────────────────────────

def _filter_by_window(songs: List[Dict], window_days: int) -> List[Dict]:
    """Return songs released within window_days of today."""
    cutoff = datetime.now() - timedelta(days=window_days)
    fresh  = []
    for song in songs:
        date_str = song.get("release_date", "").strip()
        try:
            if datetime.strptime(date_str, "%Y-%m-%d") >= cutoff:
                fresh.append(song)
        except ValueError:
            pass  # skip songs with missing / unparseable release_date
    return fresh


def _get_fresh_songs(songs: List[Dict]) -> List[Dict]:
    """
    Return fresh songs, relaxing the time window if too few pass the filter.

    Tries FRESHNESS_WINDOW_DAYS first, then each entry in FALLBACK_WINDOWS.
    The last sentinel (9999) returns all songs, guaranteeing a non-empty pool
    even when the dataset pre-dates the freshness window entirely.
    """
    for window in [FRESHNESS_WINDOW_DAYS] + FALLBACK_WINDOWS:
        if window >= 9999:
            return songs

        fresh = _filter_by_window(songs, window)
        if len(fresh) >= MIN_FRESH_SONGS:
            if window != FRESHNESS_WINDOW_DAYS:
                print(
                    f"[retriever] freshness window relaxed to {window} days "
                    f"({len(fresh):,} songs pass)"
                )
            return fresh

    return songs  # unreachable but satisfies type checker


# ── cosine similarity helper ──────────────────────────────────────────────────

def _cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity: query (D,) vs every row of matrix (N x D).

    Returns an array of shape (N,) with values in [-1, 1].

    Instead of computing similarity one row at a time, this treats the whole
    matrix as a single operation:
        similarity[i] = dot(query, matrix[i]) / (||query|| * ||matrix[i]||)
    numpy broadcasts the division across all N rows at once, making it
    ~100× faster than a Python loop for large matrices.
    """
    query_norm  = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return np.zeros(len(matrix), dtype=np.float32)

    row_norms = np.linalg.norm(matrix, axis=1)       # shape (N,)
    # Replace zero norms with 1.0 so division produces 0.0 (dot=0) not NaN
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)

    dots = matrix @ query                             # shape (N,)
    return (dots / (query_norm * safe_norms)).astype(np.float32)


# ── retrieval strategies ──────────────────────────────────────────────────────

def _similarity_pool(
    fresh_songs: List[Dict],
    user_embedding: np.ndarray,
    song_embeddings: np.ndarray,
    song_id_to_index: Dict,
    top_n: int,
) -> List[str]:
    """
    Return the top_n fresh songs most similar to the user embedding.

    Only songs that have a row in song_embeddings (i.e. their song_id appears
    in the index) are scored.  Others are silently skipped — this can happen
    if songs_full.csv was updated after the last precompute.py run.
    """
    # Pair each fresh song with its embedding row index (skip unmapped songs)
    indexed: List[Tuple[str, int]] = []
    for song in fresh_songs:
        sid = song.get("song_id", "")
        idx = song_id_to_index.get(sid)
        if idx is not None:
            indexed.append((sid, int(idx)))

    if not indexed:
        return []

    row_indices = [idx for _, idx in indexed]
    sub_matrix  = song_embeddings[row_indices]        # shape (M x D)

    similarities = _cosine_batch(user_embedding, sub_matrix)

    # np.argpartition finds the top-k indices without a full sort (O(N) vs O(N log N))
    k = min(top_n, len(similarities))
    top_local = np.argpartition(similarities, -k)[-k:]
    # Sort those k indices by similarity descending for a ranked result
    top_local = top_local[np.argsort(similarities[top_local])[::-1]]

    return [indexed[i][0] for i in top_local]


def _artist_pool(
    fresh_songs: List[Dict],
    user_profile: Dict,
    artist_embeddings: Optional[np.ndarray],
    embedding_index: Optional[Dict],
) -> List[str]:
    """
    Return fresh songs whose artist is similar to the user's preferences.

    Mode 1 — embedding path (requires artist_embeddings + artist_vector in profile):
        Compute cosine(user_profile["artist_vector"], artist_embeddings)
        → take the top TOP_SIMILAR_ARTISTS artist_ids
        → collect their songs from the fresh pool

    Mode 2 — name-match fallback (no embeddings or no artist_vector):
        Collect songs whose artist_name overlaps with the user's
        favorite_artists or recent_artists lists.

    The name-match fallback is intentionally broad: it checks if ANY artist in
    a semicolon-separated artist_name matches the user's known artists, so
    collaboration tracks are caught correctly.
    """
    # Embedding-based artist similarity
    if (
        artist_embeddings is not None
        and embedding_index is not None
        and "artist_vector" in user_profile
    ):
        artist_id_to_index = embedding_index.get("artist_id_to_index", {})
        index_to_artist_id = embedding_index.get("index_to_artist_id", {})

        user_artist_vec = np.array(user_profile["artist_vector"], dtype=np.float32)
        similarities    = _cosine_batch(user_artist_vec, artist_embeddings)

        k = min(TOP_SIMILAR_ARTISTS, len(similarities))
        top_local       = np.argpartition(similarities, -k)[-k:]
        similar_art_ids = {
            index_to_artist_id[str(i)]
            for i in top_local
            if str(i) in index_to_artist_id
        }

        return [
            song["song_id"]
            for song in fresh_songs
            if song.get("artist_id", "") in similar_art_ids
        ]

    # Name-match fallback
    known_names = (
        set(user_profile.get("favorite_artists", []))
        | set(user_profile.get("recent_artists",   []))
    )
    result = []
    for song in fresh_songs:
        song_artists = {
            name.strip() for name in song.get("artist_name", "").split(";")
        }
        if song_artists & known_names:
            result.append(song["song_id"])
    return result


# ── public API ────────────────────────────────────────────────────────────────

def build_candidate_pool(
    user_profile: Dict,
    target_pool_size: int = TARGET_MAX,
) -> List[str]:
    """
    Build a candidate pool of fresh, relevant songs for the ranking step.

    The pool is the union of:
      • Embedding-similarity pool  — top (target_pool_size // 2) fresh songs
        by cosine distance to the user's behaviour embedding
      • Artist pool  — fresh songs from artists similar to the user's favourites

    Returns a deduplicated list of song_ids with at most target_pool_size entries.
    The similarity pool is listed first so the ranking step can use list order
    as a cheap tie-breaker.

    Returns an empty list when:
      - songs_full.csv is missing
      - no songs pass the freshness filter (including all fallback windows)

    Args:
        user_profile:     User profile dict (fields from current_user.json).
        target_pool_size: Maximum number of song_ids returned.  Capped at TARGET_MAX.
    """
    target_pool_size = min(target_pool_size, TARGET_MAX)

    songs       = _load_songs()
    fresh_songs = _get_fresh_songs(songs)

    if not fresh_songs:
        return []

    song_embeddings, artist_embeddings, embedding_index = _load_embeddings()
    user_embedding = _load_user_embedding()

    # Strategy 1: embedding similarity
    similarity_ids: List[str] = []
    if user_embedding is not None and song_embeddings is not None and embedding_index is not None:
        song_id_to_index = embedding_index.get("song_id_to_index", {})
        top_n = max(MIN_FRESH_SONGS, target_pool_size // 2)
        similarity_ids = _similarity_pool(
            fresh_songs, user_embedding, song_embeddings, song_id_to_index, top_n
        )

    # Strategy 2: artist-based
    artist_ids = _artist_pool(
        fresh_songs, user_profile, artist_embeddings, embedding_index
    )

    # Union: similarity pool (relevance-ranked) first, then artist additions
    seen: set = set(similarity_ids)
    combined   = list(similarity_ids)
    for sid in artist_ids:
        if sid not in seen:
            seen.add(sid)
            combined.append(sid)

    return combined[:target_pool_size]
