"""
src/recommender/initial.py

Cold-start recommender — generates first recommendations before any user
listening history exists.

Two scoring formulas are used:

  Familiar songs (known artist):
      score = 0.5 * artist_similarity
            + 0.3 * genre_similarity
            + 0.2 * popularity

  Discovery songs (new artist):
      score = 0.40 * genre_similarity
            + 0.35 * mood_compatibility   ← energy + valence vs user targets
            + 0.25 * popularity

This module intentionally avoids loading the trained ranking model so it works
on the very first run, before any ML training has happened.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── paths ───────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR       = REPO_ROOT / "data"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"
SONGS_CSV      = DATA_DIR / "songs_full.csv"

# ── scoring weights ───────────────────────────────────────────────────────────────
# Familiar pool (known artist)
ARTIST_WEIGHT     = 0.5
GENRE_WEIGHT      = 0.3
POPULARITY_WEIGHT = 0.2

# Discovery pool (new artist) — no artist term; mood replaces it
DISC_GENRE_WEIGHT  = 0.40
DISC_MOOD_WEIGHT   = 0.35
DISC_POP_WEIGHT    = 0.25

# Max songs from the same artist allowed in the final top-N list
MAX_SONGS_PER_ARTIST = 1

# How many candidates to draw from each pool before applying diversity.
# With CANDIDATE_MULTIPLIER=5 and top_n=10, we evaluate 50 familiar and
# 50 discovery songs before selecting the final diverse 10.
# This gives enough room to find distinct artists even for popular artists
# who have many songs (e.g. Taylor Swift may occupy 30 slots in a raw top-50).
CANDIDATE_MULTIPLIER = 5

# ── genre vocabulary ─────────────────────────────────────────────────────────────
# This list must stay in sync with GENRE_VOCABULARY in src/profile/survey.py.
# It is duplicated here to avoid importing Streamlit as a side-effect of that file.
GENRE_VOCABULARY: List[str] = sorted([
    "acoustic", "alternative", "ambient", "anime", "blues",
    "chill", "classical", "country", "dance", "edm",
    "electronic", "folk", "funk", "gospel", "hip-hop",
    "house", "indie", "indie-pop", "j-pop", "jazz",
    "k-pop", "latin", "metal", "pop", "punk",
    "r-n-b", "reggae", "rock", "soul", "synth-pop",
])

# Pre-built reverse mapping: genre string → index in GENRE_VOCABULARY (O(1) lookup)
_GENRE_INDEX: Dict[str, int] = {genre: idx for idx, genre in enumerate(GENRE_VOCABULARY)}


# ── data loading ─────────────────────────────────────────────────────────────────

def load_songs() -> List[Dict]:
    """Load every row from songs_full.csv as a plain Python dict."""
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_artist_embeddings() -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Try to load precomputed artist embeddings from the embeddings/ directory.

    Returns (matrix, index_dict) if both files exist, else (None, None).
    These files are produced by PLAN.md Step 3 (run on Google Colab).
    If they do not exist yet, artist similarity falls back to name matching.
    """
    emb_path   = EMBEDDINGS_DIR / "artist_embeddings.npy"
    index_path = EMBEDDINGS_DIR / "embedding_index.json"

    if not emb_path.exists() or not index_path.exists():
        return None, None

    embeddings = np.load(str(emb_path))
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return embeddings, index


# ── similarity helpers ────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] range. Returns 0.0 if either vector is zero."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_artist_similarity(
    user_profile: Dict,
    song: Dict,
    artist_embeddings: Optional[np.ndarray] = None,
    embedding_index: Optional[Dict] = None,
) -> float:
    """
    Score how well the song artist matches the user's artist preferences.

    Mode 1 — with embeddings (available after PLAN.md Step 3):
        cosine(user_profile["artist_vector"], artist_embeddings[song.artist_id])

    Mode 2 — name-match fallback (works on first run, no embeddings needed):
        1.0  if song artist is in user's favorite_artists
        0.7  if song artist is in user's recent_artists
        0.0  otherwise

    The song's artist_name field may contain multiple artists separated by ";".
    Any one of them matching is enough to trigger the higher score.
    """
    # Embedding-based path: requires embeddings AND artist_vector in profile
    if (
        artist_embeddings is not None
        and embedding_index is not None
        and "artist_vector" in user_profile
    ):
        artist_id_to_idx = embedding_index.get("artist_id_to_index", {})
        song_artist_id   = song.get("artist_id", "")

        if song_artist_id in artist_id_to_idx:
            row_idx         = artist_id_to_idx[song_artist_id]
            song_artist_vec = artist_embeddings[row_idx]
            user_artist_vec = np.array(user_profile["artist_vector"])
            return cosine_similarity(user_artist_vec, song_artist_vec)

    # Name-match fallback
    favorite_artists = set(user_profile.get("favorite_artists", []))
    recent_artists   = set(user_profile.get("recent_artists", []))

    song_artists = {
        name.strip() for name in song.get("artist_name", "").split(";")
    }

    if song_artists & favorite_artists:
        return 1.0
    if song_artists & recent_artists:
        return 0.7
    return 0.0


def compute_genre_similarity(user_profile: Dict, song: Dict) -> float:
    """
    Dot product of the user's genre_vector and the song's genre one-hot encoding.

    genre_vector is a multi-hot list (one float per genre in GENRE_VOCABULARY).
    Each song has a single genre string. The result is the weight the user placed
    on that genre: 1.0 if it is a favourite genre, 0.0 if not (or unknown genre).
    """
    genre_vector = user_profile.get("genre_vector", [])
    song_genre   = song.get("genre", "").strip().lower()

    genre_idx = _GENRE_INDEX.get(song_genre)
    if genre_idx is None or genre_idx >= len(genre_vector):
        # Song genre is not in the known vocabulary
        return 0.0

    return float(genre_vector[genre_idx])


def compute_mood_compatibility(user_profile: Dict, song: Dict) -> float:
    """
    How closely the song's energy and valence match the user's mood targets.

    The user's mood_targets are derived from their survey answers, e.g.:
        Calm      → target_energy=0.30, target_valence=0.50
        Energetic → target_energy=0.80, target_valence=0.70

    We measure the average absolute distance between the song's audio features
    and those targets, then convert to a similarity score (1.0 = perfect match).
    """
    targets = user_profile.get("mood_targets", {})
    target_energy  = float(targets.get("target_energy",  0.5))
    target_valence = float(targets.get("target_valence", 0.5))

    try:
        song_energy  = float(song.get("energy",  0.5) or 0.5)
        song_valence = float(song.get("valence", 0.5) or 0.5)
    except (ValueError, TypeError):
        return 0.0

    energy_dist  = abs(song_energy  - target_energy)
    valence_dist = abs(song_valence - target_valence)
    avg_dist = (energy_dist + valence_dist) / 2.0
    return round(1.0 - avg_dist, 4)


def score_song(
    user_profile: Dict,
    song: Dict,
    artist_embeddings: Optional[np.ndarray] = None,
    embedding_index: Optional[Dict] = None,
) -> float:
    """
    Apply the cold-start scoring formula from PLAN.md Step 2.

        score = 0.5 * artist_similarity
              + 0.3 * genre_similarity
              + 0.2 * popularity
    """
    artist_sim = compute_artist_similarity(
        user_profile, song, artist_embeddings, embedding_index
    )
    genre_sim = compute_genre_similarity(user_profile, song)

    try:
        popularity = float(song.get("popularity_norm", 0.0))
    except (ValueError, TypeError):
        popularity = 0.0

    return (
        ARTIST_WEIGHT     * artist_sim
        + GENRE_WEIGHT    * genre_sim
        + POPULARITY_WEIGHT * popularity
    )


def score_discovery_song(user_profile: Dict, song: Dict) -> float:
    """
    Score a discovery song (unknown artist).

    Artist similarity is intentionally excluded — these are songs from artists
    the user has never heard of, so we score by how well the song fits their
    taste in other dimensions:

        score = 0.40 * genre_similarity     ← does it match their favourite genres?
              + 0.35 * mood_compatibility   ← does the energy/vibe match?
              + 0.25 * popularity           ← is it broadly well-regarded?

    Weighting genre and mood higher than popularity means a niche song that
    perfectly matches the user's vibe will beat a popular song that doesn't.
    """
    genre_sim   = compute_genre_similarity(user_profile, song)
    mood_compat = compute_mood_compatibility(user_profile, song)

    try:
        popularity = float(song.get("popularity_norm", 0.0) or 0.0)
    except (ValueError, TypeError):
        popularity = 0.0

    return round(
        DISC_GENRE_WEIGHT * genre_sim
        + DISC_MOOD_WEIGHT  * mood_compat
        + DISC_POP_WEIGHT   * popularity,
        4,
    )


# ── diversity helper ──────────────────────────────────────────────────────────────

def _pick_diverse(pool: List[Dict], n: int, strict: bool = False) -> List[Dict]:
    """
    Pick the top n songs from pool while respecting MAX_SONGS_PER_ARTIST.

    pool must already be sorted by score (descending).

    We check ALL artists in a collab song (e.g. "Dua Lipa;BLACKPINK"), not just
    the first-listed one. This prevents a collaboration from slipping through
    after one of its artists was already selected under their own song.

    strict=False (default): if the pool runs out of distinct artists before n
    slots are filled, the fallback relaxes the constraint and fills with whatever
    songs are left — you always get n results.

    strict=True: stop as soon as the diverse candidates run out and return
    whatever was found. The caller is responsible for routing the shortfall
    elsewhere (e.g. giving unfilled familiar slots to the discovery pool).
    """
    counts: Dict[str, int] = {}
    result: List[Dict] = []

    for song in pool:
        all_artists = {a.strip() for a in song["artist_name"].split(";")}
        # Skip this song if any of its artists already hit the per-artist cap
        if any(counts.get(a, 0) >= MAX_SONGS_PER_ARTIST for a in all_artists):
            continue
        result.append(song)
        for a in all_artists:
            counts[a] = counts.get(a, 0) + 1
        if len(result) == n:
            return result

    if strict:
        # Caller handles the shortfall — don't repeat artists to pad the list
        return result

    # Fallback: pool doesn't have n distinct-artist songs — fill without limit
    shown_ids = {s["song_id"] for s in result}
    for song in pool:
        if song["song_id"] not in shown_ids:
            result.append(song)
        if len(result) == n:
            break

    return result


# ── main API ──────────────────────────────────────────────────────────────────────

def recommend(user_profile: Dict, top_n: int = 10) -> List[Dict]:
    """
    Return the top `top_n` cold-start recommendations, honouring exploration_weight.

    Pipeline:
      1. Split every song into familiar (known artist) or discovery (new artist).
      2. Score each pool with its own formula:
           familiar  → score_song()           (artist + genre + popularity)
           discovery → score_discovery_song() (genre + mood/energy + popularity)
      3. Reserve slots per exploration_weight and pick the best from each pool.
      4. Apply artist diversity: at most MAX_SONGS_PER_ARTIST per artist.
    """
    songs             = load_songs()
    artist_embeddings, embedding_index = load_artist_embeddings()

    known_artists: set = (
        set(user_profile.get("favorite_artists", []))
        | set(user_profile.get("recent_artists",   []))
    )

    familiar_pool: List[Dict] = []
    discovery_pool: List[Dict] = []

    for song in songs:
        song_artists = {name.strip() for name in song.get("artist_name", "").split(";")}

        if song_artists & known_artists:
            s = score_song(user_profile, song, artist_embeddings, embedding_index)
            pool = familiar_pool
        else:
            # Discovery songs use a different formula that accounts for mood/vibe
            s = score_discovery_song(user_profile, song)
            pool = discovery_pool

        pool.append({
            "song_id":     song["song_id"],
            "title":       song["title"],
            "artist_name": song["artist_name"],
            "genre":       song["genre"],
            "mood":        song.get("mood", ""),
            "energy":      song.get("energy", ""),
            "score":       s,
        })

    familiar_pool.sort( key=lambda x: x["score"], reverse=True)
    discovery_pool.sort(key=lambda x: x["score"], reverse=True)

    # Slot reservation based on exploration preference
    exploration_weight = float(user_profile.get("exploration_weight", 0.5))
    discovery_slots    = round(top_n * exploration_weight)
    familiar_slots     = top_n - discovery_slots

    # Familiar pool is small (~tens of songs, few distinct artists).
    # strict=True stops at the diversity limit instead of repeating an artist
    # to pad the list.  Any unfilled slots are routed to discovery, which has
    # tens of thousands of songs and never runs short.
    top_familiar  = _pick_diverse(familiar_pool,  familiar_slots, strict=True)
    fam_shortfall = familiar_slots - len(top_familiar)
    top_discovery = _pick_diverse(discovery_pool, discovery_slots + fam_shortfall)

    combined = top_familiar + top_discovery
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined


def recommend_from_file(
    profile_path: Optional[Path] = None,
    top_n: int = 10,
) -> List[Dict]:
    """
    Convenience wrapper: load a profile JSON file and return recommendations.
    Defaults to data/current_user.json when profile_path is None.
    """
    if profile_path is None:
        profile_path = DATA_DIR / "current_user.json"

    with open(profile_path, encoding="utf-8") as f:
        user_profile = json.load(f)

    return recommend(user_profile, top_n=top_n)
