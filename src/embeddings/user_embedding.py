"""
src/embeddings/user_embedding.py

Compute and persist a behaviour-based user embedding from liked songs.

The embedding is the mean of the PCA song vectors for songs the user liked.
When fewer than BLEND_THRESHOLD songs are liked, it blends the behaviour
signal with the survey's artist_vector to stabilise the noisy estimate
(PLAN.md §3b hybrid formula: 0.7 behaviour + 0.3 survey).

If the precomputed embedding files do not yet exist (precompute.py has not
been run), every public function returns None gracefully so the rest of the
app keeps working in name-match fallback mode.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR       = REPO_ROOT / "data"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"

BEHAVIOR_WEIGHT = 0.7
SURVEY_WEIGHT   = 0.3

# Below this many liked songs the mean embedding is noisy — blend with the
# survey's artist_vector to add a stabilising prior (see PLAN.md §3b pitfall).
BLEND_THRESHOLD = 5


def load_song_embeddings():
    """
    Load the precomputed song embedding matrix and id↔index mapping.
    Returns (matrix, index_dict) or (None, None) when files are missing.
    """
    emb_path   = EMBEDDINGS_DIR / "song_embeddings.npy"
    index_path = EMBEDDINGS_DIR / "embedding_index.json"

    if not emb_path.exists() or not index_path.exists():
        return None, None

    embeddings = np.load(str(emb_path))
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    return embeddings, index


def compute_user_embedding(
    liked_song_ids: List[str],
    profile: Optional[Dict] = None,
) -> Optional[np.ndarray]:
    """
    Build a user embedding from the songs they liked, save it to disk,
    and return the vector.

    Steps:
      1. Look up each liked song's PCA row in song_embeddings.npy.
      2. Take the element-wise mean → behaviour_vector.
      3. If fewer than BLEND_THRESHOLD songs were liked AND the profile
         carries an artist_vector (same dimension), blend:
             user_vec = 0.7 * behaviour + 0.3 * survey_artist_vector
      4. Save to data/user_embedding.npy for downstream pipeline steps.

    Returns None when:
      - liked_song_ids is empty
      - precomputed embedding files don't exist yet
      - none of the liked song IDs appear in the index
    """
    if not liked_song_ids:
        return None

    embeddings, index = load_song_embeddings()
    if embeddings is None:
        return None

    song_id_to_index = index.get("song_id_to_index", {})

    rows = []
    for song_id in liked_song_ids:
        row_idx = song_id_to_index.get(song_id)
        if row_idx is not None:
            rows.append(embeddings[int(row_idx)])

    if not rows:
        return None

    behavior_vec = np.mean(rows, axis=0)

    user_vec = behavior_vec
    if profile is not None and len(rows) < BLEND_THRESHOLD:
        artist_vec = profile.get("artist_vector")
        # artist_vector is added to the profile in a future step (Step 3 of PLAN.md).
        # Until then this branch is skipped gracefully.
        if artist_vec is not None and len(artist_vec) == len(behavior_vec):
            user_vec = (
                BEHAVIOR_WEIGHT * behavior_vec
                + SURVEY_WEIGHT  * np.array(artist_vec, dtype=np.float32)
            )

    user_vec = user_vec.astype(np.float32)

    DATA_DIR.mkdir(exist_ok=True)
    np.save(str(DATA_DIR / "user_embedding.npy"), user_vec)

    return user_vec
