"""
src/ranking/ranker.py

Load the trained LightGBM model and score a candidate pool.

Usage:
    from ranking.ranker import rank

    top_50 = rank(
        candidate_song_ids=retriever.build_candidate_pool(...),
        user_profile=profile_dict,
        user_id="some_user_id",
    )

Each returned dict has: song_id, score, and the 6 raw feature values.

Fallback behaviour:
    If models/ranking_model.pkl does not exist OR lightgbm is not installed,
    rank() falls back to a hand-weighted formula using the same 6 features.
    The pipeline never hard-crashes because of a missing model.
    Check model_status() to see which mode is active.
"""

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from ranking.features import FEATURE_COLUMNS, compute_features, feature_matrix

REPO_ROOT  = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "ranking_model.pkl"

TOP_N = 50

# Module-level cache: the model is loaded once on first call, then reused.
_model = None
_model_load_attempted = False


def _try_load_model():
    """
    Attempt to load the LightGBM pkl exactly once.

    Returns the model object, or None if:
      - the file does not exist yet
      - lightgbm is not installed on this machine

    Either way, rank() continues using the fallback scorer.
    """
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True

    if not MODEL_PATH.exists():
        return None

    try:
        import lightgbm  # noqa: F401 — needed so pickle can deserialise LGBMClassifier
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    except (ImportError, Exception):
        # lightgbm not installed, or the pkl is from an incompatible version
        _model = None

    return _model


def _fallback_score(row: Dict) -> float:
    """
    Hand-weighted score used when no trained model is available.

    Weights are chosen to roughly replicate what LightGBM learns on typical
    music interaction data: content match and collaborative signal dominate,
    freshness and popularity are secondary, artist match is a small bonus.

    This is intentionally simple — its purpose is to keep the pipeline
    functional, not to replace the trained model.
    """
    return (
        0.35 * row.get("content_similarity",  0.0)
        + 0.25 * row.get("collaborative_score", 0.0)
        + 0.20 * row.get("artist_similarity",   0.0)
        + 0.10 * row.get("freshness",            0.0)
        + 0.10 * row.get("popularity",           0.0)
        # diversity_penalty intentionally excluded from the fallback sum:
        # a positive penalty would lower the score of songs that should
        # still rank well — it is handled differently in each context.
    )


def rank(
    candidate_song_ids: List[str],
    user_profile: Dict,
    user_id: str,
    top_n: int = TOP_N,
) -> List[Dict]:
    """
    Score and rank the candidate pool for a given user.

    Args:
        candidate_song_ids: Output of retriever.build_candidate_pool().
        user_profile:       User's profile dict (from current_user.json).
        user_id:            User's ID string, used to look up liked songs
                            in interactions.csv for the collaborative feature.
        top_n:              How many songs to return (default 50).

    Returns:
        List of dicts sorted by score descending, length <= top_n.
        Each dict contains:
            song_id, score, content_similarity, artist_similarity,
            freshness, popularity, collaborative_score, diversity_penalty

    Example:
        top_50 = rank(candidate_ids, profile, "user_123")
        for song in top_50[:10]:
            print(song["song_id"], f"{song['score']:.3f}")
    """
    if not candidate_song_ids:
        return []

    feature_rows = compute_features(candidate_song_ids, user_profile, user_id)
    if not feature_rows:
        return []

    model = _try_load_model()

    if model is not None:
        X = feature_matrix(feature_rows)       # shape (N, 6)
        scores = model.predict_proba(X)[:, 1]  # probability of label=1
    else:
        scores = np.array(
            [_fallback_score(row) for row in feature_rows], dtype=np.float32
        )

    for row, score in zip(feature_rows, scores):
        row["score"] = float(score)

    ranked = sorted(feature_rows, key=lambda r: r["score"], reverse=True)
    return ranked[:top_n]


def model_status() -> str:
    """
    Return a human-readable string describing which scoring mode is active.

    Useful for debugging and for displaying in the UI.
    """
    model = _try_load_model()
    if model is not None:
        return "LightGBM (trained model)"
    if not MODEL_PATH.exists():
        return "fallback (models/ranking_model.pkl not found)"
    return "fallback (lightgbm not installed on this machine)"
