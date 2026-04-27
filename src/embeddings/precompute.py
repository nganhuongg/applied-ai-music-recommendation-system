"""
src/embeddings/precompute.py

Standalone script — run once to build song and artist embeddings from audio
features using PCA.  After running this, initial.py upgrades from its
name-match fallback to real vector similarity automatically.

Usage:
    python src/embeddings/precompute.py

Reads:   data/songs_full.csv
Writes:
    embeddings/song_embeddings.npy     — shape [N_songs   x N_COMPONENTS]
    embeddings/artist_embeddings.npy   — shape [N_artists x N_COMPONENTS]
    embeddings/embedding_index.json    — id ↔ row-index lookup tables
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT      = Path(__file__).resolve().parents[2]
SONGS_CSV      = REPO_ROOT / "data" / "songs_full.csv"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"

# Continuous audio features used as PCA input.
# StandardScaler normalises each one before PCA, so scale differences
# (e.g. tempo_bpm ranges to 250, energy only to 1) don't skew the axes.
AUDIO_FEATURES = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo_bpm",
    "loudness",
    "popularity_norm",
]

# Number of PCA dimensions kept in the output embedding.
# 32 captures ~80-90% of variance for typical audio feature datasets
# while staying small enough for fast cosine similarity at inference time.
N_COMPONENTS = 32


def _safe_float(value: str) -> float:
    """Parse a CSV string to float; return 0.0 on failure or empty string."""
    try:
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0


def main() -> None:
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    print(f"Loading songs from {SONGS_CSV} ...")
    songs: list = []
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            songs.append(row)
    print(f"  {len(songs):,} rows loaded.")

    # Build feature matrix.  All rows are kept (missing values become 0.0).
    feature_rows: list = []
    for song in songs:
        row = [_safe_float(song.get(feat, "")) for feat in AUDIO_FEATURES]
        feature_rows.append(row)

    X = np.array(feature_rows, dtype=np.float32)
    print(f"  Feature matrix: {X.shape}  ({len(AUDIO_FEATURES)} features per song)")

    # StandardScaler → mean=0, std=1 per feature column, then PCA.
    print("Running StandardScaler + PCA ...")
    X_scaled = StandardScaler().fit_transform(X)

    n_comp = min(N_COMPONENTS, X.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    song_embeddings = pca.fit_transform(X_scaled).astype(np.float32)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.1%}")

    # ── song id ↔ row-index maps ─────────────────────────────────────────────
    song_id_to_index: dict = {}
    index_to_song_id: dict = {}
    for idx, song in enumerate(songs):
        sid = song["song_id"]
        song_id_to_index[sid]      = idx
        index_to_song_id[str(idx)] = sid

    # ── artist embeddings ────────────────────────────────────────────────────
    # Each song has one artist_id (even for collabs where artist_name uses
    # semicolons).  We group song embeddings by artist_id and take the mean,
    # producing a vector that summarises that artist's typical sound.
    artist_song_indices: dict = defaultdict(list)
    for idx, song in enumerate(songs):
        aid = song.get("artist_id", "").strip()
        if aid:
            artist_song_indices[aid].append(idx)

    artist_ids = sorted(artist_song_indices.keys())
    artist_id_to_index: dict = {}
    index_to_artist_id: dict = {}
    artist_rows: list = []

    for art_idx, artist_id in enumerate(artist_ids):
        indices    = artist_song_indices[artist_id]
        artist_vec = song_embeddings[indices].mean(axis=0)
        artist_rows.append(artist_vec)
        artist_id_to_index[artist_id]        = art_idx
        index_to_artist_id[str(art_idx)]     = artist_id

    artist_embeddings = np.array(artist_rows, dtype=np.float32)

    # ── save ─────────────────────────────────────────────────────────────────
    np.save(str(EMBEDDINGS_DIR / "song_embeddings.npy"),   song_embeddings)
    np.save(str(EMBEDDINGS_DIR / "artist_embeddings.npy"), artist_embeddings)

    index = {
        "song_id_to_index":   song_id_to_index,
        "index_to_song_id":   index_to_song_id,
        "artist_id_to_index": artist_id_to_index,
        "index_to_artist_id": index_to_artist_id,
    }
    with open(EMBEDDINGS_DIR / "embedding_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    print(f"\nSaved to {EMBEDDINGS_DIR}/")
    print(f"  song_embeddings.npy:   {song_embeddings.shape}")
    print(f"  artist_embeddings.npy: {artist_embeddings.shape}")
    print(
        f"  embedding_index.json:  {len(song_id_to_index):,} songs, "
        f"{len(artist_id_to_index):,} artists"
    )


if __name__ == "__main__":
    main()
