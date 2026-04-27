# PLAN.md — AI-Integrated Hybrid Music Recommendation System

## Purpose of This Document

Authoritative implementation plan and current state record for this project.
When reading code or deciding what to build next, treat this as the source of truth.

---

## Current State (as of 2026-04-26)

All core modules are built. Only evaluation and tests remain.

| File | Status | Role |
|---|---|---|
| `data/songs_full.csv` | ✅ Done | 81,343 songs with audio features |
| `data/interactions.csv` | ✅ Done | User feedback log (user_id, song_id, label, play_count, timestamp) |
| `data/user_profiles/` | ✅ Done | Per-user profile JSONs |
| `data/current_user.json` | ✅ Done | Most recently confirmed profile |
| `src/profile/survey.py` | ✅ Done | Streamlit app: 8-step survey → like/skip → **playlist view** (step 20) |
| `src/recommender/initial.py` | ✅ Done | Cold-start recommender (artist + genre + popularity, slot reservation, artist diversity) |
| `src/embeddings/precompute.py` | ✅ Done | One-time script: PCA(32) → song_embeddings.npy, artist_embeddings.npy, embedding_index.json |
| `src/embeddings/user_embedding.py` | ✅ Done | Mean of liked song PCA vectors; blends with survey prior when < 5 likes |
| `src/retrieval/retriever.py` | ✅ Done | Freshness-aware candidate pool (cosine + artist union, progressive window fallback) |
| `src/ranking/features.py` | ✅ Done | 6 features per song, returns `List[Dict]`; use `feature_matrix()` for numpy |
| `notebooks/train_on_colab.ipynb` | ✅ Done | Self-contained Colab training notebook |
| `models/ranking_model.pkl` | ✅ Done | Trained LightGBM ranker (downloaded from Colab) |
| `models/ranking_model_meta.json` | ✅ Done | Model metadata |
| `src/ranking/ranker.py` | ✅ Done | `rank()`, `model_status()`, fallback hand-weighted scoring if LightGBM unavailable |
| `src/rag/build_knowledge_base.py` | ✅ Done | Run once: Wikipedia bios + song descriptions + genre/purpose guides → Gemini embeddings → .npy + .json |
| `src/rag/rag_layer.py` | ✅ Done | `recommend()`, `search_knowledge_base()`, Gemini chat with JSON mode, two fallback paths |
| `src/main.py` | ✅ Done | Streamlit app for returning users: loads profile → intent input → full pipeline → song cards |
| `src/evaluation.py` | ⬜ Next | Offline evaluation metrics (Precision@K, NDCG@K, diversity, novelty) |
| `tests/` | ⬜ | Unit tests for retrieval, ranking, rag modules |

---

## How to Run

### New user (first time)
```bash
streamlit run src/profile/survey.py
```
Survey (8 steps) → Profile Confirmation → Like/Skip cold-start songs → **"See my playlist →"** → playlist view with full pipeline.

### Returning user
```bash
streamlit run src/main.py
```
Loads `data/current_user.json` directly → intent input → full pipeline → song cards.

### Build the RAG knowledge base (run once, optional)
```bash
python src/rag/build_knowledge_base.py
```
Fetches Wikipedia artist bios, generates song descriptions, embeds with Gemini `text-embedding-004`. Without this file the RAG layer still works — it skips retrieval and passes history context only.

---

## Full UI Flow (implemented)

```
streamlit run src/profile/survey.py
  ↓
Landing page
  ↓  (8-question survey: name, artists, genres, mood, purpose, exploration, popularity)
Profile Confirmation screen   ← user can go back to edit
  ↓  Confirm
Song Feedback screen          ← user Likes or Skips each of the 10 cold-start songs
  ↓  all songs rated (and user liked ≥ 1)
"See my playlist →" button    ← advances to step 20
  ↓
Playlist view (step 20)
  ├── optional free-text intent input  ("studying tonight, calm focus, no lyrics")
  ├── "Refresh Playlist" button
  ├── Song cards with title, artist, per-song explanation, Like/Skip buttons
  └── "Start over" resets all state
```

If user skips ALL songs → search fallback (step 11) → manually pick 2–5 known songs → "See my playlist →" → step 20.

---

## Architecture (End-to-End, as built)

```
Survey (8 questions)                    [src/profile/survey.py]
  ↓  build_profile() → save_profile()
Profile JSON                            [data/current_user.json]
  ↓
Initial Recommender (cold-start top 10) [src/recommender/initial.py]
  ↓
Song Feedback (Like / Skip)             [src/profile/survey.py — step_song_feedback()]
  ↓  save_interactions() → data/interactions.csv
  ↓  compute_user_embedding() → data/user_embedding.npy
  ↓  "See my playlist →" → step 20
═══════════════════════════════════════════════════════════════════════════
Playlist refresh (step 20 in survey.py, or directly via src/main.py):
═══════════════════════════════════════════════════════════════════════════
User types optional intent              "calm focus music, studying tonight"
  ↓
Retriever                               [src/retrieval/retriever.py]
  ├── embedding similarity (cosine, numpy) vs fresh songs
  └── artist-based (from favorite/similar artists)
  ↓
Candidate Pool (up to 200 song IDs)
  ↓
Ranking Model                           [src/ranking/ranker.py]
  ├── loads models/ranking_model.pkl (LightGBM trained on Colab)
  └── fallback: hand-weighted formula if LightGBM unavailable
  ↓
Top 50 scored songs
  ↓
RAG Layer                               [src/rag/rag_layer.py]
  ├── embed user intent (Gemini text-embedding-004, RETRIEVAL_QUERY)
  ├── cosine search against knowledge base (numpy dot product, L2-normalised)
  │     [built once by src/rag/build_knowledge_base.py]
  │     contains: Wikipedia artist bios + auto-generated song descriptions
  │               + genre guides (31) + purpose guides (6)
  │     stored as: embeddings/knowledge_base_vectors.npy [N×768]
  │                embeddings/knowledge_base_docs.json   [parallel list]
  └── Gemini 2.0 Flash reasons over: retrieved docs + top-50 songs + history summary
  ↓
Final Top 10 with per-song explanations
  ↓
Song cards in Streamlit UI (Like/Skip recorded to interactions.csv)
```

---

## Actual Folder Structure

```
applied-ai-music-recommendation-system/
├── data/
│   ├── songs_full.csv              ✅ 81,343 songs
│   ├── interactions.csv            ✅ user feedback (user_id, song_id, label, play_count, timestamp)
│   ├── current_user.json           ✅ most recent confirmed profile
│   └── user_profiles/              ✅ per-user profile JSONs
│
├── embeddings/
│   ├── song_embeddings.npy         ✅ PCA(32) song vectors [N_songs × 32]
│   ├── artist_embeddings.npy       ✅ PCA(32) artist vectors [N_artists × 32]
│   ├── embedding_index.json        ✅ song_id/artist_id ↔ row index maps
│   ├── knowledge_base_vectors.npy  ✅ Gemini text-embedding-004 [N_docs × 768], L2-normalised
│   └── knowledge_base_docs.json    ✅ parallel list of {text, type, title} dicts
│
├── models/
│   ├── ranking_model.pkl           ✅ trained LightGBM classifier (from Colab)
│   └── ranking_model_meta.json     ✅ feature names, training date, metrics
│
├── notebooks/
│   └── train_on_colab.ipynb        ✅ self-contained Colab training notebook
│
├── src/
│   ├── profile/
│   │   └── survey.py               ✅ onboarding survey + playlist view (step 20)
│   ├── recommender/
│   │   └── initial.py              ✅ cold-start recommender
│   ├── embeddings/
│   │   ├── precompute.py           ✅ one-time PCA embedding script
│   │   └── user_embedding.py       ✅ compute user embedding from liked songs
│   ├── retrieval/
│   │   └── retriever.py            ✅ build_candidate_pool()
│   ├── ranking/
│   │   ├── features.py             ✅ compute_features() → List[Dict], feature_matrix() → np.ndarray
│   │   └── ranker.py               ✅ rank(), model_status()
│   ├── rag/
│   │   ├── __init__.py             ✅ (empty)
│   │   ├── build_knowledge_base.py ✅ run once offline
│   │   └── rag_layer.py            ✅ recommend(), search_knowledge_base()
│   └── main.py                     ✅ Streamlit returning-user entry point
│
├── tests/                          ⬜ to be written
├── .env                            GEMINI_API_KEY=... (gitignored)
├── .env.example                    placeholder for reference
├── .streamlit/config.toml          dark theme config
├── PLAN.md                         this file
├── reflection.md                   session learning log
├── README.md
└── requirements.txt
```

---

## Step-by-Step Reference (completed steps)

### Step 1 — Dataset (`data/songs_full.csv`)
81,343 songs. Columns include: `song_id, title, artist_id, artist_name, genre, mood, release_date, energy, tempo_bpm, valence, danceability, acousticness, instrumentalness, speechiness, popularity, popularity_norm`.

---

### Step 2 — Survey → User Profile (`src/profile/survey.py`)

8-question Streamlit survey. Produces a profile JSON with these fields:
```python
{
    "user_id":               str,          # uuid4
    "name":                  str,
    "created_at":            str,          # ISO datetime
    "favorite_artists":      List[str],    # 3–5 artist names
    "recent_artists":        List[str],    # 2–3 artist names
    "favorite_genres":       List[str],    # from 30-genre vocabulary
    "selected_moods":        List[str],    # Calm / Energetic / Upbeat / Melancholic / Intense
    "genre_vector":          List[float],  # multi-hot over GENRE_VOCABULARY (30 dims)
    "mood_targets":          Dict,         # {"target_energy": float, "target_valence": float}
    "purpose":               str,          # study | workout | relax | commute | explore
    "exploration_weight":    float,        # 0.1 (familiar) → 0.9 (discovery)
    "popularity_preference": str,          # mainstream | mixed | underground
}
```

Saved to `data/user_profiles/{uuid}.json` and `data/current_user.json`.

After like/skip the survey now flows to **step 20 (playlist view)** — no need to run a separate app.

---

### Step 3 — Embeddings (`src/embeddings/precompute.py`)

Run once: `python src/embeddings/precompute.py`

PCA(32) on audio features (energy, tempo, valence, danceability, acousticness, instrumentalness, speechiness). Produces:
- `embeddings/song_embeddings.npy` — shape [N_songs × 32]
- `embeddings/artist_embeddings.npy` — shape [N_artists × 32] (mean of artist's song vectors)
- `embeddings/embedding_index.json` — `{song_id_to_index, index_to_song_id, artist_id_to_index, index_to_artist_id}`

---

### Step 4 — Cold-Start Recommender (`src/recommender/initial.py`)

Used only for the initial like/skip screen (before any interaction history). Two scoring formulas:

**Familiar songs (artist known):**
```
score = 0.5 × artist_similarity + 0.3 × genre_similarity + 0.2 × popularity_norm
```

**Discovery songs (artist unknown):**
```
score = 0.40 × genre_similarity + 0.35 × mood_compatibility + 0.25 × popularity_norm
```

Slot reservation by `exploration_weight` prevents familiar artists from dominating when the user asked for discovery.

---

### Step 5 — Feedback Collection

Implemented inside `survey.py` as `save_interactions()`. Appends to `data/interactions.csv`:
```
user_id, song_id, label, play_count, timestamp
```
`label=1` = liked, `label=0` = skipped. Both are recorded — skips are negative training examples.

---

### Step 6 — User Embedding (`src/embeddings/user_embedding.py`)

`compute_user_embedding(liked_song_ids, profile)` → saves `data/user_embedding.npy`.

Mean of liked song PCA vectors. Blends with survey prior when fewer than 5 songs liked:
```
user_vec = 0.7 × behavior_mean + 0.3 × artist_vector_from_survey
```

---

### Step 7 — Retriever (`src/retrieval/retriever.py`)

`build_candidate_pool(user_profile, target_pool_size=200)` → `List[str]` (song IDs).

Two strategies combined via union:
1. **Embedding similarity** — cosine(user_embedding, song_embeddings) over fresh songs
2. **Artist-based** — songs from artists similar to user favorites

Freshness: starts at 90 days, relaxes progressively (180 → 365 → 730 → all songs) until ≥ 50 songs pass.

---

### Step 8 — Feature Engineering (`src/ranking/features.py`)

`compute_features(song_ids, user_profile, user_id)` → `List[Dict]`

6 features per song:
- `content_similarity` — cosine(user_embedding, song_embedding)
- `collaborative_score` — mean similarity to user's liked songs via item-item cosine
- `artist_similarity` — cosine(user_artist_vector, song_artist_embedding)
- `freshness` — `exp(-0.01 × age_in_days)`
- `popularity` — normalised popularity_norm from CSV
- `diversity_penalty` — count of songs by same artist already scored

**Note:** returns `List[Dict]`, not `pd.DataFrame` — pandas cannot be installed on this machine (MINGW Python, no MSVC). Use `feature_matrix(rows)` to get a numpy array for LightGBM.

---

### Steps 9–10 — Colab Training

`notebooks/train_on_colab.ipynb` runs precompute + feature engineering + LightGBM training. Model downloaded to `models/ranking_model.pkl`.

---

### Step 11 — Ranker (`src/ranking/ranker.py`)

`rank(candidate_song_ids, user_profile, user_id, top_n=50)` → `List[Dict]`

Each returned dict: `{song_id, score, content_similarity, artist_similarity, freshness, popularity, collaborative_score, diversity_penalty}`.

If `lightgbm` not installed or `.pkl` missing → falls back to hand-weighted formula (no crash). Check `model_status()` to see which mode is active.

---

### Step 12a — Knowledge Base (`src/rag/build_knowledge_base.py`)

Run once: `python src/rag/build_knowledge_base.py`

Builds 4 document types, embeds with Gemini `text-embedding-004` (768-dim), saves:
- `embeddings/knowledge_base_vectors.npy` — L2-normalised, shape [N_docs × 768]
- `embeddings/knowledge_base_docs.json` — parallel list of `{text, type, title}`

| Type | Count | Source |
|---|---|---|
| `artist_bio` | up to 500 | Wikipedia summaries (`wikipedia` package) |
| `song_description` | up to 5,000 | Auto-generated from CSV audio features |
| `genre_guide` | 31 | Hardcoded prose per genre |
| `purpose_guide` | 6 | Hardcoded prose: studying, working out, relaxing, sleeping, social, commuting |

Gemini embedding REST endpoint used:
```
POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={GEMINI_API_KEY}
```
Task type `RETRIEVAL_DOCUMENT` for building the index; `RETRIEVAL_QUERY` for querying.

FAISS **was not used** — `faiss-cpu` cannot be built on this machine. Replaced with numpy dot product on L2-normalised vectors, which is mathematically equivalent and takes ~2 ms per query at this scale.

---

### Step 12b — RAG Layer (`src/rag/rag_layer.py`)

Called at every playlist refresh.

Public API:
```python
recommend(user_intent, top_50_songs, history_summary, api_key, k=10) -> List[Dict]
search_knowledge_base(query_text, api_key, top_k=5) -> List[Dict]
```

Output dict per song: `{song_id, title, artist, score, explanation}`.

Pipeline:
1. Load KB vectors + docs into module-level cache on first call
2. If `user_intent` is non-empty: embed with Gemini (`RETRIEVAL_QUERY`), cosine-search KB, retrieve top 5 docs
3. Build structured prompt with songs + retrieved docs + history summary
4. Call `gemini-2.0-flash` with `responseMimeType: application/json` → returns `{"recommendations": [...]}`
5. Parse JSON, join with songs_full.csv metadata, return

Fallbacks:
- KB files missing → skip retrieval, Gemini still reasons from history_summary
- Gemini call fails → return `top_50_songs[:k]` with empty explanations (ML ranking stands)

---

### Step 13 — Wiring (`src/main.py` + `src/profile/survey.py` step 20)

`src/main.py` — Streamlit app for returning users. Loads profile from `data/current_user.json`, shows intent input + Refresh Playlist → runs full pipeline → song cards.

`src/profile/survey.py` step 20 (`step_playlist()`) — same pipeline embedded inside the survey app. Both completion screens ("All done!" after like/skip and "Profile updated." after search fallback) now show a "See my playlist →" button that advances to step 20.

---

## Remaining Steps

### Step 14 — Evaluation (`src/evaluation.py`)

**Goal:** Compute offline metrics from `data/interactions.csv` to assess recommendation quality.

Metrics to implement:

| Metric | What it measures |
|---|---|
| Precision@K | Fraction of top-K recommendations that the user actually liked |
| Recall@K | Fraction of all liked songs that appeared in top-K |
| NDCG@K | Rewards putting liked songs near the top (log-penalises lower positions) |
| Diversity@K | `1 − (repeat-artist fraction in top K)` — are we showing variety? |
| Novelty@K | `mean(−log(popularity_norm))` of top K — are we surfacing non-obvious songs? |

**Suggested public API:**
```python
def evaluate(user_id: str, k: int = 10) -> Dict[str, float]:
    """Run the full pipeline for user_id and compute all metrics against interactions.csv."""

def evaluate_all_users(k: int = 10) -> Dict[str, Dict[str, float]]:
    """Run evaluate() for every user in interactions.csv; return per-user and averaged results."""
```

**Implementation note:** evaluation requires running the retriever + ranker, then comparing the ranked list against held-out liked songs from interactions.csv. Split interactions by timestamp: train on earlier half, evaluate on later half.

---

### Step 15 — Tests (`tests/`)

Suggested test files:

| File | What to test |
|---|---|
| `tests/test_retrieval.py` | `build_candidate_pool()` returns non-empty list; deduplication; fallback windows trigger correctly |
| `tests/test_ranking.py` | `compute_features()` returns correct keys; `feature_matrix()` shape is (N, 6); fallback score is in [0, 1] |
| `tests/test_rag.py` | `search_knowledge_base()` returns list of dicts with correct keys; `recommend()` fallback when Gemini fails |
| `tests/test_evaluation.py` | Precision@K = 1.0 when all top-K are liked; NDCG@K = 1.0 when perfect order |

---

## Key Design Decisions

| Decision | Chosen approach | Why |
|---|---|---|
| Cold-start | Survey → artist/genre vectors | No behavior data yet; survey provides initial bias |
| Discovery injection | Slot reservation by `exploration_weight` | Pure score-sort always favours familiar artists |
| Embedding type | PCA(32) on audio features | Simple, no GPU; captures musical similarity without training |
| Ranker model | LightGBM | Fast, handles feature interactions, trains on Colab, runs locally |
| RAG embeddings | Gemini `text-embedding-004` (768-dim) | Available via GEMINI_API_KEY; supports RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY task types |
| RAG vector store | numpy dot product on L2-normalised vectors | `faiss-cpu` cannot be installed on MINGW Python (no MSVC); numpy gives identical results at this KB size |
| RAG reasoning model | Gemini 2.0 Flash | User has GEMINI_API_KEY; `responseMimeType: application/json` forces clean JSON output |
| RAG scope | Top 50 ML-ranked songs + 5 retrieved KB docs | Limits prompt tokens and cost; ML handles bulk filtering, Gemini handles nuance |
| Feature return type | `List[Dict]` not `pd.DataFrame` | pandas cannot be installed on this machine; `feature_matrix()` produces numpy array for LightGBM |
| User embedding blend | 0.7 behavior + 0.3 survey when < 5 likes | Survey prior stabilises noisy mean on small sample |
| Freshness decay | Exponential `exp(-0.01 × days)` | Smooth decay; configurable via `DECAY_LAMBDA` constant |
| Single API key | Gemini for both embeddings and generation | No Anthropic key, no MiniMax key — Gemini handles everything |

---

## Environment Variables

```
GEMINI_API_KEY=your_key_here
```

One key only. Used by:
- `build_knowledge_base.py` — embedding documents (`text-embedding-004`)
- `rag_layer.py` — embedding queries + chat generation (`gemini-2.0-flash`)
- `survey.py` step 20 and `main.py` — same rag_layer calls

---

## Data Schemas

### interactions.csv
```
user_id, song_id, label, play_count, timestamp
```
- `label`: 1 = liked, 0 = skipped
- `play_count`: 1 for liked, 0 for skipped

### song returned by `ranker.rank()`
```python
{
    "song_id":              str,
    "score":                float,
    "content_similarity":   float,
    "collaborative_score":  float,
    "artist_similarity":    float,
    "freshness":            float,
    "popularity":           float,
    "diversity_penalty":    float,
}
```

### song returned by `rag_layer.recommend()`
```python
{
    "song_id":     str,
    "title":       str,
    "artist":      str,
    "score":       float,   # original ML score
    "explanation": str,     # one sentence from Gemini (empty string if fallback)
}
```

### embedding_index.json
```json
{
  "song_id_to_index":   {"song_abc": 0, ...},
  "index_to_song_id":   {"0": "song_abc", ...},
  "artist_id_to_index": {"artist_xyz": 0, ...},
  "index_to_artist_id": {"0": "artist_xyz", ...}
}
```

---

## Dependencies (what actually works on this machine)

```
numpy
scikit-learn
streamlit
requests
python-dotenv
wikipedia
```

Cannot install (MINGW Python, no MSVC): `pandas`, `lightgbm`, `faiss-cpu`.

All code handles these missing packages gracefully:
- No `pandas` anywhere — `List[Dict]` + numpy throughout
- No `lightgbm` locally — `ranker.py` falls back to hand-weighted scoring
- No `faiss-cpu` — `rag_layer.py` uses numpy dot product

The Colab notebook can install all of these for training.
