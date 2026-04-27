# Model Card: Applied AI Music Recommendation System

## 1. Model Name

**VibeMatch AI — Retrieve → Rank → Explain Pipeline**

---

## 2. Goal / Task

This system recommends a personalised playlist from a catalog of 81,343 songs by
combining three AI stages: TF-IDF keyword retrieval, LightGBM ML ranking, and a
Groq language model (Llama-3.3-70b) that writes one-sentence explanations for each
recommendation.

The system answers two questions at once: "Which songs in this catalog match what
the user is asking for?" (retrieval) and "Of those, which ones best fit this
particular user's taste based on their history?" (ranking). The language model
then translates the result into plain English.

This is not a production streaming service. It is an educational project
demonstrating how modern recommendation pipelines work at each stage.

---

## 3. Intended Use and Non-Intended Use

**Intended use:** personal music discovery, classroom learning, and portfolio
demonstration of RAG + ML ranking concepts.

**Not intended for:** commercial music streaming, high-stakes content curation,
or any setting where algorithmic bias could cause harm to users or artists.

---

## 4. Data Used

### Original data sources

Two public Kaggle datasets were combined to build the catalog and interaction log:

- **Last.fm Dataset** — [harshal19t/lastfm-dataset](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)
  Provided user listening history and song interaction records used to seed
  `interactions.csv` (user-song play counts and implicit feedback signals).

- **Spotify Tracks Dataset** — [maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
  Provided audio features per track: energy, tempo, valence, danceability,
  acousticness, instrumentalness, speechiness, and popularity.

These two datasets were merged with the assistance of Claude (AI assistant):
duplicate songs were removed, song IDs were standardised so that user
interactions from the Last.fm data could be correctly joined to song audio
features from the Spotify data, and the schema was normalised into the single
`songs_full.csv` format this project uses.

### Resulting files

**Catalog:** `data/songs_full.csv` — 81,343 songs with the following features:
`song_id`, `title`, `artist_name`, `artist_id`, `genre`, `mood`, `release_date`,
`energy`, `tempo_bpm`, `valence`, `danceability`, `acousticness`,
`instrumentalness`, `speechiness`, `popularity_norm`.

**Interaction log:** `data/interactions.csv` — Like/Skip feedback collected
during user sessions. Each row records `user_id`, `song_id`, `label` (1=liked,
0=skipped), `play_count`, and `timestamp`. This file is the training data for
the LightGBM ranking model.

**User profiles:** `data/user_profiles/` and `data/current_user.json` — survey
answers converted into structured fields including `genre_vector` (multi-hot
encoding), `mood_targets` (numeric energy and valence targets), `exploration_weight`,
and `popularity_preference`.

---

## 5. Algorithm Summary

The system runs three stages in sequence on every playlist refresh:

**Stage 1 — TF-IDF Retrieval** (`src/rag/`)
Each song has a generated text description packed with searchable keywords: genre,
mood, era, tempo label, use-case phrases, and "instrumental no lyrics" for tracks
with high instrumentalness. A TF-IDF vectorizer is fitted on all 81k descriptions.
At query time, the user's free-text intent is converted to a TF-IDF vector and
dot-producted against the index matrix to retrieve the 100 most relevant songs.

**Stage 2 — ML Ranking** (`src/ranking/`)
Six features are computed per candidate song:
- `content_similarity` — cosine(user embedding, song embedding)
- `artist_similarity` — cosine(user artist vector, artist embedding)
- `freshness` — exp(−0.01 × age_in_days)
- `popularity` — normalised popularity from the CSV
- `collaborative_score` — mean cosine(candidate, all liked songs)
- `diversity_penalty` — count of same-artist songs ranked above this one

A LightGBM classifier trained on `interactions.csv` predicts the probability that
the user will like each song. A hand-weighted fallback runs when LightGBM is
unavailable locally.

**Stage 3 — Groq LLM Explanation** (`src/rag/rag_layer.py`)
The top 50 ML-ranked songs and the user's original intent text are sent to
Groq (Llama-3.3-70b). Groq writes one sentence per song explaining why it fits,
citing the user's own words where possible. The final playlist returns 10 songs.

When the user types no intent, Stage 1 is replaced by a profile-based retriever
(`src/retrieval/retriever.py`) that uses cosine similarity over PCA audio embeddings.

---

## 6. Observed Behavior and Biases

**What works well:**
The system separates calm acoustic styles from energetic dance-oriented styles
reliably. When a user's survey answers and liked songs point in the same direction,
the recommendations feel cohesive and accurate.

**Keyword dependency:**
TF-IDF is purely keyword-based. It cannot understand that "mellow" and "calm" are
synonyms unless both words appear in song descriptions. Queries using uncommon
synonyms may return weaker results than queries using the exact words embedded
during indexing.

**Genre and popularity concentration:**
The catalog has uneven genre distribution — some genres have thousands of songs,
others have dozens. Users who prefer underrepresented genres receive a weaker
candidate pool, which the ranking model cannot fully compensate for.

**Cold-start artist bias:**
Early in a session, before enough liked songs accumulate to build a strong user
embedding, the cold-start recommender leans toward the user's stated favourite
artists. This means users who explicitly asked for discovery (high `exploration_weight`)
still saw familiar artists dominating their first playlist. This was identified
during testing and addressed — see Section 7.

---

## 7. Testing Methods and Results

### Automated fallback testing

All three fallback paths were verified manually:
- **Missing TF-IDF index** → pipeline falls back to profile-based retrieval without crashing
- **Groq API failure** (tested with an invalid key) → ML-ranked songs are returned with empty explanation fields; the app continues normally
- **Missing LightGBM model** → ranker uses hand-weighted formula; output is still ordered and reasonable

### Freshness window testing

The 81k catalog pre-dates 2024 for most songs. The progressive freshness window
relaxation (90 → 180 → 365 → 730 → all songs) was verified to trigger automatically
and print a log message when a wider window was needed.

### Friends testing

The system was tested with a small group of friends who completed the survey and
rated songs. The main finding: **the recommended songs mostly matched our actual
musical taste**, confirming that the combination of survey-based cold-start and
interaction-based ranking was producing sensible results.

**One issue found during testing:**
Even when a user set their `exploration_weight` to maximum (asking for discovery),
the initial cold-start playlist was still dominated by their declared favourite
artists. Users who wanted to discover new music were not getting it.

**Fix implemented:**
Two mechanisms were added to address this:
1. **Slot reservation** in `src/recommender/initial.py` — `exploration_weight`
   now controls how many of the 10 cold-start slots go to familiar artists vs.
   discovery songs. A user who sets exploration to maximum gets mostly unfamiliar
   artists in their first playlist.
2. **Artist similarity via embeddings** in `src/retrieval/retriever.py` — the
   retriever now finds songs from artists whose audio embedding is *similar* to
   the user's favourites, not just artists the user explicitly named. This surfaces
   genuinely new artists that match the same sonic vibe.

---

## 8. Responsible AI Reflection

### Limitations and biases

- **Filter bubble risk:** the collaborative score rewards songs similar to what
  the user already liked. Over time, this can narrow recommendations and make it
  harder to discover genuinely different music. The `exploration_weight` and
  `diversity_penalty` features partially counter this but do not eliminate it.
- **Popularity bias:** the `popularity` feature rewards well-known songs.
  Underground or niche artists are disadvantaged even when their music would be
  a good taste match.
- **Language and culture blindness:** the system operates entirely on audio
  features and keywords. It has no understanding of lyrics, language, cultural
  context, or artist background.
- **Dataset quality:** the catalog's mood and genre labels are assigned by a
  third party and may not match how users describe music themselves.

### Could this AI be misused, and how would you prevent it?

The most realistic misuse would be using the interaction log to profile users'
listening patterns for commercial purposes without their knowledge.

Preventions built into this project:
- All data (`interactions.csv`, user profile JSONs) is stored locally and never
  transmitted to any server.
- The `.env` file and `.gitignore` ensure API keys and personal data are not
  accidentally committed to a public repository.
- The system has no account system or persistent identity beyond a local UUID —
  there is nothing to sell or leak to a third party.

At a larger scale, this kind of system would also need: explicit consent for data
collection, the ability to delete your interaction history, and transparency about
how recommendations are generated (the explanations from Groq partially address this).

### What surprised me during reliability testing

The most surprising discovery was how well `"no lyrics"` and `"instrumental"` worked
as search terms. During testing I typed `"calm piano no lyrics for studying"` and
the TF-IDF immediately returned mostly instrumental classical and ambient songs —
because those exact words (`instrumental`, `no lyrics`) are embedded in every
qualifying song's description during the build step. The keyword design decision
paid off in a way that was not obvious until we tested it with real queries.

The second surprise was the Groq fallback: when tested with an intentionally invalid
API key, the app showed the ML-ranked songs with blank explanation fields and
continued working normally. The graceful degradation that was designed into the code
actually worked as intended.

---

## 9. AI Collaboration During This Project

Building this project involved significant collaboration with Claude (AI assistant)
across multiple sessions. Here are two specific examples that illustrate both the
value and the limits of that collaboration.

**One instance where the AI suggestion was genuinely helpful:**

When the cold-start bias problem was identified (users wanting discovery still seeing
only familiar artists), the AI suggested the **slot reservation mechanism** — using
`exploration_weight` not just as a label but as a numeric control over how many
playlist slots are allocated to familiar vs. discovery artists. It also suggested
routing any "familiar shortfall" (when not enough distinct favourite artists exist)
into the discovery pool rather than repeating artists. This was a clean design
pattern ("policy vs. mechanism") that I would not have arrived at as quickly on my own.

**One instance where the AI suggestion was flawed:**

The original AI-designed retrieval plan used **FAISS** (Facebook AI Similarity Search)
for fast vector search over the 81k song embeddings. The AI recommended this
confidently as the industry-standard tool for this kind of problem — and it is, at
larger scale. However, FAISS requires MSVC (Microsoft Visual C++) to compile on
Windows, and this project runs on a MINGW Python that cannot build it. The suggestion
failed at the installation step. The fix was to replace FAISS with a plain numpy dot
product (`matrix @ query`), which is mathematically identical at this catalog size
and works without any native compilation. The AI's suggestion was correct in principle
but did not account for the specific environment constraint.

---

## 10. Strengths

- The three-stage pipeline gives each component a job it handles well. No single
  component is overloaded with tasks it cannot reliably do.
- Every stage has a graceful fallback so the system never crashes due to a missing
  file, unavailable package, or API error.
- The explanation step makes recommendations transparent — the user can read why
  each song was chosen in their own words, which builds trust and makes errors easy
  to spot.

---

## 11. Ideas for Further Improvement

1. **Incremental model retraining** — currently the LightGBM model must be retrained
   manually on Colab. Adding an automatic retraining trigger when enough new
   interactions accumulate would make the system truly adaptive.
2. **Softer genre similarity** — TF-IDF treats "jazz" and "smooth jazz" as different
   keywords. A genre taxonomy or embedding could capture that they are related.
3. **Session context** — the current system uses the same profile for every session.
   Adding a "mood right now" input at the start of each session (separate from the
   permanent profile) would let the system adapt to how the user feels today, not
   just their general taste.
