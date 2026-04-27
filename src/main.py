"""
src/main.py

Full inference pipeline — Streamlit entry point for ongoing recommendations.

Run with:
    streamlit run src/main.py

Prerequisites:
    1. Complete onboarding: streamlit run src/profile/survey.py
    2. (Optional) Build the knowledge base: python src/rag/build_knowledge_base.py
    3. Set GROQ_API_KEY in your .env file

What happens on each "Refresh Playlist" click (with intent typed):
    1. rag_layer.retrieve_songs()  — embed intent, cosine-search 81k catalog → top-100
    2. ranker.rank()               — ML scoring using user's taste profile → top-50
    3. Gemini                      — writes one-sentence explanations for top-10
    4. Results cached in session_state — Like/Skip buttons work without re-running the pipeline

Without intent: freshness-aware profile retrieval → ML ranking → top-10 (no explanations).
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import streamlit as st
from dotenv import load_dotenv

# Add src/ to sys.path so sibling packages (retrieval, ranking, rag) resolve
# without a top-level __init__.py.  Same approach used by survey.py.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ranking.ranker import model_status
from rag.rag_layer import recommend

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT         = Path(__file__).resolve().parents[1]
DATA_DIR          = REPO_ROOT / "data"
EMBEDDINGS_DIR    = REPO_ROOT / "embeddings"
SONGS_CSV         = DATA_DIR / "songs_full.csv"
CURRENT_USER_PATH = DATA_DIR / "current_user.json"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"

# ── constants ──────────────────────────────────────────────────────────────────
FINAL_K             = 10  # recommendations shown to the user
TOP_GENRES_IN_SUMMARY = 3  # genres to mention in the history sentence
INTERACTION_FIELDS  = ["user_id", "song_id", "label", "play_count", "timestamp"]


# ── CSS (same Spotify-dark theme as survey.py) ────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #121212 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.main .block-container {
    max-width: 640px !important;
    padding: 3rem 2rem 5rem !important;
    margin: 0 auto;
}

/* ── page header ──────────────────────────────────── */
.page-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #1DB954;
    font-weight: 600;
    margin: 0 0 0.5rem;
    display: block;
}
.page-title {
    font-size: 2.25rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    margin: 0 0 0.35rem;
    line-height: 1.1;
}
.page-sub {
    font-size: 0.9rem;
    color: #9B9B9B;
    margin: 0 0 1.5rem;
}

/* ── status badges ─────────────────────────────────── */
.status-badge {
    display: inline-block;
    font-size: 0.7rem;
    color: #6A6A6A;
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    margin: 0 0.2rem 0.2rem 0;
    font-family: monospace;
}

/* ── intent text input ────────────────────────────── */
.stTextInput input {
    background-color: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    font-family: inherit !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.15s;
}
.stTextInput input:focus {
    border-color: #1DB954 !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: #4A4A4A !important; }

/* ── refresh button ───────────────────────────────── */
.stButton > button {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: background-color 0.15s;
}
.stButton > button:hover { background-color: #1ed760 !important; }

/* ── song list heading ────────────────────────────── */
.songs-heading {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 2.5rem 0 0.75rem;
    letter-spacing: -0.01em;
}

/* ── song card ────────────────────────────────────── */
.song-card { padding: 0.85rem 0; border-bottom: 1px solid #1E1E1E; }
.song-num   { font-size: 0.75rem; color: #6A6A6A; padding-top: 0.1rem; }
.song-title  { font-size: 0.97rem; color: #FFFFFF; font-weight: 600;
               display: block; margin-bottom: 0.1rem; }
.song-artist { font-size: 0.82rem; color: #9B9B9B; display: block; }
.song-explanation {
    font-size: 0.81rem; color: #6A6A6A; margin: 0.45rem 0 0 0;
    line-height: 1.55; font-style: italic;
}
.song-liked   { font-size: 0.75rem; color: #1DB954; margin-top: 0.3rem; font-weight: 600; }
.song-skipped { font-size: 0.75rem; color: #535353; margin-top: 0.3rem; }

/* ── like / skip action buttons ───────────────────── */
/*
   These buttons live inside a nested columns layout:
     outer col [info] | outer col [actions → [Like] [Skip]]
   Target with :first-child / :last-child inside the inner horizontal block.
*/
[data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child .stButton > button {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    min-width: 0 !important; width: 100% !important;
    padding: 0.45rem 0 !important;
    font-size: 0.77rem !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child .stButton > button {
    background-color: transparent !important;
    color: #9B9B9B !important;
    border: 1px solid #3A3A3A !important;
    border-radius: 6px !important;
    min-width: 0 !important; width: 100% !important;
    padding: 0.45rem 0 !important;
    font-size: 0.77rem !important; font-weight: 400 !important;
    letter-spacing: 0.04em !important;
}

/* ── info box (setup messages) ────────────────────── */
.info-box {
    background-color: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    font-size: 0.9rem;
    color: #9B9B9B;
    line-height: 1.6;
    margin: 2rem 0;
}
.info-box code {
    background-color: #282828;
    border-radius: 3px;
    padding: 0.1rem 0.4rem;
    font-size: 0.85rem;
    color: #FFFFFF;
}

/* ── hide Streamlit chrome ────────────────────────── */
#MainMenu  { visibility: hidden !important; }
footer     { visibility: hidden !important; }
header     { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
"""


# ── data helpers ───────────────────────────────────────────────────────────────

def load_user_profile() -> Optional[Dict]:
    """Load the most recently confirmed profile from data/current_user.json."""
    if not CURRENT_USER_PATH.exists():
        return None
    with open(CURRENT_USER_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_song_genre_map() -> Dict[str, str]:
    """
    Build a song_id → genre lookup from songs_full.csv.

    Cached by Streamlit: read once on first call, reused for every history
    summary within the session.  The 81k-row CSV stays in memory as a plain dict.
    """
    genre_map: Dict[str, str] = {}
    if not SONGS_CSV.exists():
        return genre_map
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = row.get("song_id", "").strip()
            if sid:
                genre_map[sid] = row.get("genre", "unknown").strip()
    return genre_map


def build_history_summary(user_id: str) -> str:
    """
    Build a one-sentence description of this user's listening history.

    Used as context in the Gemini prompt so it can reason about the user's
    taste when filtering and explaining recommendations.

    Example output:
        "Liked 12 songs. Skipped 3. Mostly chill, ambient, classical."
    """
    if not INTERACTIONS_PATH.exists():
        return "No listening history yet."

    genre_map     = _load_song_genre_map()
    liked_count   = 0
    skipped_count = 0
    liked_genres: Dict[str, int] = {}

    with open(INTERACTIONS_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("user_id", "").strip() != user_id:
                continue
            sid   = row.get("song_id", "").strip()
            label = int(row.get("label", 0))
            if label == 1:
                liked_count += 1
                genre = genre_map.get(sid, "unknown")
                liked_genres[genre] = liked_genres.get(genre, 0) + 1
            else:
                skipped_count += 1

    if liked_count == 0 and skipped_count == 0:
        return "No listening history yet."

    parts = [f"Liked {liked_count} song{'s' if liked_count != 1 else ''}."]
    if skipped_count:
        parts.append(f"Skipped {skipped_count}.")
    if liked_genres:
        top = sorted(liked_genres, key=lambda g: -liked_genres[g])[:TOP_GENRES_IN_SUMMARY]
        parts.append(f"Mostly {', '.join(top)}.")

    return " ".join(parts)


def _append_interaction(user_id: str, song_id: str, label: int) -> None:
    """
    Append one like (label=1) or skip (label=0) row to interactions.csv.

    This is the same format as survey.py's save_interactions(), kept inline
    here so main.py has no import dependency on Streamlit's survey module.
    """
    file_exists = INTERACTIONS_PATH.exists()
    with open(INTERACTIONS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INTERACTION_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "user_id":    user_id,
            "song_id":    song_id,
            "label":      label,
            "play_count": 1 if label == 1 else 0,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


# ── pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(
    user_profile: Dict,
    user_intent: str,
    api_key: str,
) -> tuple:
    """
    Execute the recommendation pipeline for one playlist refresh.

    With intent (user typed something):
        1. Embed intent → cosine-search 81k song catalog → top-100 candidates  [RAG]
        2. ML-rank candidates using the user's taste profile                    [ranker]
        3. Gemini writes one-sentence explanations for top-10                   [LLM]

    Without intent (empty text box):
        1. Freshness-aware profile retrieval → up to 200 candidates             [retriever]
        2. ML-rank → top-10                                                     [ranker]
        (No explanations — nothing specific to explain without a query)

    All of this is handled inside rag_layer.recommend().  run_pipeline() is just
    the entry point that passes the right arguments and handles the error case
    where both retrieval paths return empty (rare — means embeddings are missing).

    Returns:
        (results: List[Dict], None)    on success
        (None, error_message: str)     on failure
    """
    history_summary = build_history_summary(user_profile["user_id"])
    results = recommend(user_intent, user_profile, history_summary, api_key, k=FINAL_K)

    if not results:
        return None, (
            "No songs found. If you typed a request, the knowledge base may not be built yet. "
            "Run: python src/rag/build_knowledge_base.py"
        )
    return results, None


# ── UI components ──────────────────────────────────────────────────────────────

def _render_song_card(
    song: Dict,
    rank_num: int,
    user_id: str,
    liked_ids: Set[str],
    skipped_ids: Set[str],
) -> None:
    """
    Render one song card with title, artist, explanation, and Like/Skip buttons.

    After the user interacts with a song, the song_id is stored in session state
    and a visual label appears ("Liked" / "Skipped") instead of the buttons.
    The playlist is not re-run — the user refreshes when they want new songs.
    """
    sid         = song.get("song_id", "")
    title       = song.get("title", "Unknown")
    artist      = song.get("artist", "Unknown")
    explanation = song.get("explanation", "")

    already_liked   = sid in liked_ids
    already_skipped = sid in skipped_ids

    col_info, col_actions = st.columns([8, 2])

    with col_info:
        info_html = (
            f'<div class="song-card">'
            f'  <div style="display:flex;gap:0.7rem;align-items:flex-start;">'
            f'    <span class="song-num">{rank_num}</span>'
            f'    <div>'
            f'      <span class="song-title">{title}</span>'
            f'      <span class="song-artist">{artist}</span>'
            f'    </div>'
            f'  </div>'
        )
        if explanation:
            info_html += f'<div class="song-explanation">{explanation}</div>'
        if already_liked:
            info_html += '<div class="song-liked">Liked</div>'
        elif already_skipped:
            info_html += '<div class="song-skipped">Skipped</div>'
        info_html += "</div>"
        st.markdown(info_html, unsafe_allow_html=True)

    with col_actions:
        # Only show buttons if user hasn't interacted with this song yet
        if not already_liked and not already_skipped:
            sub_like, sub_skip = st.columns([1, 1])
            with sub_like:
                if st.button("Like", key=f"like_{sid}"):
                    _append_interaction(user_id, sid, label=1)
                    st.session_state.liked_ids.add(sid)
                    st.rerun()
            with sub_skip:
                if st.button("Skip", key=f"skip_{sid}"):
                    _append_interaction(user_id, sid, label=0)
                    st.session_state.skipped_ids.add(sid)
                    st.rerun()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Music Recommendations", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)

    # ── load prerequisites ─────────────────────────────────────────────────────
    user_profile = load_user_profile()
    if user_profile is None:
        st.markdown(
            '<div class="info-box">'
            "No user profile found. Run the onboarding survey first:"
            "<br><br><code>streamlit run src/profile/survey.py</code>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.markdown(
            '<div class="info-box">'
            "GROQ_API_KEY not found. Add it to your <code>.env</code> file:"
            "<br><br><code>GROQ_API_KEY=your_key_here</code>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── session state initialisation ───────────────────────────────────────────
    if "liked_ids"   not in st.session_state:
        st.session_state.liked_ids   = set()
    if "skipped_ids" not in st.session_state:
        st.session_state.skipped_ids = set()

    # ── page header ────────────────────────────────────────────────────────────
    name = user_profile.get("name", "")
    greeting = f"Welcome back, {name}" if name else "Welcome back"
    st.markdown(
        f'<span class="page-eyebrow">{greeting}</span>'
        f'<h1 class="page-title">Your Playlist</h1>'
        f'<p class="page-sub">Refresh any time for personalised picks.</p>',
        unsafe_allow_html=True,
    )

    # ── status badges — ranker mode and KB availability ────────────────────────
    kb_path  = EMBEDDINGS_DIR / "knowledge_base_vectors.npy"
    kb_label = "KB: ready" if kb_path.exists() else "KB: not built"
    st.markdown(
        f'<span class="status-badge">Ranker: {model_status()}</span>'
        f'<span class="status-badge">{kb_label}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    # ── intent text input ──────────────────────────────────────────────────────
    user_intent = st.text_input(
        "intent",
        value=st.session_state.get("last_intent", ""),
        placeholder="What are you in the mood for? (e.g. studying, calm focus, no lyrics)",
        label_visibility="collapsed",
    )

    refresh_clicked = st.button("Refresh Playlist")

    if refresh_clicked:
        with st.spinner("Finding your music …"):
            results, error = run_pipeline(user_profile, user_intent, api_key)

        if error:
            st.error(error)
        else:
            # Cache results; reset per-session interaction state
            st.session_state.recommendations = results
            st.session_state.last_intent     = user_intent
            st.session_state.liked_ids       = set()
            st.session_state.skipped_ids     = set()
            st.rerun()

    # ── recommendation cards ───────────────────────────────────────────────────
    if "recommendations" in st.session_state:
        recs        = st.session_state.recommendations
        last_intent = st.session_state.get("last_intent", "")
        heading     = "Picks for you" + (f' — "{last_intent}"' if last_intent else "")

        st.markdown(f'<p class="songs-heading">{heading}</p>', unsafe_allow_html=True)

        for i, song in enumerate(recs, 1):
            _render_song_card(
                song, i,
                user_profile["user_id"],
                st.session_state.liked_ids,
                st.session_state.skipped_ids,
            )
    else:
        st.markdown(
            '<div class="info-box" style="margin-top:2rem;">'
            "Click <strong>Refresh Playlist</strong> to generate your recommendations."
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
