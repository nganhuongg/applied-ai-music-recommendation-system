"""
src/profile/survey.py

Multi-step onboarding survey that collects user music preferences
and saves a structured UserProfile JSON for the recommendation pipeline.

Run with:
    streamlit run src/profile/survey.py
"""

import csv
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv

# survey.py is run directly by Streamlit, so it is not part of a package.
# We add the src/ directory to sys.path so Python can find the sibling
# `recommender` package without requiring a src/__init__.py file.
_SRC_DIR = Path(__file__).resolve().parents[1]  # .../src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from recommender.initial import recommend
from embeddings.user_embedding import compute_user_embedding

# ── paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR     = REPO_ROOT / "data"
PROFILES_DIR = DATA_DIR / "user_profiles"
SONGS_CSV    = DATA_DIR / "songs_full.csv"

# ── constants ──────────────────────────────────────────────────────────────────

TOTAL_STEPS      = 8
STEP_SONG_SEARCH = 11   # fallback step when user skipped all recommendations
STEP_PLAYLIST    = 20   # playlist view — full inference pipeline

# Genres the user will recognise — subset of the 113 in songs_full.csv.
GENRE_VOCABULARY: List[str] = sorted([
    "acoustic", "alternative", "ambient", "anime", "blues",
    "chill", "classical", "country", "dance", "edm",
    "electronic", "folk", "funk", "gospel", "hip-hop",
    "house", "indie", "indie-pop", "j-pop", "jazz",
    "k-pop", "latin", "metal", "pop", "punk",
    "r-n-b", "reggae", "rock", "soul", "synth-pop",
])

# Each mood maps to audio-feature targets used by the ranking model.
# Keys are what the user sees; values are internal.
MOOD_FEATURES: Dict[str, Dict[str, float]] = {
    "Calm":        {"target_energy": 0.30, "target_valence": 0.50},
    "Energetic":   {"target_energy": 0.80, "target_valence": 0.70},
    "Upbeat":      {"target_energy": 0.60, "target_valence": 0.85},
    "Melancholic": {"target_energy": 0.30, "target_valence": 0.20},
    "Intense":     {"target_energy": 0.85, "target_valence": 0.25},
}

# (internal key, user-visible label)
PURPOSE_OPTIONS: List[Tuple[str, str]] = [
    ("study",   "Studying"),
    ("workout", "Working out"),
    ("relax",   "Relaxing"),
    ("commute", "Commuting"),
    ("explore", "Exploring new music"),
]

# (internal key, user-visible label, exploration_weight float)
EXPLORATION_OPTIONS: List[Tuple[str, str, float]] = [
    ("familiar", "Mostly songs I already know", 0.1),
    ("mixed",    "A bit of both",               0.5),
    ("discover", "I love finding new music",    0.9),
]

# (internal key, user-visible label)
POPULARITY_OPTIONS: List[Tuple[str, str]] = [
    ("mainstream", "Popular hits"),
    ("mixed",      "A mix of popular and underground"),
    ("underground","Less mainstream music"),
]

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── base ─────────────────────────────────────────── */
html, body, .stApp {
    background-color: #121212 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.main .block-container {
    max-width: 560px !important;
    padding: 3rem 2rem 5rem !important;
    margin: 0 auto;
}

/* ── progress bar ─────────────────────────────────── */
.stProgress {
    margin-bottom: 0 !important;
}
.stProgress > div > div {
    background-color: #282828 !important;
    height: 2px !important;
    border-radius: 0 !important;
}
.stProgress > div > div > div {
    background-color: #1DB954 !important;
    height: 2px !important;
    border-radius: 0 !important;
    transition: width 0.4s ease;
}

/* ── question typography ──────────────────────────── */
.q-counter {
    font-size: 0.7rem;
    color: #6A6A6A;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 1.75rem 0 1.25rem;
    display: block;
}

.q-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
}

.q-sub {
    font-size: 0.875rem;
    color: #9B9B9B;
    margin: 0 0 1.75rem;
    font-weight: 400;
}

/* ── text input ───────────────────────────────────── */
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

/* ── multiselect ──────────────────────────────────── */
.stMultiSelect [data-baseweb="select"] > div {
    background-color: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 6px !important;
    transition: border-color 0.15s;
}
.stMultiSelect [data-baseweb="select"] > div:focus-within {
    border-color: #1DB954 !important;
}
/* Selected tags */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #1DB954 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    border-radius: 3px !important;
}
/* Search text inside multiselect */
.stMultiSelect input { color: #FFFFFF !important; }
/* Dropdown list */
[data-baseweb="menu"] { background-color: #1E1E1E !important; }
[data-baseweb="menu"] li { color: #FFFFFF !important; }
[data-baseweb="menu"] li:hover { background-color: #2A2A2A !important; }

/* ── radio buttons ────────────────────────────────── */
/* Hide the default Streamlit radio label — we put our own title above */
.stRadio > label { display: none !important; }

.stRadio > div { gap: 6px !important; }

/* Style each option as a selectable row */
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
}

/* ── count feedback ───────────────────────────────── */
.count-ok   { font-size: 0.8rem; color: #1DB954; margin-top: 0.6rem; }
.count-hint { font-size: 0.8rem; color: #6A6A6A; margin-top: 0.6rem; }

/* ── primary button (Continue / Done) ─────────────── */
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
    transition: background-color 0.15s, opacity 0.15s;
}
.stButton > button:hover { background-color: #1ed760 !important; }
.stButton > button:disabled {
    background-color: #2A2A2A !important;
    color: #5A5A5A !important;
    cursor: not-allowed !important;
}

/* ── nav buttons (Back / Continue) ───────────────── */
/*
   st.columns() is only called inside render_nav, so targeting
   [data-testid="stColumn"] safely scopes every rule below to the nav row.
*/

/* First column → Back, left edge */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child .stButton > button {
    width: auto !important;
    min-width: 100px !important;
    background-color: transparent !important;
    color: #9B9B9B !important;
    border: 1px solid #2A2A2A !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child .stButton > button:hover {
    color: #FFFFFF !important;
    border-color: #535353 !important;
    background-color: transparent !important;
}

/* Second column → Continue, right edge */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child .stButton > button {
    width: auto !important;
    min-width: 110px !important;
}

/* ── done screen ──────────────────────────────────── */
.done-title {
    font-size: 2.25rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    margin: 2rem 0 0.4rem;
    line-height: 1.15;
}
.done-sub {
    font-size: 0.975rem;
    color: #9B9B9B;
    margin-bottom: 2.5rem;
}
.recap-label {
    font-size: 0.75rem;
    color: #6A6A6A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.3rem;
}
.recap-value {
    font-size: 0.975rem;
    color: #FFFFFF;
    border-bottom: 1px solid #1E1E1E;
    padding-bottom: 0.9rem;
}

/* ── divider ──────────────────────────────────────── */
hr { border-color: #1E1E1E !important; margin: 1.5rem 0 !important; }

/* ── landing page ─────────────────────────────────── */
.landing-wrap {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding-top: 18vh;
    padding-bottom: 4rem;
}

.landing-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #1DB954;
    font-weight: 600;
    margin: 0 0 1.4rem;
}

.landing-title {
    font-size: clamp(2.4rem, 6vw, 3.5rem);
    font-weight: 800;
    color: #FFFFFF !important;
    line-height: 1.1;
    letter-spacing: -0.035em;
    margin: 0 0 1.5rem;
}

.landing-desc {
    font-size: 1rem;
    color: #9B9B9B;
    line-height: 1.65;
    margin: 0 0 2.75rem;
    max-width: 380px;
}

/* Landing button — wider than survey buttons */
.landing-btn .stButton > button {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.07em !important;
    padding: 0.75rem 2.75rem !important;
    width: auto !important;
}
.landing-btn .stButton > button:hover {
    background-color: #1ed760 !important;
    transform: scale(1.03);
}


/* ── song-list checkboxes ──────────────────────────── */
/* Align the checkbox vertically with the song title */
.stCheckbox {
    display: flex !important;
    align-items: center !important;
    padding-top: 0.85rem !important;
    padding-bottom: 0 !important;
}
/* Green accent colour for the tick */
.stCheckbox input[type="checkbox"] {
    accent-color: #1DB954 !important;
    width: 1.1rem !important;
    height: 1.1rem !important;
    cursor: pointer !important;
}

/* ── song recommendation cards ────────────────────── */
.songs-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 2.5rem 0 1rem;
    letter-spacing: -0.01em;
}

.song-card {
    display: grid;
    grid-template-columns: 1.5rem 1fr auto;
    gap: 0.75rem;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid #1E1E1E;
}
.song-num {
    font-size: 0.75rem;
    color: #6A6A6A;
    text-align: right;
}
.song-title {
    font-size: 0.95rem;
    color: #FFFFFF;
    font-weight: 500;
    display: block;
    margin-bottom: 0.15rem;
}
.song-artist {
    font-size: 0.8rem;
    color: #9B9B9B;
    display: block;
}
.song-genre {
    font-size: 0.75rem;
    color: #6A6A6A;
}

/* ── song-feedback card ───────────────────────────── */
/* Used in the like/skip screen — simpler than .song-card (no rank number) */
.song-card-feedback {
    padding: 0.65rem 0;
    border-bottom: 1px solid #1E1E1E;
}

/* ── like / skip action buttons ───────────────────── */
/*
   These buttons live inside nested st.columns() (a col inside another col).
   That nesting means their stHorizontalBlock has a stColumn parent,
   which is NOT true for the nav row — so these rules don't affect
   the Back / Continue nav buttons.

   Layout per song row:  [8 info] [2 actions → [1 Like] [1 Skip]]
   :first-child = Like → green
   :last-child  = Skip → ghost
*/
[data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child .stButton > button {
    background-color: #1DB954 !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    min-width: 0 !important;
    width: 100% !important;
    padding: 0.55rem 0 !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child .stButton > button {
    background-color: transparent !important;
    color: #9B9B9B !important;
    border: 1px solid #3A3A3A !important;
    border-radius: 6px !important;
    min-width: 0 !important;
    width: 100% !important;
    padding: 0.55rem 0 !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
}

/* ── hide Streamlit chrome ────────────────────────── */
#MainMenu  { visibility: hidden !important; }
footer     { visibility: hidden !important; }
header     { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
</style>
"""

# ── data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_top_artists(limit: int = 500) -> List[str]:
    """
    Return the top `limit` artists by peak popularity from songs_full.csv,
    sorted alphabetically for display.
    """
    artist_max_pop: Dict[str, float] = {}
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            for raw in row["artist_name"].split(";"):
                name = raw.strip()
                if not name:
                    continue
                try:
                    pop = float(row["popularity"])
                except ValueError:
                    pop = 0.0
                if name not in artist_max_pop or artist_max_pop[name] < pop:
                    artist_max_pop[name] = pop

    top = sorted(artist_max_pop.items(), key=lambda x: -x[1])[:limit]
    return sorted(name for name, _ in top)   # alphabetical


@st.cache_data(show_spinner=False)
def load_all_songs_for_search() -> List[Dict]:
    """
    Load the minimal fields needed for the search fallback step.
    Cached once per session — 81k rows stay in memory, no re-read per keystroke.
    """
    songs = []
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            songs.append({
                "song_id":     row["song_id"],
                "title":       row["title"],
                "artist_name": row["artist_name"],
                "genre":       row["genre"],
            })
    return songs


# ── profile helpers ────────────────────────────────────────────────────────────

def build_genre_vector(selected_genres: List[str]) -> List[float]:
    """One-hot (multi-hot) encoding over GENRE_VOCABULARY."""
    return [1.0 if g in selected_genres else 0.0 for g in GENRE_VOCABULARY]


def build_mood_targets(selected_moods: List[str]) -> Dict[str, float]:
    """Average energy and valence targets across all selected moods."""
    if not selected_moods:
        return {"target_energy": 0.5, "target_valence": 0.5}
    energies = [MOOD_FEATURES[m]["target_energy"] for m in selected_moods]
    valences  = [MOOD_FEATURES[m]["target_valence"] for m in selected_moods]
    return {
        "target_energy": round(sum(energies) / len(energies), 3),
        "target_valence": round(sum(valences) / len(valences), 3),
    }


def exploration_weight_from_key(key: str) -> float:
    for k, _, w in EXPLORATION_OPTIONS:
        if k == key:
            return w
    return 0.5


def build_profile(answers: Dict) -> Dict:
    """Assemble the full UserProfile dict (format defined in PLAN.md §1.2)."""
    return {
        "user_id":               str(uuid.uuid4()),
        "name":                  answers.get("name", ""),
        "created_at":            datetime.now().isoformat(),
        "favorite_artists":      answers.get("favorite_artists", []),
        "recent_artists":        answers.get("recent_artists", []),
        "favorite_genres":       answers.get("favorite_genres", []),
        "selected_moods":        answers.get("moods", []),
        "genre_vector":          build_genre_vector(answers.get("favorite_genres", [])),
        "mood_targets":          build_mood_targets(answers.get("moods", [])),
        "purpose":               answers.get("purpose", "relax"),
        "exploration_weight":    exploration_weight_from_key(answers.get("exploration", "mixed")),
        "popularity_preference": answers.get("popularity", "mixed"),
    }


def save_profile(profile: Dict) -> None:
    """Persist to data/user_profiles/{uuid}.json and data/current_user.json."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    (PROFILES_DIR / f"{profile['user_id']}.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    (DATA_DIR / "current_user.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )


def save_interactions(songs: List[Dict], user_id: str, label: int = 1) -> None:
    """
    Append song interactions to data/interactions.csv.

    label=1 means liked, label=0 means skipped.
    Both are recorded so the ranking model can learn from negative examples too.
    """
    interactions_csv = DATA_DIR / "interactions.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fieldnames = ["user_id", "song_id", "label", "play_count", "timestamp"]

    file_exists = interactions_csv.exists()
    with open(interactions_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for song in songs:
            writer.writerow({
                "user_id":    user_id,
                "song_id":    song["song_id"],
                "label":      label,
                "play_count": 1 if label == 1 else 0,
                "timestamp":  timestamp,
            })


# ── UI helpers ─────────────────────────────────────────────────────────────────

def render_progress(step: int) -> None:
    """Thin green progress line + step counter."""
    st.progress(step / TOTAL_STEPS)
    st.markdown(
        f'<span class="q-counter">{step} of {TOTAL_STEPS}</span>',
        unsafe_allow_html=True,
    )


def render_nav(step: int, can_proceed: bool, next_label: str = "Continue") -> None:
    """Back left-aligned, Continue right-aligned, same row."""
    col_back, col_next = st.columns(2)
    with col_back:
        if step > 0:
            if st.button("Back", key="btn_back"):
                st.session_state.step -= 1
                st.rerun()
    with col_next:
        if st.button(next_label, key="btn_next", disabled=not can_proceed):
            st.session_state.step += 1
            st.rerun()


def html(tag_class: str, text: str) -> None:
    """Shorthand to emit a styled HTML element."""
    st.markdown(f'<p class="{tag_class}">{text}</p>', unsafe_allow_html=True)


# ── survey steps ───────────────────────────────────────────────────────────────

def step_0_landing() -> None:
    """Landing page — title, one-line description, single call-to-action."""
    st.markdown(
        """
        <div class="landing-wrap">
            <p class="landing-eyebrow">Personalised for you</p>
            <h1 class="landing-title">Music Recommendation<br>System</h1>
            <p class="landing-desc">
                Answer eight quick questions about your taste
                and we will build a profile that shapes every
                recommendation you receive.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="landing-btn">', unsafe_allow_html=True)
    if st.button("Get started", key="btn_start"):
        st.session_state.step = 1
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def step_1_name() -> None:
    render_progress(1)
    html("q-title", "What should we call you?")
    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    name = st.text_input(
        "name",
        value=st.session_state.answers.get("name", ""),
        placeholder="Your name",
        max_chars=60,
        label_visibility="collapsed",
    )
    st.session_state.answers["name"] = name.strip()

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=1, can_proceed=bool(name.strip()))


def step_2_favorite_artists(all_artists: List[str]) -> None:
    render_progress(2)
    html("q-title", "Which artists do you listen to most?")
    html("q-sub",   "Choose between 3 and 5.")

    selected = st.multiselect(
        "artists",
        options=all_artists,
        default=st.session_state.answers.get("favorite_artists", []),
        placeholder="Type an artist name to search",
        label_visibility="collapsed",
        key="ms_fav_artists",
    )
    st.session_state.answers["favorite_artists"] = selected

    count = len(selected)
    if count == 0:
        html("count-hint", "Start typing to search from 500 artists, sorted A to Z.")
    elif count < 3:
        html("count-hint", f"{count} selected — pick at least {3 - count} more.")
    elif count > 5:
        html("count-hint", f"{count} selected — please remove some (maximum 5).")
    else:
        html("count-ok", f"{count} selected.")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=2, can_proceed=(3 <= count <= 5))


def step_3_recent_artists(all_artists: List[str]) -> None:
    render_progress(3)
    html("q-title", "What have you been listening to lately?")
    html("q-sub",   "Choose 2 or 3 artists.")

    selected = st.multiselect(
        "recent",
        options=all_artists,
        default=st.session_state.answers.get("recent_artists", []),
        placeholder="Type an artist name to search",
        label_visibility="collapsed",
        key="ms_recent",
    )
    st.session_state.answers["recent_artists"] = selected

    count = len(selected)
    if count == 0:
        html("count-hint", "Recent listening tells us what you are into right now.")
    elif count < 2:
        html("count-hint", f"{count} selected — pick at least 1 more.")
    elif count > 3:
        html("count-hint", f"{count} selected — please keep it to 3.")
    else:
        html("count-ok", f"{count} selected.")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=3, can_proceed=(2 <= count <= 3))


def step_4_genres() -> None:
    render_progress(4)
    html("q-title", "Which genres do you enjoy?")
    html("q-sub",   "Choose as many as you like.")

    selected = st.multiselect(
        "genres",
        options=GENRE_VOCABULARY,
        default=st.session_state.answers.get("favorite_genres", []),
        placeholder="Type a genre to search",
        label_visibility="collapsed",
        key="ms_genres",
    )
    st.session_state.answers["favorite_genres"] = selected

    count = len(selected)
    if count == 0:
        html("count-hint", "Pick at least one to continue.")
    else:
        html("count-ok", f"{count} selected.")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=4, can_proceed=(count >= 1))


def step_5_moods() -> None:
    render_progress(5)
    html("q-title", "How do you like your music to feel?")
    html("q-sub",   "Choose one or more.")

    selected = st.multiselect(
        "moods",
        options=list(MOOD_FEATURES.keys()),
        default=st.session_state.answers.get("moods", []),
        placeholder="Select a vibe",
        label_visibility="collapsed",
        key="ms_moods",
    )
    st.session_state.answers["moods"] = selected

    count = len(selected)
    if count == 0:
        html("count-hint", "Pick at least one to continue.")
    else:
        html("count-ok", f"{count} selected.")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=5, can_proceed=(count >= 1))


def step_6_purpose() -> None:
    render_progress(6)
    html("q-title", "When do you usually listen to music?")
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    labels = [label for _, label in PURPOSE_OPTIONS]
    keys   = [key   for key, _  in PURPOSE_OPTIONS]

    saved   = st.session_state.answers.get("purpose", "relax")
    default = keys.index(saved) if saved in keys else 2

    choice = st.radio("purpose", options=labels, index=default,
                      label_visibility="collapsed", key="radio_purpose")
    st.session_state.answers["purpose"] = keys[labels.index(choice)]

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=6, can_proceed=True)


def step_7_exploration() -> None:
    render_progress(7)
    html("q-title", "Do you prefer familiar music or new discoveries?")
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    labels  = [label for _, label, _ in EXPLORATION_OPTIONS]
    keys    = [key   for key, _, _   in EXPLORATION_OPTIONS]

    saved   = st.session_state.answers.get("exploration", "mixed")
    default = keys.index(saved) if saved in keys else 1

    choice = st.radio("exploration", options=labels, index=default,
                      label_visibility="collapsed", key="radio_exploration")
    st.session_state.answers["exploration"] = keys[labels.index(choice)]

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=7, can_proceed=True)


def step_8_popularity() -> None:
    render_progress(8)
    html("q-title", "What kind of music do you go for?")
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    labels  = [label for _, label in POPULARITY_OPTIONS]
    keys    = [key   for key, _   in POPULARITY_OPTIONS]

    saved   = st.session_state.answers.get("popularity", "mixed")
    default = keys.index(saved) if saved in keys else 1

    choice = st.radio("popularity", options=labels, index=default,
                      label_visibility="collapsed", key="radio_popularity")
    st.session_state.answers["popularity"] = keys[labels.index(choice)]

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=8, can_proceed=True, next_label="See my profile")


def step_confirm_profile() -> None:
    """
    Profile confirmation screen (step 9).

    Shows the user a summary of everything they entered so they can review
    before committing.  "Edit" goes back to step 8; "Confirm" advances to
    the song-feedback screen.

    We also clear any leftover feedback state here so that going back from
    the song screen and re-confirming always starts completely fresh.
    """
    for key in (
        "saved_profile", "remaining_songs", "liked_count", "top_songs",
        "liked_songs", "user_embedding_computed", "search_selected",
    ):
        st.session_state.pop(key, None)

    a = st.session_state.answers

    st.markdown('<p class="done-title">Your music profile.</p>', unsafe_allow_html=True)
    html("done-sub", "Does this look right? Edit to go back, or confirm to see your songs.")

    def recap(label: str, value: str) -> None:
        html("recap-label", label)
        html("recap-value", value or "—")

    recap("Artists you love",      ", ".join(a.get("favorite_artists", [])))
    recap("Recently listening to", ", ".join(a.get("recent_artists",   [])))
    recap("Genres",                ", ".join(a.get("favorite_genres",  [])))
    recap("Vibe",                  ", ".join(a.get("moods",            [])))

    purpose_map     = dict(PURPOSE_OPTIONS)
    exploration_map = {k: l for k, l, _ in EXPLORATION_OPTIONS}
    popularity_map  = dict(POPULARITY_OPTIONS)

    recap("You listen for",       purpose_map.get(    a.get("purpose",     ""), ""))
    recap("Music explorer level", exploration_map.get(a.get("exploration", ""), ""))
    recap("You prefer",           popularity_map.get( a.get("popularity",  ""), ""))

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    render_nav(step=9, can_proceed=True, next_label="Confirm →")


def step_song_feedback() -> None:
    """
    Song feedback screen (step 10+).

    Shows the 10 recommended songs one by one.  The user likes or skips each
    one; both actions are recorded in interactions.csv (label=1 / label=0)
    and the song is removed from the visible list immediately.

    When the list is empty the user sees a completion message.
    """
    # Save profile on first visit (user just confirmed on step 9)
    if "saved_profile" not in st.session_state:
        profile = build_profile(st.session_state.answers)
        save_profile(profile)
        st.session_state.saved_profile = profile

    # Load recommendations once and keep them in session state.
    # Without caching, recommend() would re-score thousands of songs on
    # every button click (Streamlit reruns the whole script per interaction).
    if "remaining_songs" not in st.session_state:
        st.session_state.remaining_songs = recommend(
            st.session_state.saved_profile, top_n=10
        )
        st.session_state.liked_count = 0
        st.session_state.liked_songs  = []   # tracks song_ids for embedding computation

    remaining = st.session_state.remaining_songs
    user_id   = st.session_state.saved_profile["user_id"]

    # ── completion state ───────────────────────────────────────────────────────
    if not remaining:
        count = st.session_state.liked_count

        # User skipped every song — send them to the manual search fallback
        # so we still get enough signal to compute a meaningful embedding.
        if count == 0:
            st.session_state.step = STEP_SONG_SEARCH
            st.rerun()
            return

        # Compute user embedding from liked songs.
        # The flag ensures we only write the .npy file once even if the user
        # stays on this screen across multiple Streamlit reruns.
        if "user_embedding_computed" not in st.session_state:
            compute_user_embedding(
                st.session_state.liked_songs,
                profile=st.session_state.saved_profile,
            )
            st.session_state.user_embedding_computed = True

        song_label = "song" if count == 1 else "songs"
        st.markdown('<p class="done-title">All done!</p>', unsafe_allow_html=True)
        html(
            "done-sub",
            f"You liked {count} {song_label}. Ready to see your personalised playlist?",
        )
        st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
        # col_restart → ghost (Back style), col_playlist → green (Continue style)
        col_restart, col_playlist = st.columns(2)
        with col_restart:
            if st.button("Start over", key="btn_restart"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        with col_playlist:
            if st.button("See my playlist →", key="btn_to_playlist"):
                st.session_state.step = STEP_PLAYLIST
                st.rerun()
        return

    # ── song list ──────────────────────────────────────────────────────────────
    html("songs-heading", "What do you think of these songs?")
    html("done-sub", f"{len(remaining)} remaining — like the ones you enjoy, skip the rest.")
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)

    # Iterate over a copy of the list so in-loop removal doesn't cause issues.
    for song in list(remaining):
        col_info, col_actions = st.columns([8, 2])

        with col_info:
            st.markdown(
                f'<div class="song-card-feedback">'
                f'  <span class="song-title">{song["title"]}</span>'
                f'  <span class="song-artist">{song["artist_name"]}</span>'
                f'  <span class="song-genre">{song["genre"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_actions:
            # Nest two columns inside col_actions.
            # :first-child → Like (green via CSS)
            # :last-child  → Skip (ghost via CSS)
            sub_like, sub_skip = st.columns([1, 1])

            with sub_like:
                if st.button("Like", key=f"like_{song['song_id']}"):
                    save_interactions([song], user_id, label=1)
                    st.session_state.remaining_songs = [
                        s for s in remaining if s["song_id"] != song["song_id"]
                    ]
                    st.session_state.liked_count += 1
                    st.session_state.liked_songs.append(song["song_id"])
                    st.rerun()

            with sub_skip:
                if st.button("Skip", key=f"skip_{song['song_id']}"):
                    # label=0 records the skip as a negative example for training
                    save_interactions([song], user_id, label=0)
                    st.session_state.remaining_songs = [
                        s for s in remaining if s["song_id"] != song["song_id"]
                    ]
                    st.rerun()


def step_song_search(all_songs: List[Dict]) -> None:
    """
    Search fallback (step 11) — shown when the user skipped all recommendations.

    The user searches for 2–5 songs they already know they like.  Those songs
    are saved as liked interactions and used to compute the first user embedding.

    Selection persists across query changes: check a song → it moves to the
    "Selected" section at the top; click ✕ to remove it.
    """
    html("q-title", "Let us recalibrate.")
    html(
        "done-sub",
        "You skipped everything — no worries. "
        "Search for 2 to 5 songs you already enjoy and we will use them to shape your picks.",
    )
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)

    # selected maps song_id → song dict so we can show titles and save them later
    if "search_selected" not in st.session_state:
        st.session_state.search_selected = {}
    selected: Dict[str, Dict] = st.session_state.search_selected

    # ── show already-selected songs with remove buttons ────────────────────
    if selected:
        html("recap-label", "Selected songs")
        for song_id, song in list(selected.items()):
            col_info, col_rm = st.columns([10, 1])
            with col_info:
                st.markdown(
                    f'<span class="song-title">{song["title"]}</span>'
                    f'<span class="song-artist"> — {song["artist_name"]}</span>',
                    unsafe_allow_html=True,
                )
            with col_rm:
                if st.button("✕", key=f"rm_{song_id}"):
                    del st.session_state.search_selected[song_id]
                    # Clear checkbox widget state so the song reappears unchecked
                    # if the user searches for it again after removing it.
                    st.session_state.pop(f"srch_{song_id}", None)
                    st.rerun()
        st.markdown("---")

    # ── search bar ─────────────────────────────────────────────────────────
    query = st.text_input(
        "search",
        placeholder="Search by title or artist name",
        label_visibility="collapsed",
        key="search_input",
    )

    # ── results ────────────────────────────────────────────────────────────
    if query.strip():
        q = query.lower()
        results = [
            s for s in all_songs
            if (q in s["title"].lower() or q in s["artist_name"].lower())
            and s["song_id"] not in selected
        ][:20]

        if not results:
            html("count-hint", "No results — try a different title or artist name.")

        for song in results:
            label  = f"{song['title']} — {song['artist_name']}"
            cb_key = f"srch_{song['song_id']}"
            if st.checkbox(label, key=cb_key):
                # Move song from results to selected and clear widget state
                # so it does not reappear as checked if removed later.
                st.session_state.search_selected[song["song_id"]] = song
                del st.session_state[cb_key]
                st.rerun()

    # ── count + proceed ────────────────────────────────────────────────────
    count = len(selected)
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    if 2 <= count <= 5:
        html("count-ok", f"{count} {'song' if count == 1 else 'songs'} selected.")
    elif count > 5:
        html("count-hint", f"{count} selected — please remove some (maximum 5).")
    elif count > 0:
        html("count-hint", f"{count} selected — add {2 - count} more to continue.")
    else:
        html("count-hint", "Select 2 to 5 songs to continue.")

    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    if st.button("Build my profile", disabled=not (2 <= count <= 5), key="btn_search_confirm"):
        user_id = st.session_state.saved_profile["user_id"]
        save_interactions(list(selected.values()), user_id, label=1)
        compute_user_embedding(
            list(selected.keys()),
            profile=st.session_state.saved_profile,
        )
        st.session_state.step = STEP_SONG_SEARCH + 1
        st.rerun()


def step_search_done() -> None:
    """Completion screen shown after the search fallback (step 12)."""
    st.markdown('<p class="done-title">Profile updated.</p>', unsafe_allow_html=True)
    html(
        "done-sub",
        "Your song selections have been saved. Ready to see your personalised playlist?",
    )
    st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
    col_restart, col_playlist = st.columns(2)
    with col_restart:
        if st.button("Start over", key="btn_restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with col_playlist:
        if st.button("See my playlist →", key="btn_to_playlist"):
            st.session_state.step = STEP_PLAYLIST
            st.rerun()


# ── playlist pipeline ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_genre_map_for_history() -> Dict[str, str]:
    """song_id → genre lookup from songs_full.csv.  Cached once per session."""
    genre_map: Dict[str, str] = {}
    if not SONGS_CSV.exists():
        return genre_map
    with open(SONGS_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            sid = row.get("song_id", "").strip()
            if sid:
                genre_map[sid] = row.get("genre", "unknown").strip()
    return genre_map


def _build_playlist_history_summary(user_id: str) -> str:
    """
    One-sentence description of liked/skipped songs for the Gemini prompt.
    Example: "Liked 8 songs. Skipped 2. Mostly chill, ambient, jazz."
    """
    interactions_path = DATA_DIR / "interactions.csv"
    if not interactions_path.exists():
        return "No listening history yet."

    genre_map     = _load_genre_map_for_history()
    liked_count   = 0
    skipped_count = 0
    liked_genres: Dict[str, int] = {}

    with open(interactions_path, encoding="utf-8") as f:
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
        top = sorted(liked_genres, key=lambda g: -liked_genres[g])[:3]
        parts.append(f"Mostly {', '.join(top)}.")
    return " ".join(parts)


def _run_playlist_pipeline(profile: Dict, user_intent: str, api_key: str) -> tuple:
    """
    Full recommendation pipeline for one playlist refresh.

    With intent → rag_layer retrieves songs from the 81k catalog by semantic
    search, ML-ranks them by the user's taste, then Gemini explains the picks.

    Without intent → profile-based retrieval + ML ranking (no explanations).

    Imports are deferred so pipeline modules only load when the user reaches
    step 20 — not on every earlier survey page render.

    Returns (songs: List[Dict], None) on success, (None, error_str) on failure.
    """
    from rag.rag_layer import recommend as rag_recommend

    history = _build_playlist_history_summary(profile["user_id"])
    results = rag_recommend(user_intent, profile, history, api_key)

    if not results:
        return None, (
            "No songs found. If you typed a request, the knowledge base may not be "
            "built yet — run: python src/rag/build_knowledge_base.py"
        )
    return results, None


def _render_playlist_card(song: Dict, rank_num: int, user_id: str) -> None:
    """
    Render one playlist song card with Like/Skip buttons.

    Songs from rag_layer.recommend() have keys: song_id, title, artist, score, explanation.
    Like/Skip state lives in st.session_state.pl_liked / pl_skipped so button clicks
    don't re-run the pipeline — only the card's visual state updates.
    """
    sid         = song.get("song_id", "")
    title       = song.get("title", "Unknown")
    artist      = song.get("artist", "Unknown")
    explanation = song.get("explanation", "")

    already_liked   = sid in st.session_state.pl_liked
    already_skipped = sid in st.session_state.pl_skipped

    col_info, col_actions = st.columns([8, 2])

    with col_info:
        card_html = (
            f'<div class="song-card-feedback">'
            f'  <div style="display:flex;gap:0.7rem;align-items:flex-start;">'
            f'    <span class="song-num">{rank_num}</span>'
            f'    <div>'
            f'      <span class="song-title">{title}</span>'
            f'      <span class="song-artist">{artist}</span>'
            f'    </div>'
            f'  </div>'
        )
        if explanation:
            card_html += (
                f'<div style="font-size:0.81rem;color:#6A6A6A;margin:0.4rem 0 0 1.5rem;'
                f'line-height:1.5;font-style:italic;">{explanation}</div>'
            )
        if already_liked:
            card_html += (
                '<div style="font-size:0.75rem;color:#1DB954;font-weight:600;'
                'margin:0.25rem 0 0 1.5rem;">Liked</div>'
            )
        elif already_skipped:
            card_html += (
                '<div style="font-size:0.75rem;color:#535353;'
                'margin:0.25rem 0 0 1.5rem;">Skipped</div>'
            )
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

    with col_actions:
        if not already_liked and not already_skipped:
            sub_like, sub_skip = st.columns([1, 1])
            with sub_like:
                if st.button("Like", key=f"pl_like_{sid}"):
                    save_interactions([song], user_id, label=1)
                    st.session_state.pl_liked.add(sid)
                    st.rerun()
            with sub_skip:
                if st.button("Skip", key=f"pl_skip_{sid}"):
                    save_interactions([song], user_id, label=0)
                    st.session_state.pl_skipped.add(sid)
                    st.rerun()


def step_playlist() -> None:
    """
    Playlist view (step 20) — shown immediately after survey completion.

    Runs the full three-stage inference pipeline on first entry.  On subsequent
    visits the result is served from session_state so Like/Skip button clicks
    don't re-trigger the expensive Gemini call.

    The user can type a free-text intent and click "Refresh Playlist" to get
    new picks filtered by mood/context (e.g. "studying, calm, no lyrics").
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    # Prefer profile from session state (just finished survey); fall back to disk
    # so this step also works when the user navigates here on a returning visit.
    profile = st.session_state.get("saved_profile")
    if profile is None:
        profile_path = DATA_DIR / "current_user.json"
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                profile = json.load(f)
    if profile is None:
        html("count-hint", "No profile found. Please complete the survey first.")
        return

    if not api_key:
        st.error(
            "GROQ_API_KEY not found. "
            "Add GROQ_API_KEY=your_key_here to your .env file and restart."
        )
        return

    if "pl_liked"   not in st.session_state: st.session_state.pl_liked   = set()
    if "pl_skipped" not in st.session_state: st.session_state.pl_skipped = set()

    # ── header ────────────────────────────────────────────────────────────
    name = profile.get("name", "")
    if name:
        html("q-counter", f"Welcome, {name}")
    st.markdown('<p class="done-title">Your Playlist</p>', unsafe_allow_html=True)
    html("done-sub", "Refreshed just for you. Like or skip songs to improve future picks.")

    # ── intent input + refresh ─────────────────────────────────────────────
    st.markdown('<div style="height:0.75rem;"></div>', unsafe_allow_html=True)
    user_intent = st.text_input(
        "pl_intent_label",
        value=st.session_state.get("pl_intent", ""),
        placeholder="What are you in the mood for? (e.g. studying, calm, no lyrics)",
        label_visibility="collapsed",
        key="pl_intent_input",
    )
    refresh_clicked = st.button("Refresh Playlist", key="btn_pl_refresh")

    # Run pipeline on first visit (no pl_songs yet) OR on explicit refresh.
    # On error: set pl_songs = [] so a subsequent Like/Skip click doesn't auto-retry.
    needs_run = refresh_clicked or "pl_songs" not in st.session_state
    if needs_run:
        with st.spinner("Finding your music …"):
            songs, error = _run_playlist_pipeline(profile, user_intent.strip(), api_key)
        if error:
            st.error(error)
            if "pl_songs" not in st.session_state:
                st.session_state.pl_songs = []
        else:
            st.session_state.pl_songs   = songs
            st.session_state.pl_intent  = user_intent
            st.session_state.pl_liked   = set()
            st.session_state.pl_skipped = set()
            st.rerun()

    # ── song cards ────────────────────────────────────────────────────────
    songs = st.session_state.get("pl_songs", [])
    if songs:
        last_intent = st.session_state.get("pl_intent", "")
        heading = "Picks for you" + (f' — "{last_intent}"' if last_intent else "")
        html("songs-heading", heading)
        for i, song in enumerate(songs, 1):
            _render_playlist_card(song, i, profile["user_id"])

    # ── footer ────────────────────────────────────────────────────────────
    st.markdown('<div style="height:3rem;"></div>', unsafe_allow_html=True)
    col_restart, _ = st.columns(2)
    with col_restart:
        if st.button("Start over", key="btn_pl_restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Music Profile",
        layout="centered",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if "step" not in st.session_state:
        st.session_state.step = 0
    if "answers" not in st.session_state:
        st.session_state.answers = {}

    all_artists          = load_top_artists(limit=500)
    all_songs_for_search = load_all_songs_for_search()
    step = st.session_state.step

    if   step == 0:                    step_0_landing()
    elif step == 1:                    step_1_name()
    elif step == 2:                    step_2_favorite_artists(all_artists)
    elif step == 3:                    step_3_recent_artists(all_artists)
    elif step == 4:                    step_4_genres()
    elif step == 5:                    step_5_moods()
    elif step == 6:                    step_6_purpose()
    elif step == 7:                    step_7_exploration()
    elif step == 8:                    step_8_popularity()
    elif step == 9:                    step_confirm_profile()
    elif step == STEP_SONG_SEARCH:     step_song_search(all_songs_for_search)
    elif step == STEP_SONG_SEARCH + 1: step_search_done()
    elif step == STEP_PLAYLIST:        step_playlist()
    else:                              step_song_feedback()


if __name__ == "__main__":
    main()
