import os
import glob
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from recommender_engine import (
    W_G,
    build_multimodal_space,
    ContextualPriorSeeder,
    LapsedUserPriorSeeder,
    split_full_vector_to_subspaces,
    explain_recommendation,
    explanation_to_text,
    compare_anime,
    update_user_vector_rocchio,
    generate_diverse_deck,
)
from card_component import render_tinder_draggable_card
from storage_manager import (
    save_profile_to_browser,
    clear_browser_profile,
    try_restore_profile_from_browser,
)

from data_loader import (
    get_fallback_dataframe,
    normalize_dataframe,
    get_main_studio,
    fetch_jikan_catalog,
    load_hybrid_catalog,
)

# ==============================================================================
# CẤU HÌNH TRANG WEB (PAGE CONFIGURATION)
# ==============================================================================
st.set_page_config(
    page_title="AniMatch — Interactive Anime Recommender",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
div.stButton > button p {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHE NẠP DỮ LIỆU & TỪ ĐIỂN TRA CỨU O(1)
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner="Building Hybrid Catalog (Live API + Local CSV)...")
def load_and_prepare_catalog(csv_path: str | None, api_pages: int = 2):
    df = load_hybrid_catalog(csv_path=csv_path, api_pages=api_pages)
    df = df[df["type"].astype(str).str.upper().isin(["TV", "MOVIE"])].reset_index(drop=True)

    all_genres = sorted(list({
        g.strip()
        for val in (df["genres"] + "|" + df["themes"]).tolist()
        for g in re.split(r"[|,;]", str(val))
        if g.strip()
    }))
    anime_lookup = {int(row["anime_id"]): row for _, row in df.iterrows()}
    return df, all_genres, anime_lookup

# ==============================================================================
# CỬA SỔ KHỞI TẠO HỒ SƠ (ONBOARDING PERSONA MODAL DIALOG)
# ==============================================================================
@st.dialog("🧭 Onboarding Persona Setup — AniMatch", width="large")
def open_onboarding_modal(
    df, V_i, id_to_idx, anime_vectors, vectorizers, dims, all_genres,
    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
):
    st.session_state.show_onboarding_modal = False

    st.caption("Choose how you want to initialize your preference profile. Once confirmed, this menu closes and takes you directly to the swipe deck.")

    persona_choice = st.radio(
        "Select Your User Persona:",
        [
            "👀 Eyecandy First (Zero-shot visual discovery — jump straight into swiping)",
            "🏷️ Content First (Cold-start — pick your favorite genres & themes)",
            "🕰️ Lapsed User (Returning user — blend past watch history with new genres)"
        ]
    )

    seeder = LapsedUserPriorSeeder(
        anime_vectors=anime_vectors,
        genre_vectorizer=vectorizers["genres"],
        w_g=W_G,
        dim_sypnosis=dims["synopsis"],
        dim_studio=dims["studio"]
    )

    if persona_choice.startswith("👀 Eyecandy First"):
        st.info("No upfront survey needed! We will deal a diverse deck of high-rated anime so your persona evolves naturally with every swipe.")
        if st.button("🔥 Start Swiping Now", type="primary", use_container_width=True):
            v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
            st.session_state.persona_mode = "Eyecandy First"
            st.session_state.seed_params = {}
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = []
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, [], 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, id_to_idx=id_to_idx
            )
            st.session_state.onboarded = True
            st.session_state.pending_ls_save = True
            st.rerun()

    elif persona_choice.startswith("🏷️ Content First"):
        selected_genres = st.multiselect(
            "Select your preferred Genres & Themes ($G$):",
            options=all_genres,
            default=all_genres[:3] if len(all_genres) >= 3 else all_genres
        )
        if st.button("🚀 Initialize & Start Swiping", type="primary", use_container_width=True):
            v_user_0, excluded_ids = seeder.seed_cold_start("content_first", selected_genres)
            st.session_state.persona_mode = "Content First"
            st.session_state.seed_params = {"selected_genres": selected_genres}
            st.session_state.v_user_initial = v_user_0.copy()
            st.session_state.swipe_history = []
            st.session_state.watched_ids = []
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = []
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, [], 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, id_to_idx=id_to_idx
            )
            st.session_state.onboarded = True
            st.session_state.pending_ls_save = True
            st.rerun()

    else:
        col1, col2 = st.columns(2)
        with col1:
            anime_options = {f"{row['title']}": int(row["anime_id"]) for _, row in df.iterrows()}
            picked_past = st.multiselect(
                "Anime you previously watched ($V_{past}$):",
                options=list(anime_options.keys()),
                default=list(anime_options.keys())[:2]
            )
            lambda_1 = st.slider("Past Taste Weight (λ₁)", 0.1, 1.0, 0.4, 0.1)
        with col2:
            picked_new = st.multiselect(
                "New Genres you want to explore ($V_{new}$):",
                options=all_genres,
                default=all_genres[-2:] if len(all_genres) >= 2 else all_genres
            )
            lambda_2 = st.slider("New Taste Weight (λ₂)", 0.1, 1.0, 0.6, 0.1)

        if st.button("🚀 Initialize Lapsed Profile & Start Swiping", type="primary", use_container_width=True):
            past_ids = [anime_options[lbl] for lbl in picked_past]
            v_user_0, excluded_ids = seeder.seed_lapsed_user(past_ids, picked_new, lambda_1, lambda_2)
            st.session_state.persona_mode = "Lapsed User"
            st.session_state.seed_params = {
                "past_ids": past_ids,
                "new_genres": picked_new,
                "lambda_1": lambda_1,
                "lambda_2": lambda_2,
            }
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = [str(df[df["anime_id"] == aid].iloc[0]["title"]) for aid in past_ids]
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, st.session_state.swiped_titles, 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, id_to_idx=id_to_idx
            )
            st.session_state.onboarded = True
            st.session_state.pending_ls_save = True
            st.rerun()


# ==============================================================================
# QUẢN LÝ TRẠNG THÁI PHIÊN & XỬ LÝ HÀNH ĐỘNG QUẸT
# ==============================================================================
try:
    from streamlit_js_eval import get_local_storage, set_local_storage
    HAS_JS_EVAL = True
except ImportError:
    HAS_JS_EVAL = False

STORAGE_KEY = "animatch_user_profile_v1"


def save_profile_to_browser():
    if not HAS_JS_EVAL:
        return
    payload = {
        "persona_mode": st.session_state.get("persona_mode", "Not Selected"),
        "liked_ids": [int(x) for x in st.session_state.get("liked_ids", [])],
        "passed_ids": [int(x) for x in st.session_state.get("passed_ids", [])],
        "watched_ids": [int(x) for x in st.session_state.get("watched_ids", [])],
        "swipe_history": [[int(aid), str(act)] for aid, act in st.session_state.get("swipe_history", [])],
        "swipe_count": int(st.session_state.get("swipe_count", 0)),
    }
    set_local_storage(STORAGE_KEY, json.dumps(payload))


def load_profile_from_browser(
    df: pd.DataFrame,
    anime_vectors: dict,
    expected_dim: int,
    update_fn,
    gamma: float = 0.95,
    alpha: float = 0.30,
    beta: float = 0.15
) -> bool:
    if not HAS_JS_EVAL:
        return False

    raw_data = get_local_storage(STORAGE_KEY)
    if not raw_data:
        return False

    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        liked_ids = [int(x) for x in data.get("liked_ids", [])]
        passed_ids = [int(x) for x in data.get("passed_ids", [])]
        watched_ids = [int(x) for x in data.get("watched_ids", [])]
        saved_history = data.get("swipe_history", [])

        if not liked_ids and not passed_ids and not watched_ids:
            return False

        valid_ids = set(df["anime_id"].astype(int).tolist())
        id_to_title = dict(zip(df["anime_id"].astype(int), df["title"].astype(str)))
        id_to_eps = dict(zip(df["anime_id"].astype(int), df["episodes"]))

        v_initial = np.zeros(expected_dim)
        v_rebuilt = v_initial.copy()
        excluded = set()
        swiped_titles = []
        rebuilt_history = []

        if saved_history:
            pos_so_far = 0
            for item in saved_history:
                aid, act = int(item[0]), str(item[1])
                if aid in valid_ids and aid in anime_vectors:
                    length_factor = _get_length_penalty_factor({"episodes": id_to_eps.get(aid, 12)}, act)
                    base_beta = (beta * 0.4) if pos_so_far == 0 else beta
                    eff_beta = base_beta * length_factor
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], act, gamma, alpha, eff_beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, act))
                    if act in ("like", "watched"):
                        pos_so_far += 1
        else:
            for aid in passed_ids:
                if aid in valid_ids and aid in anime_vectors:
                    eff_beta = (beta * 0.4) if (not liked_ids and not watched_ids) else beta
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "pass", gamma, alpha, eff_beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, "pass"))
            for aid in watched_ids:
                if aid in valid_ids and aid in anime_vectors:
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "watched", gamma, alpha, beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, "watched"))
            for aid in liked_ids:
                if aid in valid_ids and aid in anime_vectors:
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "like", gamma, alpha, beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, "like"))

        st.session_state.persona_mode = data.get("persona_mode", "Restored Session")
        st.session_state.v_user_initial = v_initial
        st.session_state.v_user = v_rebuilt
        st.session_state.swipe_history = rebuilt_history
        st.session_state.liked_ids = [aid for aid in liked_ids if aid in valid_ids]
        st.session_state.passed_ids = [aid for aid in passed_ids if aid in valid_ids]
        st.session_state.watched_ids = [aid for aid in watched_ids if aid in valid_ids]
        st.session_state.excluded_ids = excluded
        st.session_state.swiped_titles = swiped_titles
        st.session_state.swipe_count = len(rebuilt_history)
        st.session_state.onboarded = True
        st.session_state.show_onboarding_modal = False
        return True
    except Exception:
        return False


def clear_browser_profile():
    if not HAS_JS_EVAL:
        return
    set_local_storage(STORAGE_KEY, "")


def init_session_state():
    defaults = {
        "onboarded": False,
        "show_onboarding_modal": True,
        "persona_mode": "Not Selected",
        "v_user_initial": None,
        "v_user": None,
        "swipe_history": [],
        "excluded_ids": set(),
        "cooldown_ids": {},       # {anime_id: swipes_remaining} -> khóa khỏi deck hiện tại sau khi Revert
        "swiped_titles": [],
        "deck": [],
        "liked_ids": [],
        "passed_ids": [],
        "watched_ids": [],
        "swipe_count": 0,
        "last_action_msg": "",
        "dataset_signature": "",
        "sidebar_params": None,
        "pending_ls_clear": False,
        "pending_ls_save": False,
        "ls_checked": False,
        "consecutive_passes": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_effective_excluded_ids() -> set:
    """Gộp excluded_ids vĩnh viễn và các ID đang trong thời gian chờ (cooldown) sau khi Revert."""
    base_excl = set(st.session_state.get("excluded_ids", set()))
    cd = st.session_state.get("cooldown_ids", {})
    return base_excl | set(cd.keys())

def _get_length_excluded_ids(df: pd.DataFrame, length_filter: str) -> set:
    if not length_filter or length_filter == "All Lengths":
        return set()

    eps_str = df["episodes"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    eps = pd.to_numeric(eps_str, errors="coerce").fillna(12.0)

    if length_filter.startswith("Movie"):
        return set(df.loc[eps != 1, "anime_id"].astype(int))
    if length_filter.startswith("Short"):
        return set(df.loc[(eps < 2) | (eps > 13), "anime_id"].astype(int))
    if length_filter.startswith("Medium"):
        return set(df.loc[(eps <= 13) | (eps > 26), "anime_id"].astype(int))
    if length_filter.startswith("Long"):
        return set(df.loc[(eps <= 26) | (eps > 52), "anime_id"].astype(int))
    if length_filter.startswith("Endless"):
        return set(df.loc[eps <= 52, "anime_id"].astype(int))
    return set()


def _get_all_excluded_ids(df: pd.DataFrame, length_filter: str) -> set:
    return _get_effective_excluded_ids() | _get_length_excluded_ids(df, length_filter)


def _tick_cooldowns():
    """Giảm 1 nhịp đếm ngược cooldown sau mỗi lần người dùng quẹt sang thẻ mới."""
    cd = st.session_state.get("cooldown_ids", {})
    if not cd:
        return
    updated = {aid: rem - 1 for aid, rem in cd.items() if (rem - 1) > 0}
    st.session_state.cooldown_ids = updated


def _rebuild_user_vector_from_history(
    anime_lookup: dict, anime_vectors: dict, gamma: float, alpha: float, beta: float,
    length_filter: str = "All Lengths"
):
    v_rebuilt = (
        st.session_state.v_user_initial.copy()
        if st.session_state.get("v_user_initial") is not None
        else np.zeros_like(st.session_state.v_user)
    )
    pos_so_far = 0
    for aid, act in st.session_state.get("swipe_history", []):
        length_factor = (
            _get_length_penalty_factor(anime_lookup[aid], act, length_filter)
            if aid in anime_lookup else 1.0
        )
        base_beta = (beta * 0.4) if pos_so_far == 0 else beta
        eff_beta = base_beta * length_factor

        v_rebuilt = update_user_vector_rocchio(
            v_rebuilt, anime_vectors[aid], action=act, gamma=gamma, alpha=alpha, beta=eff_beta
        )
        if act in ("like", "watched"):
            pos_so_far += 1
    st.session_state.v_user = v_rebuilt

def _get_length_penalty_factor(anime_row, action: str, length_filter: str = "All Lengths") -> float:
    if action != "pass" or length_filter.startswith("Endless"):
        return 1.0
    try:
        ep_count = float(anime_row.get("episodes", 12))
        if np.isnan(ep_count) or ep_count <= 0:
            ep_count = 12.0
    except (ValueError, TypeError):
        ep_count = 12.0

    if ep_count > 52:
        return 0.20
    elif ep_count > 26:
        return 0.50
    return 1.00
    
def process_swipe(
    action: str,
    df: pd.DataFrame,
    V_i,
    id_to_idx: dict,
    anime_lookup: dict,
    anime_vectors: dict,
    gamma: float,
    alpha: float,
    beta: float,
    mmr_lambda: float,
    max_per_studio: int,
    franchise_thresh: int,
    min_score: float = 7.2,
    quality_weight: float = 0.25,
    length_filter: str = "All Lengths"
):
    if not st.session_state.deck:
        return

    current_aid = st.session_state.deck.pop(0)
    anime_row = anime_lookup[current_aid]
    v_anime = anime_vectors[current_aid]

    # Giảm bộ đếm cooldown cho các bộ đã revert trước đó
    _tick_cooldowns()

    # Tính hệ số giảm phạt nếu user Pass phim quá dài tập
    length_factor = _get_length_penalty_factor(anime_row, action, length_filter)
    has_positive = (len(st.session_state.liked_ids) + len(st.session_state.watched_ids)) > 0
    base_beta = (beta * 0.4) if not has_positive else beta
    effective_beta = base_beta * length_factor

    st.session_state.v_user = update_user_vector_rocchio(
        st.session_state.v_user, v_anime, action=action, gamma=gamma, alpha=alpha, beta=effective_beta
    )

    st.session_state.excluded_ids.add(current_aid)
    st.session_state.swiped_titles.append(str(anime_row["title"]))
    st.session_state.swipe_history.append((current_aid, action))
    st.session_state.swipe_count += 1

    aid = int(current_aid)  # Khớp với biến ID của lá bài trên cùng trong process_swipe
    title = anime_lookup[aid]["title"] if aid in anime_lookup else f"ID {aid}"

    if action == "watched":
        if aid not in st.session_state.watched_ids:
            st.session_state.watched_ids.append(aid)
        st.session_state.excluded_ids.add(aid)
        st.session_state.swipe_history.append({
            "anime_id": aid, "title": title, "action": "watched", "taste_boosted": False
        })
        st.session_state.last_action_msg = f"👁️ Marked as Watched (Neutral taste — sequels unlocked): **{title}**"

    if action == "like":
        st.session_state.liked_ids.append(current_aid)
        st.session_state.last_action_msg = f"💚 LIKED: **{anime_row['title']}**"
    elif action == "watched":
        if aid not in st.session_state.watched_ids:
            st.session_state.watched_ids.append(aid)
        st.session_state.excluded_ids.add(aid)
        # Trọng số = 0 (Neutral): Chỉ ẩn thẻ và mở khóa Sequel, KHÔNG làm lệch vector gu v_user
        st.session_state.swipe_history.append({
            "anime_id": aid, "title": title, "action": "watched", "taste_boosted": False
        })
        st.session_state.last_action_msg = f"👁️ Marked as Watched (Neutral taste — sequels unlocked): **{title}**"
    else:
        st.session_state.passed_ids.append(current_aid)
        st.session_state.last_action_msg = f"❌ PASSED: **{anime_row['title']}**"

    if action == "pass":
        st.session_state.consecutive_passes = st.session_state.get("consecutive_passes", 0) + 1
    elif action in ("like", "watched"):
        st.session_state.consecutive_passes = 0

    streak = st.session_state.get("consecutive_passes", 0)
    effective_mmr = mmr_lambda

    # Tự động phát hiện đi sai hướng khi Pass liên tiếp
    if streak >= 5:
        # Kéo vector người dùng thoát khỏi vùng kẹt, pha trộn lại 60% vector gốc + 40% vector hiện tại
        if st.session_state.get("v_user_initial") is not None:
            st.session_state.v_user = (
                0.60 * st.session_state.v_user_initial + 0.40 * st.session_state.v_user
            )
        effective_mmr = 0.30  # Ép đa dạng hóa tối đa ở deck mới
        st.session_state.consecutive_passes = 0
        st.session_state.last_action_msg = "🧭 5 passes in a row — Auto-pivoting to fresh genres & hidden gems!"
    elif streak >= 3:
        # Giảm mmr_lambda để bắt đầu dò sang các cụm thể loại khác
        effective_mmr = min(mmr_lambda, 0.40)
        st.session_state.last_action_msg = "🔀 Exploring wider genres to find your vibe..."
    

    st.session_state.deck = generate_diverse_deck(
        df=df,
        anime_vectors=anime_vectors,
        v_user=st.session_state.v_user,
        excluded_ids=_get_all_excluded_ids(df, length_filter),
        swiped_titles=st.session_state.swiped_titles,
        deck_size=8,
        mmr_lambda=mmr_lambda,
        max_per_studio=max_per_studio,
        franchise_threshold=franchise_thresh,
        min_score=min_score,
        quality_weight=quality_weight,
        V_i=V_i,
        id_to_idx=id_to_idx
    )


def revert_swiped_anime(
    target_aid: int,
    source_list: str,
    df: pd.DataFrame,
    V_i,
    id_to_idx: dict,
    anime_lookup: dict,
    anime_vectors: dict,
    gamma: float,
    alpha: float,
    beta: float,
    mmr_lambda: float,
    max_per_studio: int,
    franchise_thresh: int,
    min_score: float,
    quality_weight: float,
    length_filter: str = "All Lengths"
):
    """
    Hoàn tác một bộ anime từ Liked, Passed hoặc Watched:
    - Xóa khỏi lịch sử & tính lại vector v_user công bằng.
    - KHÔNG nhét lên đầu deck hiện tại; đưa vào cooldown (8 lượt quẹt) để sau đó
      tự tái xuất hiện ngẫu nhiên/cạnh tranh sòng phẳng bằng MMR.
    """
    if source_list == "liked":
        st.session_state.liked_ids = [aid for aid in st.session_state.liked_ids if aid != target_aid]
    elif source_list == "watched":
        st.session_state.watched_ids = [aid for aid in st.session_state.watched_ids if aid != target_aid]
    else:
        st.session_state.passed_ids = [aid for aid in st.session_state.passed_ids if aid != target_aid]

    st.session_state.excluded_ids.discard(target_aid)

    target_title = str(anime_lookup[target_aid]["title"])
    if target_title in st.session_state.swiped_titles:
        st.session_state.swiped_titles.remove(target_title)

    st.session_state.swipe_history = [
        (aid, act) for (aid, act) in st.session_state.swipe_history if aid != target_aid
    ]
    st.session_state.swipe_count = len(st.session_state.swipe_history)

    # Tái tạo lại vector người dùng chính xác
    _rebuild_user_vector_from_history(anime_lookup, anime_vectors, gamma, alpha, beta, length_filter)

    # Đặt cooldown = 8 lượt quẹt để bộ này chắc chắn KHÔNG xuất hiện trong deck 8 lá hiện tại,
    # nhưng sẽ tự động quay trở lại kho ứng viên và xuất hiện lại vào một lúc nào đó sau này.
    cd = st.session_state.get("cooldown_ids", {})
    cd[target_aid] = 8
    st.session_state.cooldown_ids = cd

    # Giữ nguyên thẻ đang hiển thị trên màn hình (deck[0]) nếu còn, chỉ tính lại phần đuôi phía sau
    active_aid = st.session_state.deck[0] if st.session_state.deck else None
    refreshed_tail = generate_diverse_deck(
        df=df,
        anime_vectors=anime_vectors,
        v_user=st.session_state.v_user,
        excluded_ids=_get_all_excluded_ids(df, length_filter) | ({active_aid} if active_aid is not None else set()),
        swiped_titles=st.session_state.swiped_titles,
        deck_size=7 if active_aid is not None else 8,
        mmr_lambda=mmr_lambda,
        max_per_studio=max_per_studio,
        franchise_threshold=franchise_thresh,
        min_score=min_score,
        quality_weight=quality_weight,
        V_i=V_i,
        id_to_idx=id_to_idx
    )
    st.session_state.deck = ([active_aid] + refreshed_tail) if active_aid is not None else refreshed_tail
    st.session_state.last_action_msg = (
        f"↩️ REVERTED **{target_title}**: Weights undone! It will reappear naturally later (not in the current deck)."
    )

def mark_liked_as_watched(
    aid: int,
    df: pd.DataFrame,
    V_i,
    id_to_idx: dict,
    anime_lookup: dict,
    anime_vectors: dict,
    mmr_lambda: float,
    max_per_studio: int,
    franchise_thresh: int,
    min_score: float,
    quality_weight: float,
    length_filter: str
):
    """
    Chuyển một anime từ danh sách Liked Watchlist sang Already Watched:
    - Giữ nguyên đóng góp của phim vào vector sở thích v_user (không trừ điểm).
    - Mở khóa các phần tiếp theo (Sequel / Movie tổng kết) của franchise này trong deck.
    """
    aid = int(aid)
    if aid in st.session_state.liked_ids:
        st.session_state.liked_ids.remove(aid)
    if aid not in st.session_state.watched_ids:
        st.session_state.watched_ids.append(aid)

    st.session_state.excluded_ids.add(aid)
    if aid in st.session_state.cooldown_ids:
        del st.session_state.cooldown_ids[aid]

    title = anime_lookup[aid]["title"] if aid in anime_lookup else f"ID {aid}"
    if title not in st.session_state.swiped_titles:
        st.session_state.swiped_titles.append(title)

    # Cập nhật lịch sử swipe nếu có bản ghi 'like' của phim này
    for entry in reversed(st.session_state.swipe_history):
        if int(entry.get("anime_id", -1)) == aid and entry.get("action") == "like":
            entry["action"] = "watched"
            break

    # Sinh lại deck mới để các phần hậu truyện của phim vừa xem xong được mở khóa hợp lệ
    st.session_state.deck = generate_diverse_deck(
        df, anime_vectors, st.session_state.v_user,
        _get_all_excluded_ids(df, length_filter), st.session_state.swiped_titles, 8,
        mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
        V_i=V_i, id_to_idx=id_to_idx,
        watched_ids=st.session_state.watched_ids
    )
    st.session_state.last_action_msg = f"👁️ Marked **{title}** as Watched — sequels unlocked!"

def apply_watched_feedback(
    aid: int,
    sentiment: str,  # "boost" hoặc "dislike"
    df: pd.DataFrame,
    V_i,
    id_to_idx: dict,
    anime_lookup: dict,
    anime_vectors: dict,
    gamma: float,
    alpha: float,
    beta: float,
    mmr_lambda: float,
    max_per_studio: int,
    franchise_thresh: int,
    min_score: float,
    quality_weight: float,
    length_filter: str
):
    """
    Điều chỉnh vector sở thích v_user dựa trên cảm nhận của user với một bộ phim đã xem:
    - 'boost': Cộng +alpha vào v_user để tìm thêm phim có vibe tương tự.
    - 'dislike': Trừ -beta khỏi v_user và gỡ khỏi danh sách mở khóa Sequel.
    """
    aid = int(aid)
    if aid not in anime_vectors:
        return
    v_item = anime_vectors[aid]
    title = anime_lookup[aid]["title"] if aid in anime_lookup else f"ID {aid}"

    if sentiment == "boost":
        # Cập nhật vector theo hướng Like (+alpha) và chuẩn hóa L2
        v_new = (gamma * st.session_state.v_user) + (alpha * v_item)
        norm = np.linalg.norm(v_new)
        st.session_state.v_user = (v_new / norm) if norm > 1e-9 else v_new

        for entry in reversed(st.session_state.swipe_history):
            if isinstance(entry, dict) and int(entry.get("anime_id", -1)) == aid:
                entry["taste_boosted"] = True
                break
        st.session_state.last_action_msg = f"❤️ Boosted taste profile with **{title}**!"

    elif sentiment == "dislike":
        # Cập nhật vector theo hướng Pass (-beta) và chuẩn hóa L2
        v_new = (gamma * st.session_state.v_user) - (beta * v_item)
        norm = np.linalg.norm(v_new)
        st.session_state.v_user = (v_new / norm) if norm > 1e-9 else v_new

        if aid in st.session_state.watched_ids:
            st.session_state.watched_ids.remove(aid)
        if aid not in st.session_state.passed_ids:
            st.session_state.passed_ids.append(aid)
        st.session_state.excluded_ids.add(aid)
        st.session_state.last_action_msg = f"👎 Downvoted **{title}** — won't recommend similar titles or sequels."

    st.session_state.deck = generate_diverse_deck(
        df, anime_vectors, st.session_state.v_user,
        _get_all_excluded_ids(df, length_filter), st.session_state.swiped_titles, 8,
        mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
        V_i=V_i, id_to_idx=id_to_idx,
        watched_ids=st.session_state.watched_ids
    )


def main():
    init_session_state()

    if st.session_state.get("pending_ls_clear", False):
        clear_browser_profile()
        st.session_state.pending_ls_clear = False

    if st.session_state.get("pending_ls_save", False):
        save_profile_to_browser()
        st.session_state.pending_ls_save = False

    top_c1, top_c2 = st.columns([3.5, 1.5])
    with top_c1:
        st.markdown("## 🔥 AniMatch")
    with top_c2:
        if st.button("🧭 Change Persona / Reset", use_container_width=True):
            st.session_state.pending_ls_clear = True
            st.session_state.show_onboarding_modal = True
            st.rerun()

    with st.sidebar:
        st.subheader("🌐 Jikan Live Catalog")
        pages_per_type = st.slider("Pages per Type (25 anime/page)", 2, 16, 8, 2)
        if st.button("🔄 Refresh Jikan Catalog", use_container_width=True):
            st.cache_data.clear()
            fetch_jikan_catalog(pages_per_type=pages_per_type, force_refresh=True)
            st.rerun()

        st.divider()
        st.header("⚙️ Algorithm Parameters")
        gamma = st.slider("Memory Decay (γ)", 0.90, 0.98, 0.95, 0.01)
        alpha = st.slider("Like Attractor (α)", 0.05, 1.00, 0.30, 0.05)
        beta = st.slider("Pass Repeller (β)", 0.05, 1.00, 0.15, 0.05)
        mmr_lambda = st.slider("MMR Balance (λ)", 0.50, 1.00, 0.75, 0.05)
        max_per_studio = st.number_input("Studio Cap (Max per Studio)", min_value=1, max_value=5, value=2)
        franchise_thresh = st.slider("Franchise Deduplication Threshold", 75, 100, 90, 1)
        st.divider()
        st.subheader("⭐ Quality & Likeability Filter")
        min_score = st.slider("Minimum Rating Score", 5.0, 8.5, 7.2, 0.1)
        quality_weight = st.slider("Quality Boost Weight", 0.0, 0.6, 0.25, 0.05)
        st.divider()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_csv_files = sorted([
        p for p in glob.glob(os.path.join(current_dir, "*.csv"))
        if "jikan_tv_movie_cache" not in os.path.basename(p)
    ])
    main_csv_path = local_csv_files[0] if local_csv_files else None

    df, all_genres, anime_lookup = load_and_prepare_catalog(csv_path=main_csv_path, api_pages=2)
    sig = f"hybrid_{os.path.basename(main_csv_path) if main_csv_path else 'api'}_{len(df)}"
    data_source_label = f"Hybrid (Live API + {os.path.basename(main_csv_path)})" if main_csv_path else "Live API"
    sig = f"jikan_tv_movie_{pages_per_type}_{len(df)}"
    data_source_label = f"Jikan API v4 (TV & Movie • {len(df)} titles)"

    if st.session_state.get("dataset_signature", "") != sig:
        st.session_state.dataset_signature = sig
        st.session_state.onboarded = False
        st.session_state.v_user_initial = None
        st.session_state.v_user = None
        st.session_state.swipe_history = []
        st.session_state.excluded_ids = set()
        st.session_state.cooldown_ids = {}
        st.session_state.swiped_titles = []
        st.session_state.deck = []
        st.session_state.liked_ids = []
        st.session_state.passed_ids = []
        st.session_state.watched_ids = []
        st.session_state.swipe_count = 0
        st.session_state.last_action_msg = ""
        st.session_state.show_onboarding_modal = True
        st.session_state.ls_checked = False

    space_res = build_multimodal_space(df)
    if len(space_res) == 5:
        V_i, id_to_idx, anime_vectors, vectorizers, dims = space_res
    else:
        V_i, anime_vectors, vectorizers, dims = space_res
        id_to_idx = {int(row["anime_id"]): idx for idx, row in df.iterrows()}

    expected_dim = V_i.shape[1]

    if not st.session_state.get("ls_checked", False):
        st.session_state.ls_checked = True
        restored = load_profile_from_browser(
            df=df,
            anime_vectors=anime_vectors,
            expected_dim=expected_dim,
            update_fn=update_user_vector_rocchio,
            gamma=gamma,
            alpha=alpha,
            beta=beta
        )
        if restored:
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, st.session_state.v_user,
                _get_effective_excluded_ids(), st.session_state.swiped_titles, 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, id_to_idx=id_to_idx
            )
            st.toast("🔄 Restored your previous swipe session from browser storage!")

    if st.session_state.get("v_user") is None or st.session_state.v_user.shape[0] != expected_dim:
        seeder = ContextualPriorSeeder(anime_vectors, vectorizers["genres"], W_G, dims["synopsis"], dims["studio"])
        v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
        st.session_state.v_user_initial = v_user_0.copy()
        st.session_state.v_user = v_user_0
        st.session_state.excluded_ids = excluded_ids
        if st.session_state.get("persona_mode") == "Not Selected":
            st.session_state.persona_mode = "Eyecandy First (Default)"

    current_sidebar_params = (
        mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
    )
    prev_params = st.session_state.get("sidebar_params")
    if prev_params is not None and prev_params != current_sidebar_params:
        st.session_state.deck = []
    st.session_state.sidebar_params = current_sidebar_params

    if st.session_state.get("show_onboarding_modal", False):
        open_onboarding_modal(
            df, V_i, id_to_idx, anime_vectors, vectorizers, dims, all_genres,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
        )

    liked_ids = st.session_state.get("liked_ids", [])
    passed_ids = st.session_state.get("passed_ids", [])
    watched_ids = st.session_state.get("watched_ids", [])

    st.caption(
        f"Source: **{data_source_label}** • "
        f"Persona: **{st.session_state.get('persona_mode', 'Not Selected')}** • "
        f"Catalog: **{len(df)} anime** • "
        f"Swiped: **{st.session_state.get('swipe_count', 0)}** • "
        f"Liked: **{len(liked_ids)}** • "
        f"Watched: **{len(watched_ids)}**"
    )

    st.subheader("⏱️ Episode Length Preference")
    def _on_length_filter_change():
        st.session_state.deck = []

    length_filter = st.selectbox(
        "Format / Episode Length:",
        [
            "All Lengths",
            "Movie / One-Shot (1 ep)",
            "Short (2–13 eps — 1 Cour)",
            "Medium (14–26 eps — 2 Cours)",
            "Long (27–52 eps — 3–4 Cours)",
            "Endless / Marathon (> 52 eps)"
        ],
        index=0,
        key="length_filter_select",
        on_change=_on_length_filter_change
    )

    length_excluded = _get_length_excluded_ids(df, length_filter)
    active_card_invalid = bool(
        st.session_state.get("deck") and st.session_state.deck[0] in length_excluded
    )

    if not st.session_state.get("deck") or active_card_invalid:
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, st.session_state.v_user,
            _get_all_excluded_ids(df, length_filter), st.session_state.swiped_titles, 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, id_to_idx=id_to_idx
        )

    if st.session_state.get("last_action_msg"):
        st.toast(st.session_state.last_action_msg)

    # Nếu deck hiện tại cạn nhưng vẫn còn thẻ đang trong cooldown -> mở khóa cooldown ngay để user tiếp tục
    if not st.session_state.get("deck") and st.session_state.get("cooldown_ids"):
        st.session_state.cooldown_ids = {}
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, st.session_state.v_user,
            _get_effective_excluded_ids(), st.session_state.swiped_titles, 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, id_to_idx=id_to_idx
        )

    if not st.session_state.get("deck"):
        st.warning("🎉 You have swiped through all available anime in the current filter! Click **Change Persona / Reset**, revert an anime below, or lower the Minimum Rating Score in the sidebar.")
    else:
        current_aid = st.session_state.deck[0]
        current_anime = anime_lookup[current_aid]
        v_anime = anime_vectors[current_aid]

        if np.linalg.norm(st.session_state.v_user) > 1e-9 and np.linalg.norm(v_anime) > 1e-9:
            sim_val = float(cosine_similarity(st.session_state.v_user.reshape(1, -1), v_anime.reshape(1, -1))[0, 0])
        else:
            sim_val = 0.0

        user_sub = split_full_vector_to_subspaces(st.session_state.v_user, dims)
        anime_sub = split_full_vector_to_subspaces(v_anime, dims)
        exp_dict = explain_recommendation(user_sub, anime_sub, vectorizers, top_n=3)
        exp_text = explanation_to_text(exp_dict)

        pos_keywords = []
        for feat in ["genres", "synopsis", "studio"]:
            pos_keywords.extend([f"{term} ({score:.2f})" for term, score in exp_dict["positive"][feat]])

        col_card, col_side = st.columns([1.2, 1])

        with col_card:
            render_tinder_draggable_card(
                anime_row=current_anime,
                sim_score=sim_val,
                explanation_text=exp_text,
                pos_tags=pos_keywords
            )

            if st.button("BRIDGE_SWIPE_PASS", key="btn_bridge_pass"):
                process_swipe(
                    "pass", df, V_i, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight, length_filter
                )
                st.session_state.pending_ls_save = True
                st.rerun()
            if st.button("BRIDGE_SWIPE_LIKE", key="btn_bridge_like"):
                process_swipe(
                    "like", df, V_i, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight, length_filter
                )
                st.session_state.pending_ls_save = True
                st.rerun()
            if st.button("BRIDGE_SWIPE_WATCHED", key="btn_bridge_watched"):
                process_swipe(
                    "watched", df, V_i, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight, length_filter
                )
                st.session_state.pending_ls_save = True
                st.rerun()

        with col_side:
            st.markdown("### 🧠 Real-Time Vector & Deck Queue")
            if st.session_state.get("last_action_msg"):
                st.info(st.session_state.last_action_msg)

            f_role = str(current_anime.get("franchise_role", "ENTRY")).upper()
            p_title = str(current_anime.get("parent_title", "")).strip()
            if f_role in ("SEQUEL", "PREQUEL", "SPIN_OFF") and p_title and p_title.lower() != str(current_anime["title"]).lower():
                role_label = {
                    "SEQUEL": "⏩ Sequel of",
                    "PREQUEL": "⏪ Prequel to",
                    "SPIN_OFF": "🔀 Spin-off / Side Story of"
                }[f_role]
                st.warning(f"**Franchise Relation:** {role_label} **{p_title}**")

            st.write(f"**Recommendation Rationale:** *{exp_text}*")
            c_g, c_s, c_v = st.columns(3)
            for idx, feat in enumerate(["genres", "synopsis", "studio"]):
                with [c_g, c_s, c_v][idx]:
                    st.markdown(f"**+{feat.capitalize()}**")
                    if exp_dict["positive"][feat]:
                        for term, sc in exp_dict["positive"][feat]:
                            st.caption(f"🟢 `{term}` (+{sc:.2f})")
                    else:
                        st.caption("*(None yet)*")

            st.markdown("---")
            st.markdown("**🃏 Upcoming Diverse Deck (MMR + Studio Cap):**")
            upcoming_rows = []
            for rank_idx, u_aid in enumerate(st.session_state.deck[:5], start=1):
                r = anime_lookup[u_aid]
                upcoming_rows.append({
                    "#": "👉 Active" if rank_idx == 1 else f"#{rank_idx}",
                    "Anime": r["title"],
                    "Score": r["score"],
                    "Studio": get_main_studio(r["studios"]),
                    "Genres": r["genres"]
                })
            st.dataframe(pd.DataFrame(upcoming_rows), use_container_width=True, hide_index=True)

    # Khu vực quản lý toàn diện: Liked (có Revert) • Passed (có Revert) • Watched (có Revert) • Compare
    with st.expander(
        f"💚 Liked ({len(liked_ids)}) • 👁️ Watched ({len(watched_ids)}) • ❌ Passed ({len(passed_ids)}) • ⚖️ Compare",
        expanded=False
    ):
        tab_liked, tab_watched, tab_passed, tab_compare = st.tabs([
            f"💚 Liked Watchlist ({len(liked_ids)})",
            f"👁️ Already Watched ({len(watched_ids)})",
            f"❌ Passed History ({len(passed_ids)})",
            "⚖️ Side-by-Side Comparison"
        ])

        with tab_liked:
            if liked_ids:
                valid_liked = [aid for aid in liked_ids if aid in anime_lookup]
                v_u = st.session_state.v_user
                u_norm = float(np.linalg.norm(v_u))
                has_profile = u_norm > 1e-9

                # 1. Tính Cosine Similarity thực (đã chia chuẩn L2) và quy đổi sang thang % phân hóa rõ ràng
                scored_liked = []
                for aid in valid_liked:
                    row = anime_lookup[aid]
                    if has_profile and aid in anime_vectors:
                        v_i = anime_vectors[aid]
                        i_norm = float(np.linalg.norm(v_i))
                        cos_sim = float(np.dot(v_i, v_u) / (i_norm * u_norm)) if i_norm > 1e-9 else 0.0
                        # Trong không gian vector nội dung anime, cos_sim thường nằm trong khoảng [0.0, 0.65]
                        vibe_pct = int(np.clip(45 + (cos_sim / 0.65) * 53, 40, 98))
                    else:
                        vibe_pct = int(np.clip((float(row.get("score", 7.5)) / 10.0) * 90, 50, 92))
                    scored_liked.append((aid, row, vibe_pct))

                # 2. Hàm tạo lời giải thích cá nhân hóa dựa trên lịch sử Like gần nhất & DNA thể loại
                def _build_pick_explanation(target_aid: int, target_row: pd.Series, pct: int) -> list[str]:
                    reasons = []
                    t_genres = {g.strip() for g in re.split(r"[|,;]", str(target_row.get("genres", ""))) if g.strip()}
                    t_themes = {t.strip() for t in re.split(r"[|,;]", str(target_row.get("themes", ""))) if t.strip()}
                    t_tags = t_genres | t_themes
                    t_studio = get_main_studio(target_row.get("studios", ""))

                    # A. Tìm các bộ user vừa Like/Boost gần đây có độ tương đồng vector cao nhất với bộ được pick
                    recent_positives = [
                        aid for aid in reversed(liked_ids + watched_ids)
                        if aid != target_aid and aid in anime_lookup and aid in anime_vectors
                    ][:12]

                    if recent_positives and target_aid in anime_vectors:
                        v_t = anime_vectors[target_aid]
                        t_n = float(np.linalg.norm(v_t))
                        sim_peers = []
                        for p_aid in recent_positives:
                            v_p = anime_vectors[p_aid]
                            p_n = float(np.linalg.norm(v_p))
                            if t_n > 1e-9 and p_n > 1e-9:
                                s = float(np.dot(v_t, v_p) / (t_n * p_n))
                                sim_peers.append((s, anime_lookup[p_aid]["title"]))
                        sim_peers.sort(reverse=True)
                        top_peers = [name for s, name in sim_peers[:2] if s > 0.12]
                        if top_peers:
                            reasons.append(f"🔗 **Chung chất phim (Vibe):** Rất gần với **{', '.join(top_peers)}** mà bạn đã chọn gần đây.")

                    # B. Tìm các thể loại/chủ đề giao nhau với gu tổng thể của user
                    user_top_tags = {}
                    for u_aid in (liked_ids + watched_ids):
                        if u_aid in anime_lookup and u_aid != target_aid:
                            u_r = anime_lookup[u_aid]
                            for tag in re.split(r"[|,;]", f"{u_r.get('genres', '')}|{u_r.get('themes', '')}"):
                                tag = tag.strip()
                                if tag:
                                    user_top_tags[tag] = user_top_tags.get(tag, 0) + 1

                    shared_tags = sorted(
                        [tag for tag in t_tags if tag in user_top_tags],
                        key=lambda x: user_top_tags[x],
                        reverse=True
                    )[:3]
                    if shared_tags:
                        reasons.append(f"🧬 **Đúng DNA thể loại:** Khớp trực tiếp với gu **{', '.join(shared_tags)}** đang chiếm ưu thế trong hồ sơ của bạn.")

                    # C. Lý do về nhịp độ xem & chất lượng thực tế
                    ep_num = pd.to_numeric(target_row.get("episodes", 0), errors="coerce")
                    fmt = str(target_row.get("type", "TV")).upper()
                    score_val = target_row.get("score", "N/A")
                    if fmt == "MOVIE":
                        pacing_txt = "gọn trọn vẹn trong 1 buổi tối (Movie)"
                    elif pd.notna(ep_num) and 1 <= ep_num <= 13:
                        pacing_txt = f"nhịp gọn nhẹ chỉ **{int(ep_num)} tập** (dễ cày hết trong 1 cuối tuần)"
                    elif pd.notna(ep_num) and ep_num > 13:
                        pacing_txt = f"series dài hơi **{int(ep_num)} tập** để theo dõi sâu"
                    else:
                        pacing_txt = "nhịp độ vừa vặn để bắt đầu ngay"

                    studio_txt = f" của **{t_studio}**" if t_studio and t_studio != "Unknown" else ""
                    reasons.append(f"⭐ **Chất lượng & Thời lượng:** Đạt **{score_val}/10**{studio_txt}, {pacing_txt}.")

                    return reasons

                # 3. Thanh "Watch-Next Decision Bar"
                c_filter, c_pick = st.columns([3.2, 1.4])
                with c_filter:
                    watch_mood = st.selectbox(
                        "What are you in the mood to watch right now?",
                        [
                            "🔥 Best Match with My Current Vibe",
                            "🎬 Movie Night (Finish in ~2 Hours)",
                            "⚡ Weekend Binge (Short Series ≤ 13 Eps)",
                            "⭐ Highest Rated First",
                            "🕒 Recently Liked First"
                        ],
                        key="liked_sort_mood",
                        label_visibility="collapsed"
                    )

                display_list = list(reversed(scored_liked))
                if "Best Match" in watch_mood:
                    display_list = sorted(scored_liked, key=lambda x: (x[2], float(x[1].get("score", 0))), reverse=True)
                elif "Movie Night" in watch_mood:
                    movies_only = [x for x in scored_liked if str(x[1].get("type", "")).upper() == "MOVIE"]
                    display_list = sorted(movies_only, key=lambda x: x[2], reverse=True) if movies_only else display_list
                elif "Weekend Binge" in watch_mood:
                    short_tv = [
                        x for x in scored_liked
                        if str(x[1].get("type", "")).upper() == "TV"
                        and 1 <= pd.to_numeric(x[1].get("episodes", 99), errors="coerce") <= 13
                    ]
                    display_list = sorted(short_tv, key=lambda x: x[2], reverse=True) if short_tv else display_list
                elif "Highest Rated" in watch_mood:
                    display_list = sorted(scored_liked, key=lambda x: float(x[1].get("score", 0)), reverse=True)

                with c_pick:
                    if st.button("🎲 Pick 1 Tonight!", use_container_width=True, type="primary"):
                        top_pool = display_list[:3] if display_list else []
                        if top_pool:
                            # Tránh bốc trùng lại bộ vừa pick ngay trước đó nếu có >= 2 lựa chọn
                            prev_pick = st.session_state.get("tonight_pick_id")
                            candidates = [x for x in top_pool if x[0] != prev_pick] or top_pool
                            chosen_aid, _, _ = candidates[
                                int(np.random.default_rng().integers(0, len(candidates)))
                            ]
                            st.session_state.tonight_pick_id = chosen_aid

                # 4. Hiển thị thẻ "Tonight's Pick" kèm giải thích lý do chi tiết
                pick_id = st.session_state.get("tonight_pick_id")
                if pick_id and pick_id in anime_lookup and pick_id in liked_ids:
                    p_row = anime_lookup[pick_id]
                    p_pct = next((pct for a, _, pct in scored_liked if a == pick_id), 80)
                    why_list = _build_pick_explanation(pick_id, p_row, p_pct)
                    why_md = "\n".join([f"- {r}" for r in why_list])
                    st.info(
                        f"🎯 **Tonight's Pick: {p_row['title']}** (`{p_pct}% Vibe Match`)\n\n"
                        f"**Tại sao hệ thống chọn bộ này cho bạn lúc này:**\n{why_md}"
                    )

                st.caption("Click **👁️ Watched** once you finish a series to unlock its sequels, or **↩️ Revert** to undo.")

                # 5. Hiển thị toàn bộ danh sách các bộ đã Like bên dưới với % Match phân hóa chuẩn
                for l_aid, l_row, vibe_pct in display_list:
                    c_info, c_watch, c_btn = st.columns([3.6, 1.1, 0.9])
                    with c_info:
                        st.markdown(
                            f"**{l_row['title']}** `{vibe_pct}% Match` &nbsp;•&nbsp; ⭐ {l_row['score']} &nbsp;•&nbsp; "
                            f"🎬 {get_main_studio(l_row['studios'])} &nbsp;•&nbsp; `{l_row['genres']}`"
                        )
                    with c_watch:
                        if st.button("👁️ Watched", key=f"mark_watched_{l_aid}", use_container_width=True):
                            mark_liked_as_watched(
                                l_aid, df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                mmr_lambda, max_per_studio, franchise_thresh, min_score,
                                quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
                    with c_btn:
                        if st.button("↩️ Revert", key=f"revert_like_{l_aid}", use_container_width=True):
                            revert_swiped_anime(
                                l_aid, "liked", df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
            else:
                st.caption("No anime liked yet. Drag a card right (or press `→`) to add to your watchlist!")
        with tab_watched:
            if watched_ids:
                st.caption("Watched titles unlock their sequels without skewing your taste unless you click **❤️ Boost**.")
                boosted_ids = {
                    int(e["anime_id"])
                    for e in st.session_state.swipe_history
                    if isinstance(e, dict)
                    and (e.get("taste_boosted") or (e.get("action") == "watched" and e.get("from_liked")))
                }
                for w_aid in reversed(watched_ids):
                    if w_aid not in anime_lookup:
                        continue
                    w_row = anime_lookup[w_aid]
                    c_info, c_boost, c_meh, c_rev = st.columns([3.3, 0.95, 0.85, 0.9])
                    with c_info:
                        badge = " `❤️ Boosted`" if w_aid in boosted_ids else ""
                        st.markdown(
                            f"**{w_row['title']}**{badge} &nbsp;•&nbsp; ⭐ {w_row['score']} &nbsp;•&nbsp; "
                            f"🎬 {get_main_studio(w_row['studios'])}"
                        )
                    with c_boost:
                        if st.button("❤️ Boost", key=f"boost_watch_{w_aid}", use_container_width=True):
                            apply_watched_feedback(
                                w_aid, "boost", df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
                    with c_meh:
                        if st.button("👎 Meh", key=f"meh_watch_{w_aid}", use_container_width=True):
                            apply_watched_feedback(
                                w_aid, "dislike", df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
                    with c_rev:
                        if st.button("↩️ Revert", key=f"revert_watch_{w_aid}", use_container_width=True):
                            revert_swiped_anime(
                                w_aid, "watched", df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
            else:
                st.caption("No watched anime marked yet. Swipe down (or press `↓`) to skip shows you've already seen.")

        with tab_passed:
            if passed_ids:
                st.caption("Accidentally swiped left? Click **↩️ Revert** to undo its negative weight — it will return naturally after 8 swipes.")
                for p_aid in reversed(passed_ids):
                    if p_aid not in anime_lookup:
                        continue
                    p_row = anime_lookup[p_aid]
                    c_info, c_btn = st.columns([4.2, 1])
                    with c_info:
                        st.markdown(
                            f"**{p_row['title']}** &nbsp;•&nbsp; ⭐ {p_row['score']} &nbsp;•&nbsp; "
                            f"🎬 {get_main_studio(p_row['studios'])} &nbsp;•&nbsp; `{p_row['genres']}`"
                        )
                    with c_btn:
                        if st.button("↩️ Revert", key=f"revert_pass_{p_aid}", use_container_width=True):
                            revert_swiped_anime(
                                p_aid, "passed", df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight, length_filter
                            )
                            st.session_state.pending_ls_save = True
                            st.rerun()
            else:
                st.caption("No passed anime yet.")

        with tab_compare:
            anime_label_map = {f"{row['title']} (ID: {row['mal_id']})": int(row["mal_id"]) for _, row in df.iterrows()}
            labels_list = list(anime_label_map.keys())
            if len(labels_list) >= 2:
                s1, s2 = st.columns(2)
                with s1:
                    c1 = st.selectbox("Select Anime A:", options=labels_list, index=0)
                with s2:
                    options_b = [lbl for lbl in labels_list if lbl != c1]
                    c2 = st.selectbox("Select Anime B:", options=options_b, index=0)

                comp = compare_anime(df, anime_label_map[c1], anime_label_map[c2])
                comp_rows = []
                for attr in ["Genres", "Themes", "Studio"]:
                    vals = comp[attr]
                    comp_rows.append({
                        "Attribute": attr,
                        "Common": ", ".join(vals["common"]) if vals["common"] else "—",
                        f"Only in {comp['Title']['anime_a']}": ", ".join(vals["different_a"]) if vals["different_a"] else "—",
                        f"Only in {comp['Title']['anime_b']}": ", ".join(vals["different_b"]) if vals["different_b"] else "—",
                    })
                st.table(pd.DataFrame(comp_rows))

if __name__ == "__main__":
    main()
