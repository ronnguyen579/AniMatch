import os
import glob
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import get_fallback_dataframe, normalize_dataframe, get_main_studio
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
@st.cache_data(show_spinner=False)
def load_and_prepare_catalog(csv_path: str | None):
    if csv_path:
        df = normalize_dataframe(pd.read_csv(csv_path))
    else:
        df = normalize_dataframe(get_fallback_dataframe())

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
        saved_history = data.get("swipe_history", [])

        if not liked_ids and not passed_ids:
            return False

        valid_ids = set(df["anime_id"].astype(int).tolist())
        id_to_title = dict(zip(df["anime_id"].astype(int), df["title"].astype(str)))

        v_initial = np.zeros(expected_dim)
        v_rebuilt = v_initial.copy()
        excluded = set()
        swiped_titles = []
        rebuilt_history = []

        if saved_history:
            liked_so_far = 0
            for item in saved_history:
                aid, act = int(item[0]), str(item[1])
                if aid in valid_ids and aid in anime_vectors:
                    eff_beta = (beta * 0.4) if liked_so_far == 0 else beta
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], act, gamma, alpha, eff_beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, act))
                    if act == "like":
                        liked_so_far += 1
        else:
            for aid in passed_ids:
                if aid in valid_ids and aid in anime_vectors:
                    eff_beta = (beta * 0.4) if not liked_ids else beta
                    v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "pass", gamma, alpha, eff_beta)
                    excluded.add(aid)
                    swiped_titles.append(id_to_title[aid])
                    rebuilt_history.append((aid, "pass"))
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
        "swiped_titles": [],
        "deck": [],
        "liked_ids": [],
        "passed_ids": [],
        "swipe_count": 0,
        "last_action_msg": "",
        "dataset_signature": "",
        "sidebar_params": None,
        "pending_ls_clear": False,
        "pending_ls_save": False,
        "ls_checked": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_swipe(
    action: str,
    df: pd.DataFrame,
    V_i,
    id_to_idx: dict,
    anime_lookup: dict,
    anime_vectors,
    gamma: float,
    alpha: float,
    beta: float,
    mmr_lambda: float,
    max_per_studio: int,
    franchise_thresh: int,
    min_score: float = 7.2,
    quality_weight: float = 0.25
):
    if not st.session_state.deck:
        return

    current_aid = st.session_state.deck.pop(0)
    anime_row = anime_lookup[current_aid]
    v_anime = anime_vectors[current_aid]

    effective_beta = (beta * 0.4) if len(st.session_state.liked_ids) == 0 else beta

    st.session_state.v_user = update_user_vector_rocchio(
        st.session_state.v_user, v_anime, action=action, gamma=gamma, alpha=alpha, beta=effective_beta
    )

    st.session_state.excluded_ids.add(current_aid)
    st.session_state.swiped_titles.append(str(anime_row["title"]))
    st.session_state.swipe_history.append((current_aid, action))
    st.session_state.swipe_count += 1

    if action == "like":
        st.session_state.liked_ids.append(current_aid)
        st.session_state.last_action_msg = f"💚 LIKED: **{anime_row['title']}**"
    else:
        st.session_state.passed_ids.append(current_aid)
        st.session_state.last_action_msg = f"❌ PASSED: **{anime_row['title']}**"

    st.session_state.pending_ls_save = True

    st.session_state.deck = generate_diverse_deck(
        df=df,
        anime_vectors=anime_vectors,
        v_user=st.session_state.v_user,
        excluded_ids=st.session_state.excluded_ids,
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
    
def revert_passed_anime(
    target_aid: int,
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
    quality_weight: float
):
    """Hoàn tác một bộ anime đã Pass: xóa khỏi lịch sử, tính lại v_user chính xác và đưa thẻ lên đầu deck."""
    if target_aid not in st.session_state.passed_ids:
        return

    # 1. Gỡ bỏ khỏi danh sách đã Pass, excluded_ids và swiped_titles
    st.session_state.passed_ids = [aid for aid in st.session_state.passed_ids if aid != target_aid]
    st.session_state.excluded_ids.discard(target_aid)

    target_title = str(anime_lookup[target_aid]["title"])
    if target_title in st.session_state.swiped_titles:
        st.session_state.swiped_titles.remove(target_title)

    # 2. Xóa thao tác này khỏi swipe_history và tái tạo lại v_user từ vector gốc
    st.session_state.swipe_history = [
        (aid, act) for (aid, act) in st.session_state.swipe_history if aid != target_aid
    ]
    st.session_state.swipe_count = len(st.session_state.swipe_history)

    v_rebuilt = (
        st.session_state.v_user_initial.copy()
        if st.session_state.v_user_initial is not None
        else np.zeros_like(st.session_state.v_user)
    )
    liked_so_far = 0
    for aid, act in st.session_state.swipe_history:
        eff_beta = (beta * 0.4) if liked_so_far == 0 else beta
        v_rebuilt = update_user_vector_rocchio(
            v_rebuilt, anime_vectors[aid], action=act, gamma=gamma, alpha=alpha, beta=eff_beta
        )
        if act == "like":
            liked_so_far += 1

    st.session_state.v_user = v_rebuilt

    # 3. Tạo lại deck mới và đặt bộ anime vừa revert lên vị trí đầu tiên (#1) để user thao tác ngay
    new_deck = generate_diverse_deck(
        df=df,
        anime_vectors=anime_vectors,
        v_user=st.session_state.v_user,
        excluded_ids=st.session_state.excluded_ids,
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
    new_deck = [aid for aid in new_deck if aid != target_aid]
    st.session_state.deck = [target_aid] + new_deck[:7]
    st.session_state.last_action_msg = f"↩️ REVERTED: **{target_title}** is back at the top of your deck!"


# ==============================================================================
# HÀM ĐIỀU PHỐI ỨNG DỤNG CHÍNH (MAIN APPLICATION)
# ==============================================================================
def main():
    init_session_state()

    # 1. Xử lý đồng bộ localStorage (Clear / Save / Load an toàn bằng .get)
    if st.session_state.get("pending_ls_clear", False):
        clear_browser_profile()
        st.session_state.pending_ls_clear = False

    if st.session_state.get("pending_ls_save", False):
        save_profile_to_browser()
        st.session_state.pending_ls_save = False

    # 2. Thanh tiêu đề & nút đổi Persona / Reset
    top_c1, top_c2 = st.columns([3.5, 1.5])
    with top_c1:
        st.markdown("## 🔥 AniMatch")
    with top_c2:
        if st.button("🧭 Change Persona / Reset", use_container_width=True):
            st.session_state.pending_ls_clear = True
            st.session_state.show_onboarding_modal = True
            st.rerun()

    # 3. Tự động nạp CSV cục bộ hoặc dữ liệu mẫu dự phòng (đã cache)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_csv_files = sorted(glob.glob(os.path.join(current_dir, "*.csv")))

    if local_csv_files:
        if len(local_csv_files) > 1:
            selected_local_path = st.sidebar.selectbox(
                "📁 Select Local CSV Dataset:",
                options=local_csv_files,
                format_func=lambda p: os.path.basename(p)
            )
        else:
            selected_local_path = local_csv_files[0]

        try:
            df, all_genres, anime_lookup = load_and_prepare_catalog(selected_local_path)
            file_name = os.path.basename(selected_local_path)
            sig = f"local_{file_name}_{len(df)}"
            data_source_label = f"Local CSV ({file_name})"
        except Exception as e:
            st.error(f"Error reading `{os.path.basename(selected_local_path)}`: {e}")
            st.stop()
    else:
        df, all_genres, anime_lookup = load_and_prepare_catalog(None)
        sig = f"default_{len(df)}"
        data_source_label = "Built-in Sample Dataset"

    # Reset state khi đổi sang dataset khác
    if st.session_state.get("dataset_signature", "") != sig:
        st.session_state.dataset_signature = sig
        st.session_state.onboarded = False
        st.session_state.v_user_initial = None
        st.session_state.v_user = None
        st.session_state.swipe_history = []
        st.session_state.excluded_ids = set()
        st.session_state.swiped_titles = []
        st.session_state.deck = []
        st.session_state.liked_ids = []
        st.session_state.passed_ids = []
        st.session_state.swipe_count = 0
        st.session_state.last_action_msg = ""
        st.session_state.show_onboarding_modal = True
        st.session_state.ls_checked = False

    # 4. Thanh tham số thuật toán ở Sidebar
    with st.sidebar:
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

    # 5. Xây dựng không gian vector đa phương thức (hỗ trợ cả trả về 4 hoặc 5 biến)
    space_res = build_multimodal_space(df)
    if len(space_res) == 5:
        V_i, id_to_idx, anime_vectors, vectorizers, dims = space_res
    else:
        V_i, anime_vectors, vectorizers, dims = space_res
        id_to_idx = {int(row["anime_id"]): idx for idx, row in df.iterrows()}

    expected_dim = V_i.shape[1]

    # 6. Khôi phục dữ liệu từ localStorage trình duyệt (chạy 1 lần khi mở/F5 trang)
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
                st.session_state.excluded_ids, st.session_state.swiped_titles, 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, id_to_idx=id_to_idx
            )
            st.toast("🔄 Restored your previous swipe session from browser storage!")

    # 7. Khởi tạo vector mặc định nếu chưa có
    if st.session_state.get("v_user") is None or st.session_state.v_user.shape[0] != expected_dim:
        seeder = ContextualPriorSeeder(anime_vectors, vectorizers["genres"], W_G, dims["synopsis"], dims["studio"])
        v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
        st.session_state.v_user_initial = v_user_0.copy()
        st.session_state.v_user = v_user_0
        st.session_state.excluded_ids = excluded_ids
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, v_user_0, excluded_ids, [], 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, id_to_idx=id_to_idx
        )

    # 8. Tự động làm mới deck ngay lập tức nếu người dùng thay đổi thanh trượt Sidebar
    current_sidebar_params = (mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight)
    prev_params = st.session_state.get("sidebar_params")
    if prev_params is not None and prev_params != current_sidebar_params:
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, st.session_state.v_user,
            st.session_state.excluded_ids, st.session_state.swiped_titles, 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, id_to_idx=id_to_idx
        )
    st.session_state.sidebar_params = current_sidebar_params

    # 9. Mở cửa sổ Onboarding nếu cần
    if st.session_state.get("show_onboarding_modal", False):
        open_onboarding_modal(
            df, V_i, id_to_idx, anime_vectors, vectorizers, dims, all_genres,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
        )

    # 10. Thanh trạng thái tổng quan
    st.caption(
        f"Source: **{data_source_label}** • "
        f"Persona: **{st.session_state.get('persona_mode', 'Not Selected')}** • "
        f"Catalog: **{len(df)} anime** • "
        f"Swiped: **{st.session_state.get('swipe_count', 0)}** • "
        f"Liked: **{len(st.session_state.get('liked_ids', []))}**"
    )

    if st.session_state.get("last_action_msg"):
        st.toast(st.session_state.last_action_msg)

    # 11. Khu vực chính: Thẻ quẹt Tinder & Bảng phân tích vector thời gian thực
    if not st.session_state.get("deck"):
        st.warning("🎉 You have swiped through all available anime in the current filter! Click **Change Persona / Reset**, revert a passed anime below, or lower the Minimum Rating Score in the sidebar.")
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
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.session_state.pending_ls_save = True
                st.rerun()
            if st.button("BRIDGE_SWIPE_LIKE", key="btn_bridge_like"):
                process_swipe(
                    "like", df, V_i, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.session_state.pending_ls_save = True
                st.rerun()

        with col_side:
            st.markdown("### 🧠 Real-Time Vector & Deck Queue")
            if st.session_state.get("last_action_msg"):
                st.info(st.session_state.last_action_msg)

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

    # 12. Khu vực quản lý Liked Watchlist, Passed History (Revert) & Compare View
    liked_ids = st.session_state.get("liked_ids", [])
    passed_ids = st.session_state.get("passed_ids", [])

    with st.expander(
        f"💚 Liked ({len(liked_ids)}) • ❌ Passed ({len(passed_ids)}) • ⚖️ Compare",
        expanded=False
    ):
        tab_liked, tab_passed, tab_compare = st.tabs([
            f"💚 Liked Watchlist ({len(liked_ids)})",
            f"❌ Passed History & Revert ({len(passed_ids)})",
            "⚖️ Side-by-Side Comparison"
        ])

        with tab_liked:
            if liked_ids:
                liked_df = df[df["anime_id"].isin(liked_ids)][
                    ["anime_id", "title", "type", "genres", "studio", "score", "episodes", "year"]
                ]
                st.dataframe(liked_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No anime liked yet. Drag a card to the right to like it!")

        with tab_passed:
            if passed_ids:
                st.caption("Accidentally swiped left? Click **↩️ Revert** to restore the card to the top of your deck and undo its negative weight.")
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
                            revert_passed_anime(
                                p_aid, df, V_i, id_to_idx, anime_lookup, anime_vectors,
                                gamma, alpha, beta, mmr_lambda, max_per_studio,
                                franchise_thresh, min_score, quality_weight
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
