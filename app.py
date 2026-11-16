import os
import glob
import re
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
from storage_manager import (
    save_profile_to_browser,
    clear_browser_profile,
    try_restore_profile_from_browser,
)
from card_component import render_tinder_draggable_card

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
    df, V_i, V_artstyle, id_to_idx, anime_vectors, vectorizers, dims, all_genres,
    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
):
    st.session_state.show_onboarding_modal = False

    st.caption("Choose how you want to initialize your preference profile. Once confirmed, this menu closes and takes you directly to the swipe deck.")

    persona_choice = st.radio(
        "Select Your User Persona:",
        [
            "👀 Eyecandy First (Zero-shot visual discovery — 8 distinct Danbooru artstyles)",
            "🏷️ Content First (Cold-start — pick your favorite genres & themes)",
            "🕰️ Lapsed User (Returning user — blend past watch history with new genres)"
        ]
    )

    seeder = LapsedUserPriorSeeder(
        anime_vectors=anime_vectors,
        genre_vectorizer=vectorizers["genres"],
        w_g=W_G,
        dim_sypnosis=dims["synopsis"],
        dim_studio=dims["studio"],
        dim_artstyle=dims["artstyle"]
    )

    if persona_choice.startswith("👀 Eyecandy First"):
        st.info("No upfront survey needed! We deal 8 cards with maximally distinct Danbooru visual artstyles so your taste evolves from what catches your eye.")
        if st.button("🔥 Start Swiping Now", type="primary", use_container_width=True):
            v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
            st.session_state.persona_mode = "Eyecandy First"
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = []
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, [], 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=True
            )
            st.session_state.onboarded = True
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
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = []
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, [], 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=False
            )
            st.session_state.onboarded = True
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
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = [str(df[df["anime_id"] == aid].iloc[0]["title"]) for aid in past_ids]
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, st.session_state.swiped_titles, 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=False
            )
            st.session_state.onboarded = True
            st.rerun()


# ==============================================================================
# QUẢN LÝ TRẠNG THÁI PHIÊN & XỬ LÝ HÀNH ĐỘNG QUẸT
# ==============================================================================
def init_session_state():
    defaults = {
        "onboarded": False,
        "show_onboarding_modal": True,
        "persona_mode": "Not Selected",
        "v_user": None,
        "excluded_ids": set(),
        "swiped_titles": [],
        "deck": [],
        "liked_ids": [],
        "passed_ids": [],
        "swipe_count": 0,
        "last_action_msg": "",
        "dataset_signature": "",
        "ls_checked": False,
        "reset_counter": 0,
        "sidebar_params": None 
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_swipe(
    action: str,
    df: pd.DataFrame,
    V_i,
    V_artstyle,
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
    st.session_state.swipe_count += 1

    if action == "like":
        st.session_state.liked_ids.append(current_aid)
        st.session_state.last_action_msg = f"💚 LIKED: **{anime_row['title']}**"
    else:
        st.session_state.passed_ids.append(current_aid)
        st.session_state.last_action_msg = f"❌ PASSED: **{anime_row['title']}**"

    is_eyecandy = (st.session_state.persona_mode == "Eyecandy First")
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
        V_artstyle=V_artstyle,
        id_to_idx=id_to_idx,
        is_eyecandy_mode=is_eyecandy
    )
    save_profile_to_browser(st.session_state.dataset_signature)


# ==============================================================================
# HÀM ĐIỀU PHỐI ỨNG DỤNG CHÍNH (MAIN APPLICATION)
# ==============================================================================
def main():
    init_session_state()

    top_c1, top_c2 = st.columns([3.5, 1.5])
    with top_c1:
        st.markdown("## 🔥 AniMatch")
    with top_c2:
        if st.button("🧭 Change Persona / Reset", use_container_width=True):
            st.session_state.reset_counter += 1
            clear_browser_profile()
            st.session_state.show_onboarding_modal = True
            st.rerun()

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

    if st.session_state.dataset_signature != sig:
        st.session_state.dataset_signature = sig
        st.session_state.onboarded = False
        st.session_state.v_user = None
        st.session_state.excluded_ids = set()
        st.session_state.swiped_titles = []
        st.session_state.deck = []
        st.session_state.liked_ids = []
        st.session_state.passed_ids = []
        st.session_state.swipe_count = 0
        st.session_state.last_action_msg = ""
        st.session_state.show_onboarding_modal = True

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

    V_i, V_artstyle, id_to_idx, anime_vectors, vectorizers, dims = build_multimodal_space(df)

    expected_dim = V_i.shape[1]
    if st.session_state.v_user is None or st.session_state.v_user.shape[0] != expected_dim:
        seeder = ContextualPriorSeeder(
            anime_vectors, vectorizers["genres"], W_G, dims["synopsis"], dims["studio"], dims["artstyle"]
        )
        v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
        st.session_state.v_user = v_user_0
        st.session_state.excluded_ids = excluded_ids
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, v_user_0, excluded_ids, [], 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=True
        )

    is_eyecandy = (st.session_state.persona_mode == "Eyecandy First")
    current_sidebar_params = (mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight)
    if st.session_state.sidebar_params is not None and st.session_state.sidebar_params != current_sidebar_params:
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, st.session_state.v_user,
            st.session_state.excluded_ids, st.session_state.swiped_titles, 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
            V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=is_eyecandy
        )
    st.session_state.sidebar_params = current_sidebar_params

    if not st.session_state.ls_checked:
        ls_status = try_restore_profile_from_browser(
            dataset_sig=sig,
            df=df,
            anime_lookup=anime_lookup,
            anime_vectors=anime_vectors,
            expected_dim=expected_dim,
            update_fn=update_user_vector_rocchio,
            gamma=gamma,
            alpha=alpha,
            beta=beta
        )
        if ls_status == "restored":
            st.session_state.ls_checked = True
            is_ec = (st.session_state.persona_mode == "Eyecandy First")
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, st.session_state.v_user,
                st.session_state.excluded_ids, st.session_state.swiped_titles, 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight,
                V_i=V_i, V_artstyle=V_artstyle, id_to_idx=id_to_idx, is_eyecandy_mode=is_ec
            )
            st.toast("🔄 Restored your previous swipe session from browser storage!")
        elif ls_status == "empty":
            st.session_state.ls_checked = True

    if st.session_state.show_onboarding_modal:
        open_onboarding_modal(
            df, V_i, V_artstyle, id_to_idx, anime_vectors, vectorizers, dims, all_genres,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
        )

    st.caption(
        f"Source: **{data_source_label}** • "
        f"Persona: **{st.session_state.persona_mode}** • "
        f"Catalog: **{len(df)} anime** • "
        f"Swiped: **{st.session_state.swipe_count}** • "
        f"Liked: **{len(st.session_state.liked_ids)}**"
    )

    if st.session_state.last_action_msg:
        st.toast(st.session_state.last_action_msg)

    if not st.session_state.deck:
        st.warning("🎉 You have swiped through all available anime in the current filter! Click **Change Persona / Reset** or lower the Minimum Rating Score in the sidebar.")
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
        for feat in ["genres", "artstyle", "synopsis", "studio"]:
            pos_keywords.extend([f"{term} ({score:.2f})" for term, score in exp_dict["positive"].get(feat, [])])

        # Nếu là Eyecandy First ở lượt đầu (chưa có từ khóa dương), hiển thị luôn tag Danbooru Artstyle của thẻ
        if not pos_keywords and str(current_anime.get("artstyle_tags", "")).strip():
            raw_art = [t.strip() for t in str(current_anime["artstyle_tags"]).split("|") if t.strip()][:4]
            pos_keywords = [f"🎨 {t}" for t in raw_art]

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
                    "pass", df, V_i, V_artstyle, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.rerun()
            if st.button("BRIDGE_SWIPE_LIKE", key="btn_bridge_like"):
                process_swipe(
                    "like", df, V_i, V_artstyle, id_to_idx, anime_lookup, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.rerun()

        with col_side:
            st.markdown("### 🧠 Real-Time Vector & Deck Queue")
            if st.session_state.last_action_msg:
                st.info(st.session_state.last_action_msg)

            st.write(f"**Recommendation Rationale:** *{exp_text}*")
            c_g, c_a, c_s, c_v = st.columns(4)
            for idx, feat in enumerate(["genres", "artstyle", "synopsis", "studio"]):
                with [c_g, c_a, c_s, c_v][idx]:
                    st.markdown(f"**+{feat.capitalize()}**")
                    if exp_dict["positive"].get(feat):
                        for term, sc in exp_dict["positive"][feat]:
                            st.caption(f"🟢 `{term}` (+{sc:.2f})")
                    else:
                        st.caption("*(None yet)*")

            st.markdown("---")
            st.markdown("**🃏 Upcoming Diverse Deck (Artstyle MMR + Studio Cap):**")
            BAD_ART_TAGS = {"male_focus", "female_focus", "cover", "poster", "solo_focus"}
            upcoming_rows = []
            for rank_idx, u_aid in enumerate(st.session_state.deck[:5], start=1):
                r = anime_lookup[u_aid]
                raw_tags = str(r.get("artstyle_tags", "")).strip()
                # Nếu artstyle_tags trống, lấy từ visual_tags_full (cũng là tag thị giác WD14), không lấy genres
                if not raw_tags:
                    raw_tags = str(r.get("visual_tags_full", "")).strip()

                clean_tags = [
                    t.strip() for t in raw_tags.split("|")
                    if t.strip() and t.strip() not in BAD_ART_TAGS
                ][:4]

                upcoming_rows.append({
                    "#": "👉 Active" if rank_idx == 1 else f"#{rank_idx}",
                    "Anime": r["title"],
                    "Score": r["score"],
                    "Studio": get_main_studio(r["studios"]),
                    "Artstyle (WD14)": ", ".join(clean_tags) if clean_tags else "standard_anime_style"
                })
            st.dataframe(pd.DataFrame(upcoming_rows), use_container_width=True, hide_index=True)

    with st.expander(f"💚 Liked Anime ({len(st.session_state.liked_ids)}) & ⚖️ Side-by-Side Compare View", expanded=False):
        if st.session_state.liked_ids:
            display_cols = ["anime_id", "title", "type", "genres", "studio", "score", "episodes", "year"]
            if "artstyle_tags" in df.columns and df["artstyle_tags"].str.len().sum() > 0:
                display_cols.insert(4, "artstyle_tags")
            liked_df = df[df["anime_id"].isin(st.session_state.liked_ids)][display_cols]
            st.dataframe(liked_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No anime liked yet. Drag a card to the right to like it!")

        st.markdown("#### ⚖️ Side-by-Side Anime Comparison")
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
            for attr in ["Genres", "Themes", "Artstyle", "Studio"]:
                if attr not in comp:
                    continue
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
