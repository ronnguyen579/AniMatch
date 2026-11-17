import json
import streamlit as st
from streamlit_js_eval import get_local_storage, set_local_storage

STORAGE_KEY = "animatch_user_profile_v2"


def save_profile_to_browser(dataset_sig: str):
    """Đồng bộ trạng thái Persona, Onboarding Seed và lịch sử Like/Pass xuống localStorage."""
    payload = {
        "dataset_sig": dataset_sig,
        "persona_mode": st.session_state.persona_mode,
        "seed_params": st.session_state.get("seed_params", {}),
        "liked_ids": [int(x) for x in st.session_state.liked_ids],
        "passed_ids": [int(x) for x in st.session_state.passed_ids],
        "swipe_count": int(st.session_state.swipe_count),
    }
    set_local_storage(
        STORAGE_KEY,
        json.dumps(payload),
        component_key=f"save_ls_{st.session_state.swipe_count}_{len(st.session_state.liked_ids)}_{st.session_state.persona_mode}"
    )


def clear_browser_profile():
    """Xóa sạch dữ liệu lưu trữ trên trình duyệt khi người dùng Reset."""
    reset_idx = st.session_state.get("reset_counter", 0) + 1
    st.session_state.reset_counter = reset_idx
    set_local_storage(
        STORAGE_KEY,
        "",
        component_key=f"clear_ls_{reset_idx}"
    )


def try_restore_profile_from_browser(
    dataset_sig: str,
    anime_lookup: dict,
    anime_vectors,
    seeder,
    update_fn,
    gamma: float = 0.95,
    alpha: float = 0.30,
    beta: float = 0.15
) -> str:
    """
    Đọc localStorage từ trình duyệt và tái tạo lại trạng thái phiên làm việc.
    Trả về:
      - "pending": Đang chờ trình duyệt phản hồi dữ liệu ở lượt render đầu tiên.
      - "restored": Đã khôi phục thành công lịch sử quẹt & vector v_user.
      - "empty": Không có dữ liệu cũ hoặc dữ liệu thuộc bộ dataset khác.
    """
    raw_data = get_local_storage(STORAGE_KEY, component_key="read_animatch_ls")

    if raw_data is None:
        # Chỉ chờ tối đa 1 lượt render; nếu trình duyệt chặn JS thì thoát pending ngay
        if st.session_state.get("ls_wait_done", False):
            return "empty"
        st.session_state.ls_wait_done = True
        return "pending"

    if raw_data in ("", "null", "{}"):
        return "empty"

    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(data, dict):
            return "empty"

        if data.get("dataset_sig") != dataset_sig:
            return "empty"

        persona_mode = data.get("persona_mode", "Eyecandy First")
        seed_params = data.get("seed_params", {})
        liked_ids = [int(x) for x in data.get("liked_ids", []) if int(x) in anime_lookup]
        passed_ids = [int(x) for x in data.get("passed_ids", []) if int(x) in anime_lookup]

        # 1. Tái tạo vector gốc v_user_0 theo đúng Persona đã chọn lúc Onboarding
        if persona_mode == "Content First":
            selected_genres = seed_params.get("selected_genres", [])
            v_rebuilt, excluded = seeder.seed_cold_start("content_first", selected_genres)
            swiped_titles = []
        elif persona_mode == "Lapsed User":
            past_ids = [int(x) for x in seed_params.get("past_ids", []) if int(x) in anime_lookup]
            new_genres = seed_params.get("new_genres", [])
            l1 = float(seed_params.get("lambda_1", 0.4))
            l2 = float(seed_params.get("lambda_2", 0.6))
            v_rebuilt, excluded = seeder.seed_lapsed_user(past_ids, new_genres, l1, l2)
            swiped_titles = [str(anime_lookup[aid]["title"]) for aid in past_ids]
        else:
            v_rebuilt, excluded = seeder.seed_cold_start("eyecandy_first", [])
            swiped_titles = []

        if persona_mode == "Eyecandy First" and not liked_ids and not passed_ids:
            return "empty"

        # 2. Áp dụng lại các lượt Pass & Like vào vector v_rebuilt
        for aid in passed_ids:
            if aid in anime_vectors:
                eff_beta = (beta * 0.4) if not liked_ids else beta
                v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "pass", gamma, alpha, eff_beta)
                excluded.add(aid)
                swiped_titles.append(str(anime_lookup[aid]["title"]))

        for aid in liked_ids:
            if aid in anime_vectors:
                v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "like", gamma, alpha, beta)
                excluded.add(aid)
                swiped_titles.append(str(anime_lookup[aid]["title"]))

        # 3. Cập nhật vào Streamlit Session State
        st.session_state.persona_mode = persona_mode
        st.session_state.seed_params = seed_params
        st.session_state.v_user = v_rebuilt
        st.session_state.liked_ids = liked_ids
        st.session_state.passed_ids = passed_ids
        st.session_state.excluded_ids = excluded
        st.session_state.swiped_titles = swiped_titles
        st.session_state.swipe_count = int(data.get("swipe_count", len(liked_ids) + len(passed_ids)))
        st.session_state.onboarded = True
        st.session_state.show_onboarding_modal = False
        return "restored"
    except Exception:
        return "empty"