import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_js_eval import get_local_storage, set_local_storage

STORAGE_KEY = "animatch_user_profile_v2"


def save_profile_to_browser(dataset_sig: str):
    """Đồng bộ trạng thái Like/Pass và Persona xuống localStorage của trình duyệt."""
    payload = {
        "dataset_sig": dataset_sig,
        "persona_mode": st.session_state.persona_mode,
        "liked_ids": [int(x) for x in st.session_state.liked_ids],
        "passed_ids": [int(x) for x in st.session_state.passed_ids],
        "swipe_count": int(st.session_state.swipe_count),
    }
    # Đổi key mỗi lần lưu để ép component JS thực thi lệnh ghi mới nhất
    set_local_storage(
        STORAGE_KEY,
        json.dumps(payload),
        component_key=f"save_ls_{st.session_state.swipe_count}_{len(st.session_state.liked_ids)}"
    )


def clear_browser_profile():
    """Xóa sạch dữ liệu lưu trữ trên trình duyệt khi người dùng Reset."""
    set_local_storage(STORAGE_KEY, "", component_key=f"clear_ls_{st.session_state.get('reset_counter', 0)}")


def try_restore_profile_from_browser(
    dataset_sig: str,
    df: pd.DataFrame,
    anime_lookup: dict,
    anime_vectors: dict,
    expected_dim: int,
    update_fn,
    gamma: float = 0.95,
    alpha: float = 0.30,
    beta: float = 0.15
) -> str:
    """
    Đọc localStorage từ trình duyệt.
    Trả về:
      - "pending": Đang chờ trình duyệt gửi dữ liệu lên ở lượt rerun đầu tiên.
      - "restored": Đã khôi phục thành công lịch sử quẹt & vector v_user.
      - "empty": Trình duyệt không có dữ liệu cũ hoặc dữ liệu thuộc dataset khác.
    """
    raw_data = get_local_storage(STORAGE_KEY, component_key="read_animatch_ls")

    # Ở lượt render đầu tiên ngay sau F5, streamlit-js-eval luôn trả về None trước khi rerun
    if raw_data is None:
        return "pending"

    if raw_data == "" or raw_data == "null":
        return "empty"

    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        if not isinstance(data, dict):
            return "empty"

        # Chỉ khôi phục nếu trùng đúng bộ dataset đang mở
        if data.get("dataset_sig") != dataset_sig:
            return "empty"

        liked_ids = [int(x) for x in data.get("liked_ids", []) if int(x) in anime_lookup]
        passed_ids = [int(x) for x in data.get("passed_ids", []) if int(x) in anime_lookup]

        if not liked_ids and not passed_ids:
            return "empty"

        # Tái tạo lại vector v_user từ lịch sử đã lưu
        v_rebuilt = np.zeros(expected_dim)
        excluded = set()
        swiped_titles = []

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

        st.session_state.persona_mode = data.get("persona_mode", "Eyecandy First")
        st.session_state.v_user = v_rebuilt
        st.session_state.liked_ids = liked_ids
        st.session_state.passed_ids = passed_ids
        st.session_state.excluded_ids = excluded
        st.session_state.swiped_titles = swiped_titles
        st.session_state.swipe_count = int(data.get("swipe_count", len(excluded)))
        st.session_state.onboarded = True
        st.session_state.show_onboarding_modal = False
        return "restored"
    except Exception:
        return "empty"
