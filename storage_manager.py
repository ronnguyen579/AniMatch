import json
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_js_eval import get_local_storage, set_local_storage

STORAGE_KEY = "animatch_user_profile_v1"


def save_profile_to_browser():
    """Lưu trạng thái hiện tại của người dùng xuống localStorage trình duyệt."""
    payload = {
        "persona_mode": st.session_state.persona_mode,
        "liked_ids": [int(x) for x in st.session_state.liked_ids],
        "passed_ids": [int(x) for x in st.session_state.passed_ids],
        "swipe_count": int(st.session_state.swipe_count),
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
    """Đọc dữ liệu từ localStorage khi mới mở/F5 trang và khôi phục lại vector v_user."""
    raw_data = get_local_storage(STORAGE_KEY)
    if not raw_data:
        return False

    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        liked_ids = [int(x) for x in data.get("liked_ids", [])]
        passed_ids = [int(x) for x in data.get("passed_ids", [])]

        if not liked_ids and not passed_ids:
            return False

        # Tái tạo lại vector v_user từ lịch sử Like/Pass
        v_rebuilt = np.zeros(expected_dim)
        valid_ids = set(df["anime_id"].astype(int).tolist())
        excluded = set()
        swiped_titles = []

        # Khôi phục các lượt Pass trước (với lực trừ nhẹ nếu chưa có Like)
        for idx, aid in enumerate(passed_ids):
            if aid in valid_ids and aid in anime_vectors:
                eff_beta = (beta * 0.4) if not liked_ids else beta
                v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "pass", gamma, alpha, eff_beta)
                excluded.add(aid)
                swiped_titles.append(str(df[df["anime_id"] == aid].iloc[0]["title"]))

        # Khôi phục các lượt Like
        for aid in liked_ids:
            if aid in valid_ids and aid in anime_vectors:
                v_rebuilt = update_fn(v_rebuilt, anime_vectors[aid], "like", gamma, alpha, beta)
                excluded.add(aid)
                swiped_titles.append(str(df[df["anime_id"] == aid].iloc[0]["title"]))

        st.session_state.persona_mode = data.get("persona_mode", "Restored Session")
        st.session_state.v_user = v_rebuilt
        st.session_state.liked_ids = [aid for aid in liked_ids if aid in valid_ids]
        st.session_state.passed_ids = [aid for aid in passed_ids if aid in valid_ids]
        st.session_state.excluded_ids = excluded
        st.session_state.swiped_titles = swiped_titles
        st.session_state.swipe_count = int(data.get("swipe_count", len(excluded)))
        st.session_state.onboarded = True
        st.session_state.show_onboarding_modal = False
        return True
    except Exception:
        return False


def clear_browser_profile():
    """Xóa sạch dữ liệu trong localStorage khi người dùng bấm Reset."""
    set_local_storage(STORAGE_KEY, "")
