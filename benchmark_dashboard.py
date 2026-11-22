import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Import các hàm lõi từ project AniMatch của bạn
from recommender_engine import generate_diverse_deck, get_franchise_root

try:
    from data_loader import enrich_franchise_metadata
except ImportError:
    enrich_franchise_metadata = None

st.set_page_config(
    page_title="AniMatch — Honest Evaluation Benchmark",
    page_icon="📊",
    layout="wide",
)


# =============================================================================
# 1. NẠP DỮ LIỆU & DỰNG MA TRẬN VECTOR (CÓ CACHE)
# =============================================================================
@st.cache_data(show_spinner="Đang nạp dữ liệu catalog và dựng ma trận vector...")
def load_benchmark_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Ưu tiên file đã enrich đồ thị AniList
    enriched_files = sorted(glob.glob(os.path.join(current_dir, "*_graph_enriched.csv")))
    if enriched_files:
        csv_path = enriched_files[0]
    else:
        csv_files = sorted([
            p for p in glob.glob(os.path.join(current_dir, "*.csv"))
            if "jikan_tv_movie_cache" not in os.path.basename(p)
        ])
        if not csv_files:
            raise FileNotFoundError("Không tìm thấy file CSV dữ liệu anime trong thư mục project!")
        csv_path = csv_files[0]

    df = pd.read_csv(csv_path)

    # Chuẩn hóa tên cột ID
    if "anime_id" not in df.columns and "mal_id" in df.columns:
        df["anime_id"] = df["mal_id"]
    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce").fillna(0).astype(int)
    df = df[df["anime_id"] > 0].drop_duplicates(subset=["anime_id"]).reset_index(drop=True)

    # Đảm bảo các cột cơ bản tồn tại
    for col, default_val in [
        ("title", "Unknown"),
        ("genres", ""),
        ("studios", ""),
        ("synopsis", ""),
        ("type", "TV"),
        ("score", 7.0),
        ("year", 2020),
    ]:
        if col not in df.columns:
            df[col] = default_val

    if enrich_franchise_metadata is not None:
        df = enrich_franchise_metadata(df)
    else:
        if "franchise_role" not in df.columns:
            df["franchise_role"] = "ENTRY"
        if "entry_anime_id" not in df.columns:
            df["entry_anime_id"] = df["anime_id"]
        if "prereq_anime_id" not in df.columns:
            df["prereq_anime_id"] = 0

    # Dựng TF-IDF kết hợp genres (nhấn mạnh trọng số) + studios + synopsis
    corpus = (
        (df["genres"].fillna("").astype(str) + " ") * 3
        + (df["studios"].fillna("").astype(str) + " ") * 2
        + df["synopsis"].fillna("").astype(str)
    )
    vectorizer = TfidfVectorizer(max_features=2500, stop_words="english")
    V_i = vectorizer.fit_transform(corpus).toarray().astype(np.float32)

    # Chuẩn hóa L2 từng dòng để tính Cosine nhanh và ổn định
    norms = np.linalg.norm(V_i, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V_i = V_i / norms

    id_to_idx = {int(aid): idx for idx, aid in enumerate(df["anime_id"].astype(int))}
    anime_vectors = {int(aid): V_i[idx] for aid, idx in id_to_idx.items()}
    anime_lookup = df.set_index("anime_id").to_dict(orient="index")

    return df, V_i, id_to_idx, anime_vectors, anime_lookup, os.path.basename(csv_path)


# =============================================================================
# 2. HÀM PHỤ TRỢ: DỰNG VECTOR GU MỤC TIÊU & TÍNH ĐA DẠNG
# =============================================================================
def build_persona_vector(df: pd.DataFrame, V_i: np.ndarray, keywords: list[str]) -> np.ndarray:
    mask = df["genres"].fillna("").astype(str).apply(
        lambda g: any(k.lower() in g.lower() for k in keywords)
    )
    idxs = np.where(mask.values)[0]
    if len(idxs) == 0:
        vec = np.mean(V_i, axis=0)
    else:
        vec = np.mean(V_i[idxs], axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def calc_ild(deck_ids: list[int], V_i: np.ndarray, id_to_idx: dict) -> float:
    if len(deck_ids) <= 1:
        return 0.0
    vecs = V_i[[id_to_idx[aid] for aid in deck_ids]]
    pw = cosine_similarity(vecs)
    triu = pw[np.triu_indices(len(deck_ids), k=1)]
    return float(1.0 - np.mean(triu))


def get_franchise_key(row: dict, aid: int) -> str:
    e_id = int(row.get("entry_anime_id", 0) or 0)
    if e_id > 0:
        return f"entry_{e_id}"
    return f"root_{get_franchise_root(str(row.get('title', '')))}"


# =============================================================================
# 3. ENGINE MÔ PHỎNG ĐỐI CHỨNG 3 HỆ THỐNG (25 BƯỚC + MOOD SHIFT)
# =============================================================================
def run_full_mvp_benchmark(
    df: pd.DataFrame,
    V_i: np.ndarray,
    id_to_idx: dict,
    anime_vectors: dict,
    anime_lookup: dict
