import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from data_loader import clean_pipe_text, get_main_studio, same_franchise

# Trọng số cho 3 không gian con
W_G = 2.5
W_S = 0.8
W_V = 0.35


class SparseVectorLookup:
    """Tra cứu vector mật độ dày theo yêu cầu O(1) từ ma trận thưa V_i, tiết kiệm RAM và thời gian khởi động."""
    def __init__(self, V_i: csr_matrix, id_to_idx: dict):
        self.V_i = V_i
        self.id_to_idx = id_to_idx

    def __contains__(self, anime_id: int) -> bool:
        return int(anime_id) in self.id_to_idx

    def __getitem__(self, anime_id: int) -> np.ndarray:
        idx = self.id_to_idx[int(anime_id)]
        return self.V_i.getrow(idx).toarray()[0]


# Xây dựng ma trận vector TF-IDF cho 3 không gian con (Genres, Synopsis, Studio)
@st.cache_resource(show_spinner=False)
def build_multimodal_space(df: pd.DataFrame):
    tfidf_genres = TfidfVectorizer(stop_words="english")
    tfidf_studio = TfidfVectorizer()

    min_df_value = 4 if len(df) >= 4 else 1
    tfidf_synopsis = TfidfVectorizer(stop_words="english", min_df=min_df_value, max_df=0.4)

    combined_genres_themes = (
        clean_pipe_text(df["genres"]) + " " + clean_pipe_text(df["themes"])
    ).str.strip()
    cleaned_studio = clean_pipe_text(df["studio"])

    V_genres = tfidf_genres.fit_transform(combined_genres_themes) * W_G
    try:
        V_synopsis = tfidf_synopsis.fit_transform(df["synopsis"]) * W_S
    except ValueError:
        tfidf_synopsis = TfidfVectorizer(stop_words="english", min_df=1)
        V_synopsis = tfidf_synopsis.fit_transform(df["synopsis"]) * W_S

    V_studio = tfidf_studio.fit_transform(cleaned_studio) * W_V
    V_i = hstack([V_genres, V_synopsis, V_studio]).tocsr()

    id_to_idx = {int(row["anime_id"]): idx for idx, row in df.iterrows()}

    anime_vectors_dict = {
        int(row["anime_id"]): V_i.getrow(idx).toarray()[0]
        for idx, row in df.iterrows()
    }

    vectorizers = {
        "genres": tfidf_genres,
        "synopsis": tfidf_synopsis,
        "studio": tfidf_studio
    }
    dims = {
        "genres": V_genres.shape[1],
        "synopsis": V_synopsis.shape[1],
        "studio": V_studio.shape[1]
    }
    return V_i, id_to_idx, anime_vectors_dict, vectorizers, dims


# Bộ khởi tạo vector sở thích ban đầu cho người dùng mới (Cold-start)
class ContextualPriorSeeder:
    def __init__(self, anime_vectors, genre_vectorizer, w_g=2.5, dim_sypnosis=0, dim_studio=0):
        self.anime_vectors = anime_vectors
        self.genre_vectorizer = genre_vectorizer
        self.w_g = w_g
        self.dim_sypnosis = dim_sypnosis
        self.dim_studio = dim_studio

    def _encode_genres_to_full_space(self, selected_genres: list[str]) -> np.ndarray:
        if not selected_genres:
            total_dim = len(self.genre_vectorizer.get_feature_names_out()) + self.dim_sypnosis + self.dim_studio
            return np.zeros(total_dim)
        cleaned_genres = [g.replace("|", " ").replace(",", " ") for g in selected_genres]
        genre_vecs = self.genre_vectorizer.transform(cleaned_genres).toarray()
        v_g_mean = np.mean(genre_vecs, axis=0) * self.w_g
        return np.concatenate([v_g_mean, np.zeros(self.dim_sypnosis), np.zeros(self.dim_studio)])

    def seed_cold_start(self, mode: str, selected_items: list) -> tuple[np.ndarray, set]:
        if mode == "content_first":
            return self._encode_genres_to_full_space(selected_items), set()
        elif mode == "eyecandy_first":
            picked_vecs = [self.anime_vectors[aid] for aid in selected_items if aid in self.anime_vectors]
            if picked_vecs:
                v_user_0 = np.mean(picked_vecs, axis=0)
            else:
                total_dim = len(self.genre_vectorizer.get_feature_names_out()) + self.dim_sypnosis + self.dim_studio
                v_user_0 = np.zeros(total_dim)
            return v_user_0, set(selected_items)
        else:
            raise ValueError("Invalid onboarding mode!")


# Bộ khởi tạo vector cho người dùng cũ quay lại (Lapsed User)
class LapsedUserPriorSeeder(ContextualPriorSeeder):
    def seed_lapsed_user(
        self,
        past_anime_ids: list,
        new_genres: list[str],
        lambda_1: float = 0.4,
        lambda_2: float = 0.6
    ) -> tuple[np.ndarray, set]:
        past_vecs = [self.anime_vectors[aid] for aid in past_anime_ids if aid in self.anime_vectors]
        mean_v_new = self._encode_genres_to_full_space(new_genres)
        mean_v_past = np.mean(past_vecs, axis=0) if past_vecs else np.zeros_like(mean_v_new)
        v_user_0 = (lambda_1 * mean_v_past) + (lambda_2 * mean_v_new)
        return v_user_0, set(past_anime_ids)


# Lọc danh sách ứng viên: loại bỏ trùng franchise và giới hạn số phim mỗi studio
def deduplicate(recommendations: pd.DataFrame, max_per_studio: int = 2, threshold: int = 90, seen_titles: list[str] = None) -> pd.DataFrame:
    selected = []
    selected_titles = list(seen_titles) if seen_titles else []
    studio_counts = {}

    for _, anime in recommendations.iterrows():
        title = anime["title"]
        studio = get_main_studio(anime["studios"])

        if any(same_franchise(title, s_title, threshold) for s_title in selected_titles):
            continue

        if studio != "Unknown" and studio_counts.get(studio, 0) >= max_per_studio:
            continue

        selected.append(anime)
        selected_titles.append(title)
        if studio != "Unknown":
            studio_counts[studio] = studio_counts.get(studio, 0) + 1

    return pd.DataFrame(selected)


# Cắt vector tổng hợp thành 3 vector con riêng biệt (Genres, Synopsis, Studio)
def split_full_vector_to_subspaces(full_vec: np.ndarray, dims: dict) -> dict:
    g_end = dims["genres"]
    s_end = g_end + dims["synopsis"]
    return {
        "genres": csr_matrix(full_vec[:g_end]),
        "synopsis": csr_matrix(full_vec[g_end:s_end]),
        "studio": csr_matrix(full_vec[s_end:])
    }


# Tìm top từ khóa đóng góp điểm tương đồng dương và âm lớn nhất
def explain_recommendation(user_vectors: dict, anime_vectors: dict, vectorizers: dict, top_n: int = 3) -> dict:
    weights = {"genres": 2.5, "synopsis": 0.8, "studio": 0.35}
    explanations = {"positive": {}, "negative": {}}

    for feature, weight in weights.items():
        contributions = (user_vectors[feature].multiply(anime_vectors[feature])).toarray()[0] * weight
        terms = vectorizers[feature].get_feature_names_out()

        pos_idx = np.argsort(contributions)[::-1]
        explanations["positive"][feature] = [
            (terms[i], float(contributions[i])) for i in pos_idx if contributions[i] > 0
        ][:top_n]

        neg_idx = np.argsort(contributions)
        explanations["negative"][feature] = [
            (terms[i], float(contributions[i])) for i in neg_idx if contributions[i] < 0
        ][:top_n]

    return explanations


# Chuyển kết quả phân tích từ khóa thành câu giải thích tự nhiên
def explanation_to_text(explanations: dict) -> str:
    positive = explanations["positive"]
    negative = explanations["negative"]
    reasons = []

    genres = [name for name, _ in positive["genres"]]
    if genres:
        reasons.append("it matches your interest in " + ", ".join(genres))

    synopsis = [name for name, _ in positive["synopsis"]]
    if synopsis:
        reasons.append("its storyline includes elements related to " + ", ".join(synopsis))

    studios = [name for name, _ in positive["studio"]]
    if studios:
        reasons.append("it features studios you tend to prefer, such as " + ", ".join(studios))

    neg_feats = []
    for feat in ["genres", "synopsis", "studio"]:
        neg_feats.extend([name for name, _ in negative[feat]])
    if neg_feats:
        reasons.append("note: it contains some elements you previously passed on such as " + ", ".join(neg_feats))

    if not reasons:
        return "Recommended to explore high-rated visual and studio diversity."
    return "Recommended because " + ", and ".join(reasons) + "."


# So sánh song song 2 bộ anime bất kỳ
def compare_anime(df: pd.DataFrame, anime_id_1: int, anime_id_2: int) -> dict:
    anime_1 = df[df["mal_id"] == anime_id_1].iloc[0]
    anime_2 = df[df["mal_id"] == anime_id_2].iloc[0]

    attributes = {
        "Title": "title", "Type": "type", "Genres": "genres", "Themes": "themes",
        "Synopsis": "synopsis", "Studio": "studios", "Score": "score",
        "Episodes": "episodes", "Year": "year", "Image": "image_url"
    }
    comparison = {}
    for label, column in attributes.items():
        val_a, val_b = anime_1[column], anime_2[column]
        comparison[label] = {"anime_a": val_a, "anime_b": val_b}
        if label in ["Genres", "Themes", "Studio"]:
            set_a = set([x.strip() for x in re.split(r"[|,;]", str(val_a)) if x.strip()]) if pd.notna(val_a) else set()
            set_b = set([x.strip() for x in re.split(r"[|,;]", str(val_b)) if x.strip()]) if pd.notna(val_b) else set()
            comparison[label]["common"] = sorted(set_a & set_b)
            comparison[label]["different_a"] = sorted(set_a - set_b)
            comparison[label]["different_b"] = sorted(set_b - set_a)
    return comparison


# Cập nhật vector sở thích người dùng bằng Online Rocchio (có suy giảm trí nhớ gamma)
def update_user_vector_rocchio(v_user: np.ndarray, v_anime: np.ndarray, action: str, gamma: float = 0.95, alpha: float = 0.30, beta: float = 0.15) -> np.ndarray:
    delta_v = (alpha * v_anime) if action == "like" else (-beta * v_anime)
    return (gamma * v_user) + delta_v


# Tạo bộ thẻ gợi ý kế tiếp bằng Quality Gating + Deduplication + MMR (tối ưu ma trận thưa)
def generate_diverse_deck(
    df: pd.DataFrame,
    anime_vectors,
    v_user: np.ndarray,
    excluded_ids: set,
    swiped_titles: list[str],
    deck_size: int = 8,
    mmr_lambda: float = 0.75,
    max_per_studio: int = 2,
    franchise_threshold: int = 90,
    min_score: float = 7.2,
    quality_weight: float = 0.25,
    V_i=None,
    id_to_idx: dict = None
) -> list[int]:
    mask = ~df["anime_id"].isin(excluded_ids).values
    if not np.any(mask):
        return []

    cand_df = df.loc[mask, ["anime_id", "title", "studios", "score"]].copy()
    num_scores = pd.to_numeric(cand_df["score"], errors="coerce").fillna(6.5).values
    cand_df["num_score"] = num_scores

    hq_mask = num_scores >= min_score
    if np.sum(hq_mask) >= deck_size:
        cand_df = cand_df[hq_mask]
    elif np.sum(hq_mask) > 0:
        rel_mask = num_scores >= max(min_score - 1.0, 0.0)
        if np.any(rel_mask):
            cand_df = cand_df[rel_mask]

    cand_ids = cand_df["anime_id"].tolist()

    if V_i is not None and id_to_idx is not None:
        cand_matrix = V_i[[id_to_idx[aid] for aid in cand_ids]]
    else:
        cand_matrix = np.vstack([anime_vectors[aid] for aid in cand_ids])

    norm_quality = np.clip((cand_df["num_score"].values - 5.0) / 4.5, 0.0, 1.0)

    if np.linalg.norm(v_user) > 1e-9:
        raw_cos = cosine_similarity(cand_matrix, v_user.reshape(1, -1)).flatten()
        norm_cos = (raw_cos + 1.0) / 2.0
        combined_rel = ((1.0 - quality_weight) * norm_cos) + (quality_weight * norm_quality)
    else:
        combined_rel = norm_quality

    cand_df["sim_score"] = combined_rel
    cand_df = cand_df.sort_values(by="sim_score", ascending=False)

    top_pool = cand_df.head(min(len(cand_df), max(deck_size * 5, 35)))
    gated_df = deduplicate(
        top_pool,
        max_per_studio=max_per_studio,
        threshold=franchise_threshold,
        seen_titles=swiped_titles
    )
    if gated_df.empty:
        gated_df = cand_df.head(deck_size)

    gated_ids = gated_df["anime_id"].tolist()
    n_gated = len(gated_ids)
    if n_gated <= 1:
        return gated_ids

    if V_i is not None and id_to_idx is not None:
        pairwise_sim = cosine_similarity(V_i[[id_to_idx[aid] for aid in gated_ids]])
    else:
        pairwise_sim = cosine_similarity(np.vstack([anime_vectors[aid] for aid in gated_ids]))

    user_sims = gated_df["sim_score"].values
    selected_pos = [0]
    remaining_pos = list(range(1, n_gated))

    while remaining_pos and len(selected_pos) < deck_size:
        max_sim_to_selected = np.max(pairwise_sim[np.ix_(remaining_pos, selected_pos)], axis=1)
        mmr_scores = (mmr_lambda * user_sims[remaining_pos]) - ((1.0 - mmr_lambda) * max_sim_to_selected)
        best_local_idx = int(np.argmax(mmr_scores))
        chosen_pos = remaining_pos.pop(best_local_idx)
        selected_pos.append(chosen_pos)

    return [gated_ids[p] for p in selected_pos]
