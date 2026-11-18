import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

from data_loader import (
    clean_pipe_text,
    get_main_studio,
    same_franchise,
    extract_season_number,
    get_franchise_root
)
# Trọng số cho 3 không gian con
W_G = 2.5
W_S = 0.8
W_V = 0.35

ANIME_GENERIC_WORDS = {
    "world", "life", "city", "girl", "girls", "boy", "boys", "young", "story",
    "new", "day", "days", "time", "year", "years", "make", "way", "people",
    "human", "humans", "man", "woman", "one", "two", "however", "despite",
    "begin", "begins", "find", "finds", "must", "becomes", "become"
}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS | ANIME_GENERIC_WORDS)


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
    tfidf_synopsis = TfidfVectorizer(
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
        min_df=min_df_value,
        max_df=0.35,
        max_features=3500
    )

    combined_genres_themes = (
        clean_pipe_text(df["genres"]) + " " + clean_pipe_text(df["themes"])
    ).str.strip()
    cleaned_studio = clean_pipe_text(df["studio"])

    V_genres = tfidf_genres.fit_transform(combined_genres_themes) * W_G
    try:
        V_synopsis = tfidf_synopsis.fit_transform(df["synopsis"]) * W_S
    except ValueError:
        tfidf_synopsis = TfidfVectorizer(
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=(1, 2),
            min_df=1,
            max_features=3500
        )
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
def update_user_vector_rocchio(
    v_user: np.ndarray,
    v_anime: np.ndarray,
    action: str,
    gamma: float = 0.95,
    alpha: float = 0.30,
    beta: float = 0.15
) -> np.ndarray:
    if action == "like":
        delta_v = alpha * v_anime
    elif action == "watched":
        delta_v = (0.7 * alpha) * v_anime
    else:
        delta_v = -beta * v_anime
    return (gamma * v_user) + delta_v


# Tạo bộ thẻ gợi ý kế tiếp bằng Quality Gating + Deduplication + MMR (tối ưu ma trận thưa)
def generate_diverse_deck(
    df: pd.DataFrame,
    anime_vectors: dict,
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
    id_to_idx: dict = None,
    watched_ids: list[int] | set[int] = None
) -> list[int]:
    """
    Sinh ra một bộ bài (deck) gồm `deck_size` anime tối ưu hóa giữa:
    1. Độ hợp gu cá nhân (Cosine Similarity với vector người dùng v_user).
    2. Chất lượng phim (Điểm đánh giá MAL/AniList).
    3. Entry-Point Gatekeeper: Chặn tuyệt đối Movie hậu truyện / Season sau xuất hiện trước phần gốc.
    4. Franchise Cooldown & Pacing: Ngăn việc bội thực liên tiếp cùng 1 series sau khi user bấm Like.
    5. MMR (Maximal Marginal Relevance) + Studio Cap: Ép đa dạng thể loại và xưởng phim trong cùng 1 deck.
    """

    # =========================================================================
    # BƯỚC 1: LỌC LOẠI TRỪ & LỌC CHẤT LƯỢNG (QUALITY FILTERING)
    # Tác dụng: Loại bỏ các phim user đã quẹt/xem/ẩn và lọc ra các phim đạt ngưỡng điểm tối thiểu.
    # =========================================================================
    mask = ~df["anime_id"].isin(excluded_ids).values
    if not np.any(mask):
        return []

    base_cols = ["anime_id", "title", "studios", "score", "year", "type"]
    for extra_col in ["franchise_role", "entry_anime_id", "prefix_root"]:
        if extra_col in df.columns:
            base_cols.append(extra_col)

    cand_df = df.loc[mask, base_cols].copy()
    if "franchise_role" not in cand_df.columns:
        cand_df["franchise_role"] = "ENTRY"
    if "entry_anime_id" not in cand_df.columns:
        cand_df["entry_anime_id"] = cand_df["anime_id"]

    num_scores = pd.to_numeric(cand_df["score"], errors="coerce").fillna(6.5).values
    cand_df["num_score"] = num_scores
    cand_df["num_year"] = pd.to_numeric(cand_df["year"], errors="coerce").fillna(9999).values

    # Ưu tiên lấy các phim có điểm >= min_score; nếu thiếu bài thì nới lỏng xuống (min_score - 1.0)
    hq_mask = num_scores >= min_score
    if np.sum(hq_mask) >= deck_size:
        cand_df = cand_df[hq_mask]
    elif np.sum(hq_mask) > 0:
        rel_mask = num_scores >= max(min_score - 1.0, 0.0)
        if np.any(rel_mask):
            cand_df = cand_df[rel_mask]

    cand_ids = cand_df["anime_id"].tolist()

    # =========================================================================
    # BƯỚC 2: TÍNH ĐIỂM PHÙ HỢP CÁ NHÂN (RELEVANCE & COLD-START SCORING)
    # Tác dụng: Kết hợp độ tương đồng vector (v_user) với điểm chất lượng phim.
    # Nếu user mới tinh (v_user == 0), trộn thêm nhiễu ngẫu nhiên có trọng số để mỗi lần mở app
    # nhận được bộ thẻ mở đầu khác nhau.
    # =========================================================================
    if V_i is not None and id_to_idx is not None:
        cand_matrix = V_i[[id_to_idx[aid] for aid in cand_ids]]
    else:
        cand_matrix = np.vstack([anime_vectors[aid] for aid in cand_ids])

    norm_quality = np.clip((cand_df["num_score"].values - 5.0) / 4.5, 0.0, 1.0)

    if np.linalg.norm(v_user) > 1e-9:
        raw_cos = cosine_similarity(cand_matrix, v_user.reshape(1, -1)).flatten()
        norm_cos = (raw_cos + 1.0) / 2.0
        combined_rel = ((1.0 - quality_weight) * norm_cos) + (quality_weight * norm_quality)
        cand_df["sim_score"] = combined_rel
    else:
        rng = np.random.default_rng()
        jitter = rng.uniform(0.0, 0.35, size=len(cand_df))
        cand_df["sim_score"] = (0.65 * norm_quality) + jitter

    # =========================================================================
    # BƯỚC 3: KIỂM SOÁT NHỊP ĐỘ FRANCHISE (FRANCHISE COOLDOWN & PACING)
    # Tác dụng: Ngăn hiện tượng "hiệu ứng đường hầm" — khi user vừa Like phần 1, hệ thống
    # tạm khóa các phần tiếp theo trong 4 lượt quẹt và giảm nhẹ trọng số Sequel (0.88x)
    # để user có không gian đổi sang series/thể loại mới trước khi cày tiếp phần sau.
    # =========================================================================
    recent_titles = swiped_titles[-4:] if swiped_titles else []
    recent_roots = {get_franchise_root(t) for t in recent_titles}
    if "prefix_root" in df.columns and recent_titles:
        recent_prefixes = set(
            df.loc[df["title"].isin(set(recent_titles)), "prefix_root"].dropna().astype(str)
        )
    else:
        recent_prefixes = set()

    is_seq_mask = cand_df["franchise_role"].astype(str).str.upper().isin(["SEQUEL", "PREQUEL", "SPIN_OFF"])
    cand_df.loc[is_seq_mask, "sim_score"] = cand_df.loc[is_seq_mask, "sim_score"] * 0.88

    def _is_in_franchise_cooldown(row) -> bool:
        r_root = get_franchise_root(row["title"])
        r_pref = str(row.get("prefix_root", r_root))
        return (r_root in recent_roots) or (len(r_pref) >= 4 and r_pref in recent_prefixes)

    non_cooldown_df = cand_df[~cand_df.apply(_is_in_franchise_cooldown, axis=1)]
    if len(non_cooldown_df) >= deck_size * 2:
        cand_df = non_cooldown_df

    cand_df = cand_df.sort_values(by="sim_score", ascending=False)

    # =========================================================================
    # BƯỚC 4: ENTRY-POINT GATEKEEPER (CHẶN NHẢY CÓC PHẦN SAU / MOVIE ĂN THEO)
    # Tác dụng: Kiểm tra từng ứng viên top đầu; nếu là Sequel, Prequel, Spin-off hoặc
    # Movie tổng kết (như Gintama: The Final) mà user chưa từng tương tác với phần gốc:
    # - Nếu phần gốc hợp lệ -> tự động tráo (redirect) thẻ đó thành phần gốc (TV Season 1).
    # - Nếu phần gốc không hợp lệ (VD: user đang bật lọc chỉ xem Movie) -> loại bỏ luôn.
    # =========================================================================
    # Chỉ mở khóa phần tiếp theo (Sequel/Prequel/Spin-off/Movie) khi phần gốc nằm trong WATCHED_IDS!
    watched_set = {int(x) for x in (watched_ids or [])}
    watched_titles = set(df.loc[df["anime_id"].isin(watched_set), "title"].astype(str))
    watched_roots = {get_franchise_root(t) for t in watched_titles}
    if "prefix_root" in df.columns and watched_set:
        watched_prefixes = set(
            df.loc[df["anime_id"].isin(watched_set), "prefix_root"].dropna().astype(str)
        )
    else:
        watched_prefixes = set()

    cand_row_by_id = {int(r["anime_id"]): r for _, r in cand_df.iterrows()}
    top_pool = cand_df.head(min(len(cand_df), max(deck_size * 6, 45))).copy()

    resolved_rows = []
    seen_resolved_ids = set()

    for _, row in top_pool.iterrows():
        aid = int(row["anime_id"])
        role = str(row.get("franchise_role", "ENTRY")).upper()
        entry_id = int(row.get("entry_anime_id", aid))
        root = get_franchise_root(row["title"])
        prefix = str(row.get("prefix_root", root))

        is_followup = (
            role in ("SEQUEL", "PREQUEL", "SPIN_OFF")
            or extract_season_number(row["title"]) > 1
            or (entry_id != aid)
        )
        has_watched_origin = (
            (entry_id in watched_set)
            or (root in watched_roots)
            or (len(prefix) >= 4 and prefix in watched_prefixes)
        )

        if is_followup and not has_watched_origin:
            if entry_id != aid and entry_id in cand_row_by_id:
                target_row = cand_row_by_id[entry_id]
                tid = int(target_row["anime_id"])
                if tid not in seen_resolved_ids:
                    seen_resolved_ids.add(tid)
                    resolved_rows.append(target_row)
            continue

        if aid not in seen_resolved_ids:
            seen_resolved_ids.add(aid)
            resolved_rows.append(row)

    # Dự phòng: Nếu toàn bộ top_pool đều là Sequel bị chặn, lấy các bản ENTRY thuần túy trong cand_df
    if not resolved_rows:
        pure_entries = cand_df[
            (cand_df["franchise_role"].astype(str).str.upper() == "ENTRY")
            & (cand_df["entry_anime_id"].astype(int) == cand_df["anime_id"].astype(int))
        ]
        fallback_source = pure_entries if not pure_entries.empty else cand_df
        resolved_rows = [r for _, r in fallback_source.head(deck_size * 2).iterrows()]

    resolved_pool = pd.DataFrame(resolved_rows)
    # =========================================================================
    # BƯỚC 5: KHỬ TRÙNG STUDIO & TÊN GẦN GIỐNG (DEDUPLICATION & STUDIO CAP)
    # Tác dụng: Giới hạn tối đa `max_per_studio` bộ từ cùng 1 xưởng phim trong nhóm ứng viên
    # và loại bỏ các bản trùng tên mờ (fuzzy match >= franchise_threshold).
    # =========================================================================
    gated_df = deduplicate(
        resolved_pool,
        max_per_studio=max_per_studio,
        threshold=franchise_threshold,
        seen_titles=swiped_titles
    )
    if gated_df.empty:
        gated_df = resolved_pool.head(deck_size)

    gated_ids = gated_df["anime_id"].tolist()
    n_gated = len(gated_ids)
    if n_gated <= 1:
        return gated_ids

    # =========================================================================
    # BƯỚC 6: TÁI XẾP HẠNG ĐA DẠNG HÓA MMR (MAXIMAL MARGINAL RELEVANCE)
    # Tác dụng: Chọn lần lượt từng thẻ vào deck sao cho vừa bám sát gu user (mmr_lambda),
    # vừa khác biệt nội dung với các thẻ đã chọn trước đó (1 - mmr_lambda), đồng thời khóa
    # cứng mỗi franchise chỉ xuất hiện tối đa 1 lần duy nhất trong deck 8 thẻ.
    # =========================================================================
    if V_i is not None and id_to_idx is not None:
        pairwise_sim = cosine_similarity(V_i[[id_to_idx[aid] for aid in gated_ids]])
    else:
        pairwise_sim = cosine_similarity(np.vstack([anime_vectors[aid] for aid in gated_ids]))

    user_sims = gated_df["sim_score"].values
    gated_roots = [get_franchise_root(t) for t in gated_df["title"].tolist()]

    selected_pos = [0]
    selected_roots = {gated_roots[0]}
    remaining_pos = list(range(1, n_gated))

    while remaining_pos and len(selected_pos) < deck_size:
        fresh_franchise_pos = [p for p in remaining_pos if gated_roots[p] not in selected_roots]
        pool_to_pick = fresh_franchise_pos if fresh_franchise_pos else remaining_pos

        max_sim_to_selected = np.max(pairwise_sim[np.ix_(pool_to_pick, selected_pos)], axis=1)
        mmr_scores = (mmr_lambda * user_sims[pool_to_pick]) - ((1.0 - mmr_lambda) * max_sim_to_selected)
        best_local_idx = int(np.argmax(mmr_scores))
        chosen_pos = pool_to_pick[best_local_idx]

        remaining_pos.remove(chosen_pos)
        selected_pos.append(chosen_pos)
        selected_roots.add(gated_roots[chosen_pos])

    return [gated_ids[p] for p in selected_pos]
