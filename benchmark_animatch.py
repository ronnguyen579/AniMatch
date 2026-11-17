import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. TẢI DỮ LIỆU & XÂY DỰNG 3 SUBSPACE VECTORS
# ==========================================
df = pd.read_csv("anime_cleaned.csv").dropna(subset=["title", "genres", "synopsis"]).reset_index(drop=True)
df["studios"] = df["studios"].fillna("Unknown")

vec_genre = TfidfVectorizer(max_features=1500, stop_words="english")
vec_syn = TfidfVectorizer(max_features=3000, stop_words="english")
vec_stu = TfidfVectorizer(max_features=500)

M_genre = vec_genre.fit_transform(df["genres"].astype(str)).toarray() * 2.5
M_syn = vec_syn.fit_transform(df["synopsis"].astype(str)).toarray() * 0.8
M_stu = vec_stu.fit_transform(df["studios"].astype(str)).toarray() * 0.35

# Ghép 3 không gian con và chuẩn hóa L2
V_anime = np.hstack([M_genre, M_syn, M_stu])
norms = np.linalg.norm(V_anime, axis=1, keepdims=True) + 1e-9
V_anime = V_anime / norms


# ==========================================
# 2. HÀM XẾP HẠNG: BASELINE COSINE VS ANIMATCH MMR
# ==========================================
def rank_pure_cosine(v_user, seen_idx, top_k=8):
    scores = V_anime @ v_user
    scores[list(seen_idx)] = -1e9
    return np.argsort(scores)[::-1][:top_k]


def rank_animatch_mmr(v_user, seen_idx, top_k=8, lambda_div=0.72, studio_cap=2):
    rel_scores = V_anime @ v_user
    candidate_pool = np.argsort(rel_scores)[::-1]

    selected = []
    studio_counts = {}
    seen_prefixes = {df.loc[i, "title"][:12].lower() for i in seen_idx}

    for idx in candidate_pool:
        if idx in seen_idx or len(selected) >= top_k * 4:
            continue
        prefix = df.loc[idx, "title"][:12].lower()
        studio = str(df.loc[idx, "studios"]).split(",")[0].strip()

        # Lọc trùng franchise & giới hạn studio
        if prefix in seen_prefixes:
            continue
        if studio_counts.get(studio, 0) >= studio_cap:
            continue

        selected.append(idx)
        seen_prefixes.add(prefix)
        studio_counts[studio] = studio_counts.get(studio, 0) + 1

    # Maximal Marginal Relevance (MMR)
    final_deck = []
    while len(final_deck) < top_k and selected:
        best_item, best_mmr = None, -1e9
        for cand in selected:
            rel = rel_scores[cand]
            if not final_deck:
                div_penalty = 0.0
            else:
                div_penalty = np.max(V_anime[final_deck] @ V_anime[cand])
            mmr = lambda_div * rel - (1 - lambda_div) * div_penalty
            if mmr > best_mmr:
                best_mmr = mmr
                best_item = cand
        final_deck.append(best_item)
        selected.remove(best_item)

    return np.array(final_deck)


def intra_list_diversity(indices):
    if len(indices) < 2:
        return 0.0
    sub_V = V_anime[indices]
    sim_matrix = sub_V @ sub_V.T
    n = len(indices)
    # Trung bình (1 - cosine_similarity) của các cặp khác nhau
    off_diag_sum = np.sum(1.0 - sim_matrix) - np.trace(1.0 - sim_matrix)
    return off_diag_sum / (n * (n - 1))


# ==========================================
# 3. MÔ PHỎNG NGƯỜI DÙNG & ĐỔI GU ĐỘT NGỘT (MOOD SHIFT)
# ==========================================
def simulate_session(steps=20, gamma=0.95, alpha=1.0, beta=0.6):
    # Persona A (10 lượt đầu): Action / Dark Fantasy
    mask_A = df["genres"].str.contains("Action", case=False) & df["genres"].str.contains("Fantasy", case=False)
    v_target_A = V_anime[mask_A].mean(axis=0)
    v_target_A /= np.linalg.norm(v_target_A)

    # Persona B (10 lượt sau - Đổi gu đột ngột): Slice of Life / Comedy
    mask_B = df["genres"].str.contains("Slice of Life", case=False) & df["genres"].str.contains("Comedy", case=False)
    v_target_B = V_anime[mask_B].mean(axis=0)
    v_target_B /= np.linalg.norm(v_target_B)

    v_user = np.random.normal(0, 0.01, size=V_anime.shape[1])
    v_user /= np.linalg.norm(v_user)

    seen = set()
    metrics = {
        "step": [], "target_align": [],
        "prec_cosine": [], "prec_animatch": [],
        "ild_cosine": [], "ild_animatch": [],
        "latency_ms": []
    }

    for t in range(1, steps + 1):
        v_true = v_target_A if t <= 10 else v_target_B

        t0 = time.perf_counter()
        deck_cos = rank_pure_cosine(v_user, seen, top_k=8)
        deck_mmr = rank_animatch_mmr(v_user, seen, top_k=8)

        # Thẻ đầu tiên của AniMatch được đưa cho user quyết định Like/Pass
        top_card = deck_mmr[0]
        seen.add(top_card)
        true_affinity = float(V_anime[top_card] @ v_true)

        # Ngưỡng Like giả lập
        if true_affinity >= 0.22:
            v_user = gamma * v_user + alpha * V_anime[top_card]
        else:
            v_user = gamma * v_user - beta * V_anime[top_card]
        v_user /= (np.linalg.norm(v_user) + 1e-9)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Đo Precision@8 (tỷ lệ thẻ trong deck có affinity >= 0.22 với gu hiện tại)
        prec_cos = np.mean((V_anime[deck_cos] @ v_true) >= 0.22)
        prec_mmr = np.mean((V_anime[deck_mmr] @ v_true) >= 0.22)

        metrics["step"].append(t)
        metrics["target_align"].append(float(v_user @ v_true))
        metrics["prec_cosine"].append(prec_cos)
        metrics["prec_animatch"].append(prec_mmr)
        metrics["ild_cosine"].append(intra_list_diversity(deck_cos))
        metrics["ild_animatch"].append(intra_list_diversity(deck_mmr))
        metrics["latency_ms"].append(dt_ms)

    return pd.DataFrame(metrics)


np.random.seed(42)
res = simulate_session(steps=20, gamma=0.95)

# In bảng tóm tắt ra Terminal
print("\n=== KẾT QUẢ BENCHMARK ANIMATCH (TRUNG BÌNH 20 LƯỢT QUẸT) ===")
print(f"• Precision@8 (Pure Cosine)   : {res['prec_cosine'].mean():.2%}")
print(f"• Precision@8 (AniMatch MMR)  : {res['prec_animatch'].mean():.2%}")
print(f"• Độ đa dạng ILD (Pure Cosine): {res['ild_cosine'].mean():.3f}")
print(f"• Độ đa dạng ILD (AniMatch)   : {res['ild_animatch'].mean():.3f}  (Càng cao càng ít trùng lặp)")
print(f"• Độ trễ CPU trung bình (p50) : {res['latency_ms'].median():.2f} ms")
print(f"• Độ trễ CPU tối đa (p99)     : {np.percentile(res['latency_ms'], 99):.2f} ms")


# ==========================================
# 4. VẼ 4 BIỂU ĐỒ TRỰC QUAN
# ==========================================
plt.style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle("AniMatch Offline Evaluation: Steering, Diversity, Mood-Shift & CPU Latency", fontsize=14, fontweight="bold")

# Biểu đồ 1: Độ khớp sở thích (Precision@8) & Khả năng bẻ lái khi đổi gu ở lượt 10
ax = axes[0, 0]
ax.plot(res["step"], res["prec_cosine"], "--o", color="#94a3b8", label="Baseline (Pure Cosine)")
ax.plot(res["step"], res["prec_animatch"], "-s", color="#10b981", linewidth=2.2, label="AniMatch (Rocchio + MMR)")
ax.axvline(10.5, color="#ef4444", linestyle=":", linewidth=1.8, label="Mood Shift (Action -> Slice of Life)")
ax.set_title("1. Precision@8 & Adaptation After Sudden Mood Shift")
ax.set_xlabel("Swipe Step (t)")
ax.set_ylabel("Precision@8")
ax.set_ylim(0, 1.08)
ax.legend(loc="lower right", fontsize=8.5)

# Biểu đồ 2: Độ đa dạng chống buồng vang (Intra-List Diversity)
ax = axes[0, 1]
ax.plot(res["step"], res["ild_cosine"], "--o", color="#94a3b8", label="Baseline (Echo Chamber Risk)")
ax.plot(res["step"], res["ild_animatch"], "-s", color="#3b82f6", linewidth=2.2, label="AniMatch (MMR + Studio/Series Cap)")
ax.set_title("2. Intra-List Diversity (Higher = Less Sequel Repetition)")
ax.set_xlabel("Swipe Step (t)")
ax.set_ylabel("ILD (1 - Avg Pairwise Cosine)")
ax.legend(loc="lower right", fontsize=8.5)

# Biểu đồ 3: Tốc độ hội tụ Vector người dùng (Cosine Alignment với Ground Truth)
ax = axes[1, 0]
ax.fill_between(res["step"], res["target_align"], color="#8b5cf6", alpha=0.25)
ax.plot(res["step"], res["target_align"], "-D", color="#7c3aed", linewidth=2)
ax.axvline(10.5, color="#ef4444", linestyle=":", linewidth=1.8)
ax.set_title("3. User Vector Alignment with Hidden Ground-Truth Taste")
ax.set_xlabel("Swipe Step (t)")
ax.set_ylabel("Cosine(v_user, v_target)")

# Biểu đồ 4: Phân bố độ trễ CPU (Latency ms)
ax = axes[1, 1]
ax.bar(res["step"], res["latency_ms"], color="#f59e0b", alpha=0.85, label="Step Latency (ms)")
ax.axhline(10.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Real-Time Budget (10 ms)")
ax.set_title("4. CPU Inference & Re-ranking Latency per Swipe")
ax.set_xlabel("Swipe Step (t)")
ax.set_ylabel("Latency (ms)")
ax.legend(loc="upper right", fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("animatch_benchmark_report.png", dpi=200)
print("\n📊 Đã xuất biểu đồ trực quan tại: animatch_benchmark_report.png")
plt.show()
