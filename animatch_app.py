import os
import glob
import re
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

try:
    from rapidfuzz import fuzz
    def calc_fuzz_ratio(s1: str, s2: str) -> float:
        return fuzz.ratio(s1, s2)
except ImportError:
    from difflib import SequenceMatcher
    def calc_fuzz_ratio(s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio() * 100.0


# ==============================================================================
# PAGE CONFIGURATION
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
# 0. FALLBACK SAMPLE DATASET & CSV SCHEMA NORMALIZATION
# ==============================================================================
def get_fallback_dataframe() -> pd.DataFrame:
    data = [
        {
            "anime_id": 1, "mal_id": 1, "title": "Cowboy Bebop", "type": "TV",
            "genres": "Action|Adventure|Sci-Fi", "themes": "Adult Cast|Space",
            "synopsis": "In the year 2071, humanity has colonized several of the planets and moons of the solar system. Bounty hunters Spike Spiegel and Jet Black chase criminals across space on the spaceship Bebop.",
            "studio": "Sunrise", "studios": "Sunrise", "score": 8.75, "episodes": 26, "year": 1998,
            "image_url": "https://cdn.myanimelist.net/images/anime/4/19644.jpg",
            "trailer_url": "https://www.youtube.com/watch?v=qig4KOK2R2g"
        },
        {
            "anime_id": 5, "mal_id": 5, "title": "Cowboy Bebop: Knockin' on Heaven's Door", "type": "Movie",
            "genres": "Action|Sci-Fi", "themes": "Adult Cast|Space",
            "synopsis": "As the Cowboy Bebop crew travels the stars, they learn of a massive bounty on a terrorist planning to release a deadly virus on Mars before Halloween.",
            "studio": "Bones", "studios": "Bones", "score": 8.38, "episodes": 1, "year": 2001,
            "image_url": "https://cdn.myanimelist.net/images/anime/1439/93480.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 6, "mal_id": 6, "title": "Trigun", "type": "TV",
            "genres": "Action|Adventure|Sci-Fi", "themes": "Adult Cast",
            "synopsis": "Vash the Stampede is the man with a $60,000,000,000 bounty on his head. The reason: he's a merciless villain who lays waste to all those that oppose him across a desert planet.",
            "studio": "Madhouse", "studios": "Madhouse", "score": 8.22, "episodes": 26, "year": 1998,
            "image_url": "https://cdn.myanimelist.net/images/anime/7/20310.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 16498, "mal_id": 16498, "title": "Shingeki no Kyojin", "type": "TV",
            "genres": "Action|Award Winning|Drama|Suspense", "themes": "Gore|Military|Survival",
            "synopsis": "Centuries ago, mankind was slaughtered to near extinction by monstrous humanoid creatures called Titans, forcing humans to hide in fear behind enormous concentric walls.",
            "studio": "Wit Studio", "studios": "Wit Studio", "score": 8.54, "episodes": 25, "year": 2013,
            "image_url": "https://cdn.myanimelist.net/images/anime/10/47347.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 25777, "mal_id": 25777, "title": "Shingeki no Kyojin Season 2", "type": "TV",
            "genres": "Action|Drama|Suspense", "themes": "Gore|Military|Survival",
            "synopsis": "For centuries, humanity has been hunted by giant, mysterious predators known as the Titans. Eren Yeager and the Survey Corps fight to uncover the secrets behind the walls.",
            "studio": "Wit Studio", "studios": "Wit Studio", "score": 8.51, "episodes": 12, "year": 2017,
            "image_url": "https://cdn.myanimelist.net/images/anime/4/84177.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 9253, "mal_id": 9253, "title": "Steins;Gate", "type": "TV",
            "genres": "Drama|Sci-Fi|Suspense", "themes": "Psychological|Time Travel",
            "synopsis": "Eccentric scientist Rintarou Okabe has a never-ending thirst for scientific exploration. Together with his friends in Akihabara, he accidentally invents a gadget capable of sending messages into the past.",
            "studio": "White Fox", "studios": "White Fox", "score": 9.07, "episodes": 24, "year": 2011,
            "image_url": "https://cdn.myanimelist.net/images/anime/1935/127974.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 5114, "mal_id": 5114, "title": "Fullmetal Alchemist: Brotherhood", "type": "TV",
            "genres": "Action|Adventure|Drama|Fantasy", "themes": "Military",
            "synopsis": "After a horrific alchemy experiment goes wrong in the Elric household, brothers Edward and Alphonse are left in a catastrophic new reality and search for the Philosopher's Stone.",
            "studio": "Bones", "studios": "Bones", "score": 9.09, "episodes": 64, "year": 2009,
            "image_url": "https://cdn.myanimelist.net/images/anime/1208/94745.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 38000, "mal_id": 38000, "title": "Kimetsu no Yaiba", "type": "TV",
            "genres": "Action|Award Winning|Supernatural", "themes": "Historical|Mythology",
            "synopsis": "Ever since the death of his father, the burden of supporting the family has fallen upon Tanjirou Kamado's shoulders. When his family is slaughtered by demons, he sets out to cure his sister Nezuko.",
            "studio": "ufotable", "studios": "ufotable", "score": 8.45, "episodes": 26, "year": 2019,
            "image_url": "https://cdn.myanimelist.net/images/anime/1286/99889.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 4224, "mal_id": 4224, "title": "Toradora!", "type": "TV",
            "genres": "Drama|Romance", "themes": "Love Polygon|School",
            "synopsis": "Ryuji Takasu is a gentle high school student with a love for housework; but in contrast to his kind nature, he has an intimidating face that often gets him labeled as a delinquent. He meets Taiga Aisaka at school.",
            "studio": "J.C.Staff", "studios": "J.C.Staff", "score": 8.06, "episodes": 25, "year": 2008,
            "image_url": "https://cdn.myanimelist.net/images/anime/13/22128.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 37999, "mal_id": 37999, "title": "Kaguya-sama: Love is War", "type": "TV",
            "genres": "Comedy|Romance", "themes": "Psychological|School",
            "synopsis": "At the renowned Shuchiin Academy, Miyuki Shirogane and Kaguya Shinomiya are the student body's top representatives. Too proud to confess their love, they wage psychological warfare in school.",
            "studio": "A-1 Pictures", "studios": "A-1 Pictures", "score": 8.41, "episodes": 12, "year": 2019,
            "image_url": "https://cdn.myanimelist.net/images/anime/1295/106551.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 47917, "mal_id": 47917, "title": "Bocchi the Rock!", "type": "TV",
            "genres": "Comedy", "themes": "CGDCT|Music|School",
            "synopsis": "Yearning to make friends and perform live with a band, lonely and socially anxious Hitori Gotou devotes her time to playing the guitar and joins Kessoku Band.",
            "studio": "CloverWorks", "studios": "CloverWorks", "score": 8.78, "episodes": 12, "year": 2022,
            "image_url": "https://cdn.myanimelist.net/images/anime/1448/127956.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 1535, "mal_id": 1535, "title": "Death Note", "type": "TV",
            "genres": "Supernatural|Suspense", "themes": "Detective|Psychological",
            "synopsis": "Brutal murders, petty thefts, and senseless violence pollute the human world. High school genius Light Yagami discovers the Death Note, a notebook that kills anyone whose name is written inside.",
            "studio": "Madhouse", "studios": "Madhouse", "score": 8.62, "episodes": 37, "year": 2006,
            "image_url": "https://cdn.myanimelist.net/images/anime/1079/138100.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 11061, "mal_id": 11061, "title": "Hunter x Hunter (2011)", "type": "TV",
            "genres": "Action|Adventure|Fantasy", "themes": "Super Power",
            "synopsis": "Hunters devote themselves to accomplishing hazardous tasks, from traversing the world's uncharted territories to locating rare items and monsters. Young Gon Freecss takes the Hunter Exam to find his father.",
            "studio": "Madhouse", "studios": "Madhouse", "score": 9.04, "episodes": 148, "year": 2011,
            "image_url": "https://cdn.myanimelist.net/images/anime/1337/99013.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 40748, "mal_id": 40748, "title": "Jujutsu Kaisen", "type": "TV",
            "genres": "Action|Award Winning|Supernatural", "themes": "Mythology|School",
            "synopsis": "Idly indulging in baseless paranormal activities with the Occult Club, high schooler Yuuji Itadori swallows a cursed talisman—the finger of Sukuna, a demon—and enters Jujutsu High.",
            "studio": "MAPPA", "studios": "MAPPA", "score": 8.63, "episodes": 24, "year": 2020,
            "image_url": "https://cdn.myanimelist.net/images/anime/1171/109222.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 50265, "mal_id": 50265, "title": "Spy x Family", "type": "TV",
            "genres": "Action|Award Winning|Comedy", "themes": "Childcare|Espionage",
            "synopsis": "Corrupt politicians, frenzied nationalists, and other warmongering forces constantly jeopardize the thin veneer of peace. Master spy Twilight builds a fake family with an assassin and a telepath.",
            "studio": "Wit Studio|CloverWorks", "studios": "Wit Studio|CloverWorks", "score": 8.53, "episodes": 12, "year": 2022,
            "image_url": "https://cdn.myanimelist.net/images/anime/1441/122795.jpg",
            "trailer_url": ""
        },
        {
            "anime_id": 52991, "mal_id": 52991, "title": "Sousou no Frieren", "type": "TV",
            "genres": "Adventure|Drama|Fantasy", "themes": "Magic",
            "synopsis": "During their decade-long quest to defeat the Demon King, the members of the hero's party—Himmel himself, the priest Heiter, the dwarf warrior Eisen, and the elven mage Frieren—forge bonds across time.",
            "studio": "Madhouse", "studios": "Madhouse", "score": 9.36, "episodes": 28, "year": 2023,
            "image_url": "https://cdn.myanimelist.net/images/anime/1015/138006.jpg",
            "trailer_url": ""
        }
    ]
    return pd.DataFrame(data)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "anime_id" not in df.columns and "mal_id" in df.columns:
        df["anime_id"] = df["mal_id"]
    elif "mal_id" not in df.columns and "anime_id" in df.columns:
        df["mal_id"] = df["anime_id"]
    elif "anime_id" not in df.columns and "mal_id" not in df.columns:
        df["anime_id"] = np.arange(1, len(df) + 1)
        df["mal_id"] = df["anime_id"]

    if "studio" not in df.columns and "studios" in df.columns:
        df["studio"] = df["studios"]
    elif "studios" not in df.columns and "studio" in df.columns:
        df["studios"] = df["studio"]
    elif "studio" not in df.columns and "studios" not in df.columns:
        df["studio"] = "Unknown"
        df["studios"] = "Unknown"

    for col in ["title", "genres", "synopsis"]:
        if col not in df.columns:
            raise ValueError(f"CSV file is missing a required column: '{col}'")

    if "trailer_url" not in df.columns:
        df["trailer_url"] = df["trailer"] if "trailer" in df.columns else ""

    defaults = {
        "themes": "",
        "type": "TV",
        "score": "N/A",
        "episodes": "N/A",
        "year": "N/A",
        "image_url": "",
        "trailer_url": ""
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["title"] = df["title"].fillna("Untitled Anime").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    df["themes"] = df["themes"].fillna("").astype(str)
    df["synopsis"] = df["synopsis"].fillna("").astype(str)
    df["studio"] = df["studio"].fillna("Unknown").astype(str)
    df["studios"] = df["studios"].fillna("Unknown").astype(str)
    df["image_url"] = df["image_url"].fillna("").astype(str)
    df["trailer_url"] = df["trailer_url"].fillna("").astype(str)

    df = df.drop_duplicates(subset=["anime_id"]).reset_index(drop=True)
    return df


# ==============================================================================
# 1. MULTIMODAL & ATTRIBUTE REPRESENTATION (WEIGHTED SUBSPACE TF-IDF)
# ==============================================================================
W_G = 2.5
W_S = 0.8
W_V = 0.35


def clean_pipe_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("|", " ", regex=False).str.replace(",", " ", regex=False)


@st.cache_resource(show_spinner=False)
def build_multimodal_space(df: pd.DataFrame):
    tfidf_genres = TfidfVectorizer(stop_words="english")
    tfidf_studio = TfidfVectorizer()

    min_df_value = 4 if len(df) >= 4 else 1
    tfidf_synopsis = TfidfVectorizer(stop_words="english", min_df=min_df_value)

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
    return V_i, anime_vectors_dict, vectorizers, dims


# ==============================================================================
# 2. PERSONA-BASED USER PROFILING (CONTEXTUAL PRIOR SEEDING)
# ==============================================================================
class ContextualPriorSeeder:
    def __init__(self, anime_vectors: dict, genre_vectorizer, w_g=2.5, dim_sypnosis=0, dim_studio=0):
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


class LapsedUserPriorSeeder(ContextualPriorSeeder):
    def seed_lapsed_user(
        self,
        past_anime_ids: list,
        new_genres: list[str],
        lambda_1: float = 0.4,
        lambda_2: float = 0.6
    ) -> tuple[np.ndarray, set]:
        past_vecs = [self.anime_vectors[aid] for aid in past_anime_ids if aid in self.anime_vectors]
        sum_v_new = self._encode_genres_to_full_space(new_genres) * max(len(new_genres), 1)
        sum_v_past = np.sum(past_vecs, axis=0) if past_vecs else np.zeros_like(sum_v_new)
        v_user_0 = (lambda_1 * sum_v_past) + (lambda_2 * sum_v_new)
        return v_user_0, set(past_anime_ids)


# ==============================================================================
# 3. DEDUPLICATION, EXPLAINABILITY & COMPARE VIEW
# ==============================================================================
def get_main_studio(studios_val) -> str:
    if pd.isna(studios_val) or not str(studios_val).strip():
        return "Unknown"
    first_studio = re.split(r"[|,]", str(studios_val))[0].strip()
    return first_studio if first_studio else "Unknown"


def normalize_title(title: str) -> str:
    title = str(title).lower()
    patterns = [
        r"\bseason\s*\d+\b",
        r"\b\d+(st|nd|rd|th)\s+season\b",
        r"\bpart\s*\d+\b",
        r"\bseason\s+\d+\b"
    ]
    for pattern in patterns:
        title = re.sub(pattern, "", title)
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return " ".join(title.split())


def same_franchise(title_1: str, title_2: str, threshold: int = 90) -> bool:
    return calc_fuzz_ratio(normalize_title(title_1), normalize_title(title_2)) >= threshold


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


def split_full_vector_to_subspaces(full_vec: np.ndarray, dims: dict) -> dict:
    g_end = dims["genres"]
    s_end = g_end + dims["synopsis"]
    return {
        "genres": csr_matrix(full_vec[:g_end]),
        "synopsis": csr_matrix(full_vec[g_end:s_end]),
        "studio": csr_matrix(full_vec[s_end:])
    }


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
        reasons.append("it avoids your negative preferences such as " + ", ".join(neg_feats))

    if not reasons:
        return "Recommended to explore high-rated visual and studio diversity."
    return "Recommended because " + ", and ".join(reasons) + "."


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
            set_a = set([x.strip() for x in re.split(r"[|,]", str(val_a)) if x.strip()]) if pd.notna(val_a) else set()
            set_b = set([x.strip() for x in re.split(r"[|,]", str(val_b)) if x.strip()]) if pd.notna(val_b) else set()
            comparison[label]["common"] = sorted(set_a & set_b)
            comparison[label]["different_a"] = sorted(set_a - set_b)
            comparison[label]["different_b"] = sorted(set_b - set_a)
    return comparison


# ==============================================================================
# 4. ONLINE ROCCHIO ENGINE & QUALITY-BOOSTED MMR
# ==============================================================================
def update_user_vector_rocchio(v_user: np.ndarray, v_anime: np.ndarray, action: str, gamma: float = 0.95, alpha: float = 0.30, beta: float = 0.15) -> np.ndarray:
    delta_v = (alpha * v_anime) if action == "like" else (-beta * v_anime)
    return (gamma * v_user) + delta_v


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
    quality_weight: float = 0.25
) -> list[int]:
    candidates_df = df[~df["anime_id"].isin(excluded_ids)].copy()
    if candidates_df.empty:
        return []

    numeric_scores = pd.to_numeric(candidates_df["score"], errors="coerce")
    candidates_df["num_score"] = numeric_scores.fillna(6.5)

    # Quality Gating: Filter for likeable, well-rated titles first
    high_quality_df = candidates_df[candidates_df["num_score"] >= min_score]
    if len(high_quality_df) >= deck_size:
        candidates_df = high_quality_df
    elif len(high_quality_df) > 0:
        relaxed_df = candidates_df[candidates_df["num_score"] >= max(min_score - 1.0, 0.0)]
        if not relaxed_df.empty:
            candidates_df = relaxed_df

    cand_ids = candidates_df["anime_id"].tolist()
    cand_matrix = np.vstack([anime_vectors[aid] for aid in cand_ids])

    norm_quality = np.clip((candidates_df["num_score"].values - 5.0) / 4.5, 0.0, 1.0)

    if np.linalg.norm(v_user) > 1e-9:
        raw_cos = cosine_similarity(cand_matrix, v_user.reshape(1, -1)).flatten()
        norm_cos = (raw_cos + 1.0) / 2.0
        combined_rel = ((1.0 - quality_weight) * norm_cos) + (quality_weight * norm_quality)
    else:
        combined_rel = norm_quality

    candidates_df["sim_score"] = combined_rel
    candidates_df = candidates_df.sort_values(by="sim_score", ascending=False)

    top_pool = candidates_df.head(min(len(candidates_df), max(deck_size * 5, 35)))
    gated_df = deduplicate(
        top_pool,
        max_per_studio=max_per_studio,
        threshold=franchise_threshold,
        seen_titles=swiped_titles
    )
    if gated_df.empty:
        gated_df = candidates_df.head(deck_size)

    gated_ids = gated_df["anime_id"].tolist()
    if len(gated_ids) <= 1:
        return gated_ids

    gated_vecs = {aid: anime_vectors[aid] for aid in gated_ids}
    gated_user_sims = {int(row["anime_id"]): float(row["sim_score"]) for _, row in gated_df.iterrows()}

    selected_ids = [gated_ids[0]]
    remaining_ids = list(gated_ids[1:])

    while remaining_ids and len(selected_ids) < deck_size:
        best_id, best_mmr = None, -1e9
        selected_matrix = np.vstack([gated_vecs[sid] for sid in selected_ids])

        for cid in remaining_ids:
            c_vec = gated_vecs[cid].reshape(1, -1)
            sim_to_user = gated_user_sims[cid]
            sim_to_selected = np.max(cosine_similarity(c_vec, selected_matrix))
            mmr_score = (mmr_lambda * sim_to_user) - ((1.0 - mmr_lambda) * sim_to_selected)
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_id = cid

        selected_ids.append(best_id)
        remaining_ids.remove(best_id)

    return selected_ids


# ==============================================================================
# 5. YOUTUBE TRAILER RESOLVER & TINDER DRAGGABLE CARD (LEFT / RIGHT / UP)
# ==============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def get_youtube_trailer_embed(title: str, csv_url: str = "") -> str:
    url = str(csv_url).strip()
    if url and url.lower() != "nan":
        if "youtube.com/embed/" in url:
            return url
        match_v = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if match_v:
            return f"https://www.youtube.com/embed/{match_v.group(1)}?rel=0"

    try:
        query = urllib.parse.quote(f"{title} official anime trailer")
        search_url = f"https://www.youtube.com/results?search_query={query}"
        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9"
            }
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            html = response.read().decode("utf-8", errors="ignore")
            video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
            if video_ids:
                return f"https://www.youtube.com/embed/{video_ids[0]}?rel=0"
    except Exception:
        pass

    return ""


def render_tinder_draggable_card(anime_row: pd.Series, sim_score: float, explanation_text: str, pos_tags: list[str]):
    title = str(anime_row["title"])
    img_url = str(anime_row["image_url"]).strip()
    if not img_url or img_url == "nan":
        img_url = "https://placehold.co/600x850/1e1b4b/f43f5e?text=AniMatch+Poster"

    studio = get_main_studio(anime_row["studios"])
    score = str(anime_row.get("score", "N/A"))
    year = str(anime_row.get("year", "N/A"))
    episodes = str(anime_row.get("episodes", "N/A"))
    anime_type = str(anime_row.get("type", "TV"))
    full_synopsis = str(anime_row["synopsis"]).replace('"', '&quot;').replace("'", "&#39;")
    short_synopsis = full_synopsis if len(full_synopsis) <= 230 else full_synopsis[:227] + "..."

    genres_list = [g.strip() for g in re.split(r"[|,]", str(anime_row["genres"])) if g.strip()][:5]
    genres_badges_html = "".join([f'<span class="genre-pill">{g}</span>' for g in genres_list])

    pos_tags_html = ""
    if pos_tags:
        pos_tags_html = f'<div class="key-match">✨ <b>Matched Features:</b> {", ".join(pos_tags[:4])}</div>'

    embed_url = get_youtube_trailer_embed(title, anime_row.get("trailer_url", ""))
    yt_search_query = urllib.parse.quote(f"{title} official anime trailer")
    yt_search_url = f"https://www.youtube.com/results?search_query={yt_search_query}"

    if embed_url:
        trailer_html = f"""
        <div class="trailer-container">
            <iframe src="{embed_url}" title="Anime Trailer" frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen></iframe>
        </div>
        """
    else:
        trailer_html = f"""
        <div class="no-trailer-box">
            <div>🎬 Direct trailer embed unavailable for this title.</div>
            <a href="{yt_search_url}" target="_blank" class="yt-btn">▶️ Watch Trailer on YouTube</a>
        </div>
        """

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{
            box-sizing: border-box;
            user-select: none;
            -webkit-user-select: none;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        body {{
            margin: 0;
            padding: 4px 0;
            background: transparent;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .hint-bar {{
            display: flex;
            justify-content: space-between;
            width: 100%;
            max-width: 405px;
            margin-bottom: 8px;
            font-size: 12px;
            font-weight: 700;
        }}
        .hint-left {{ color: #f43f5e; }}
        .hint-up {{ color: #38bdf8; }}
        .hint-right {{ color: #10b981; }}
        .arena {{
            position: relative;
            width: 100%;
            max-width: 420px;
            height: 605px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .tinder-card {{
            position: relative;
            width: 398px;
            height: 590px;
            background: linear-gradient(165deg, #1e293b 0%, #0f172a 100%);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 20px 45px rgba(0, 0, 0, 0.55);
            overflow: hidden;
            cursor: grab;
            touch-action: none;
            will-change: transform;
            display: flex;
            flex-direction: column;
        }}
        .tinder-card:active {{ cursor: grabbing; }}
        .poster-wrap {{
            position: relative;
            width: 100%;
            height: 295px;
            overflow: hidden;
            background: #090d16;
        }}
        .poster-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            pointer-events: none;
        }}
        .poster-gradient {{
            position: absolute;
            bottom: 0; left: 0; right: 0;
            height: 110px;
            background: linear-gradient(to top, #0f172a 5%, transparent 100%);
        }}
        .sim-badge {{
            position: absolute;
            top: 14px; right: 14px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.4);
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .meta-badge {{
            position: absolute;
            top: 14px; left: 14px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(6px);
            color: #fbbf24;
            border: 1px solid rgba(251, 191, 36, 0.4);
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
        }}
        .stamp {{
            position: absolute;
            top: 38px;
            padding: 8px 18px;
            border-radius: 10px;
            font-size: 26px;
            font-weight: 900;
            letter-spacing: 2px;
            opacity: 0;
            z-index: 20;
            pointer-events: none;
        }}
        .stamp-like {{
            left: 22px;
            color: #10b981;
            border: 4px solid #10b981;
            transform: rotate(-14deg);
            background: rgba(16, 185, 129, 0.15);
        }}
        .stamp-pass {{
            right: 22px;
            color: #f43f5e;
            border: 4px solid #f43f5e;
            transform: rotate(14deg);
            background: rgba(244, 63, 94, 0.15);
        }}
        .stamp-info {{
            left: 50%;
            top: 120px;
            transform: translateX(-50%);
            color: #38bdf8;
            border: 4px solid #38bdf8;
            background: rgba(15, 23, 42, 0.85);
            font-size: 20px;
            white-space: nowrap;
        }}
        .card-body {{
            padding: 12px 18px 14px 18px;
            display: flex;
            flex-direction: column;
            gap: 7px;
            flex: 1;
            color: #f8fafc;
        }}
        .title-row {{
            font-size: 19px;
            font-weight: 800;
            line-height: 1.2;
            color: #ffffff;
        }}
        .sub-meta {{
            font-size: 12px;
            color: #94a3b8;
            font-weight: 600;
        }}
        .genres-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        .genre-pill {{
            background: rgba(244, 63, 94, 0.18);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.35);
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
        }}
        .synopsis {{
            font-size: 12px;
            line-height: 1.4;
            color: #cbd5e1;
            max-height: 66px;
            overflow: hidden;
        }}
        .explain-box {{
            margin-top: auto;
            background: rgba(56, 189, 248, 0.1);
            border-left: 3px solid #38bdf8;
            padding: 7px 10px;
            border-radius: 8px;
            font-size: 11px;
            color: #e0f2fe;
            line-height: 1.35;
        }}
        .key-match {{
            margin-top: 3px;
            color: #7dd3fc;
            font-size: 10.5px;
        }}
        .swipe-up-cue {{
            text-align: center;
            font-size: 11px;
            color: #38bdf8;
            font-weight: 700;
            padding-top: 2px;
            cursor: pointer;
        }}

        /* DETAIL DRAWER (SWIPE UP) */
        .detail-drawer {{
            position: absolute;
            left: 0; right: 0; bottom: 0;
            height: 100%;
            background: #0f172a;
            z-index: 50;
            transform: translateY(100%);
            transition: transform 0.32s cubic-bezier(0.2, 0.85, 0.25, 1);
            display: flex;
            flex-direction: column;
            padding: 16px 18px;
            color: #f8fafc;
            cursor: default;
        }}
        .detail-drawer.open {{
            transform: translateY(0%);
        }}
        .drawer-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}
        .drawer-title {{
            font-size: 16px;
            font-weight: 800;
            color: #38bdf8;
        }}
        .close-btn {{
            background: rgba(244, 63, 94, 0.2);
            color: #fda4af;
            border: 1px solid rgba(244, 63, 94, 0.4);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
        }}
        .close-btn:hover {{
            background: rgba(244, 63, 94, 0.35);
        }}
        .drawer-content {{
            overflow-y: auto;
            flex: 1;
            padding-right: 4px;
            user-select: text;
            -webkit-user-select: text;
        }}
        .drawer-section-label {{
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            margin: 10px 0 6px 0;
        }}
        .trailer-container {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%;
            border-radius: 12px;
            overflow: hidden;
            background: #000;
            border: 1px solid rgba(255,255,255,0.15);
        }}
        .trailer-container iframe {{
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
        }}
        .no-trailer-box {{
            background: rgba(30, 41, 59, 0.8);
            border: 1px dashed rgba(56, 189, 248, 0.4);
            border-radius: 12px;
            padding: 14px;
            text-align: center;
            font-size: 12px;
            color: #cbd5e1;
        }}
        .yt-btn {{
            display: inline-block;
            margin-top: 8px;
            background: #ef4444;
            color: #ffffff;
            text-decoration: none;
            padding: 6px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 12px;
        }}
        .full-synopsis-text {{
            font-size: 13px;
            line-height: 1.55;
            color: #e2e8f0;
            white-space: pre-line;
        }}
    </style>
    </head>
    <body>
        <div class="hint-bar">
            <span class="hint-left">⬅️ Drag LEFT: PASS</span>
            <span class="hint-up">⬆️ Swipe UP: Trailer & Info</span>
            <span class="hint-right">Drag RIGHT: LIKE ➡️</span>
        </div>
        <div class="arena">
            <div id="tinderCard" class="tinder-card">
                <div id="stampLike" class="stamp stamp-like">LIKE 💚</div>
                <div id="stampPass" class="stamp stamp-pass">PASS ❌</div>
                <div id="stampInfo" class="stamp stamp-info">INFO & TRAILER ⬆️</div>

                <div class="poster-wrap">
                    <img src="{img_url}" alt="Poster" onerror="this.src='https://placehold.co/600x850/1e1b4b/f43f5e?text=Anime+Poster'"/>
                    <div class="poster-gradient"></div>
                    <div class="meta-badge">⭐ {score} | {anime_type} ({year})</div>
                    <div class="sim-badge">Match: {sim_score:.3f}</div>
                </div>

                <div class="card-body">
                    <div class="title-row">{title}</div>
                    <div class="sub-meta">🎬 Studio: {studio} &nbsp;•&nbsp; 📺 {episodes} eps</div>
                    <div class="genres-row">{genres_badges_html}</div>
                    <div class="synopsis">{short_synopsis}</div>
                    <div class="explain-box">
                        💡 <b>Explainable AI:</b> {explanation_text}
                        {pos_tags_html}
                    </div>
                    <div id="btnOpenDrawer" class="swipe-up-cue">⬆️ Drag card UP to watch Trailer & read Full Synopsis</div>
                </div>

                <!-- SWIPE-UP DETAIL DRAWER -->
                <div id="detailDrawer" class="detail-drawer">
                    <div class="drawer-header">
                        <div class="drawer-title">🎬 {title}</div>
                        <button id="btnCloseDrawer" class="close-btn">⬇️ Close</button>
                    </div>
                    <div class="drawer-content">
                        <div class="drawer-section-label">🎥 Official Trailer</div>
                        {trailer_html}
                        <div class="drawer-section-label">📖 Full Synopsis</div>
                        <div class="full-synopsis-text">{full_synopsis}</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Hide Streamlit bridge buttons completely
            function hideBridgeButtons() {{
                try {{
                    const buttons = window.parent.document.querySelectorAll('button');
                    buttons.forEach(btn => {{
                        if (btn.innerText && btn.innerText.includes('BRIDGE_SWIPE_')) {{
                            const box = btn.closest('div[data-testid="stButton"]') || btn.parentElement;
                            if (box) {{
                                box.style.position = 'fixed';
                                box.style.left = '-9999px';
                                box.style.opacity = '0';
                                box.style.height = '0';
                                box.style.overflow = 'hidden';
                            }}
                        }}
                    }});
                }} catch (e) {{}}
            }}
            hideBridgeButtons();
            setTimeout(hideBridgeButtons, 150);

            const card = document.getElementById('tinderCard');
            const stampLike = document.getElementById('stampLike');
            const stampPass = document.getElementById('stampPass');
            const stampInfo = document.getElementById('stampInfo');
            const detailDrawer = document.getElementById('detailDrawer');
            const btnCloseDrawer = document.getElementById('btnCloseDrawer');
            const btnOpenDrawer = document.getElementById('btnOpenDrawer');

            let isDragging = false;
            let isDrawerOpen = false;
            let startX = 0, startY = 0, currentX = 0, currentY = 0;
            const SWIPE_X_THRESHOLD = 100;
            const SWIPE_UP_THRESHOLD = -80;

            function openDrawer() {{
                isDrawerOpen = true;
                detailDrawer.classList.add('open');
                card.style.transform = 'translate(0px, 0px) rotate(0deg)';
                card.style.borderColor = 'rgba(56, 189, 248, 0.75)';
                stampLike.style.opacity = 0;
                stampPass.style.opacity = 0;
                stampInfo.style.opacity = 0;
            }}

            function closeDrawer() {{
                isDrawerOpen = false;
                detailDrawer.classList.remove('open');
                card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
            }}

            btnCloseDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());
            btnCloseDrawer.addEventListener('click', (e) => {{
                e.stopPropagation();
                closeDrawer();
            }});

            btnOpenDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());
            btnOpenDrawer.addEventListener('click', (e) => {{
                e.stopPropagation();
                openDrawer();
            }});

            detailDrawer.addEventListener('pointerdown', (e) => e.stopPropagation());

            function triggerStreamlitSwipe(direction) {{
                const targetMarker = direction === 'right' ? 'BRIDGE_SWIPE_LIKE' : 'BRIDGE_SWIPE_PASS';
                try {{
                    const parentButtons = window.parent.document.querySelectorAll('button');
                    for (const btn of parentButtons) {{
                        if (btn.innerText && btn.innerText.includes(targetMarker)) {{
                            btn.click();
                            return;
                        }}
                    }}
                }} catch (err) {{}}
            }}

            card.addEventListener('pointerdown', (e) => {{
                if (isDrawerOpen) return;
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                currentX = 0;
                currentY = 0;
                card.style.transition = 'none';
                card.setPointerCapture(e.pointerId);
            }});

            card.addEventListener('pointermove', (e) => {{
                if (!isDragging || isDrawerOpen) return;
                currentX = e.clientX - startX;
                currentY = e.clientY - startY;

                if (currentY < -20 && Math.abs(currentY) > Math.abs(currentX)) {{
                    const liftY = Math.max(currentY * 0.45, -70);
                    card.style.transform = `translate(0px, ${{liftY}}px) scale(1.02)`;
                    stampInfo.style.opacity = Math.min((Math.abs(currentY) - 20) / 60, 1);
                    stampLike.style.opacity = 0;
                    stampPass.style.opacity = 0;
                    card.style.borderColor = 'rgba(56, 189, 248, 0.8)';
                }} else {{
                    const dampY = currentY * 0.25;
                    const rotate = currentX * 0.06;
                    card.style.transform = `translate(${{currentX}}px, ${{dampY}}px) rotate(${{rotate}}deg)`;
                    stampInfo.style.opacity = 0;

                    if (currentX > 20) {{
                        stampLike.style.opacity = Math.min((currentX - 20) / 80, 1);
                        stampPass.style.opacity = 0;
                        card.style.borderColor = 'rgba(16, 185, 129, 0.75)';
                    }} else if (currentX < -20) {{
                        stampPass.style.opacity = Math.min((Math.abs(currentX) - 20) / 80, 1);
                        stampLike.style.opacity = 0;
                        card.style.borderColor = 'rgba(244, 63, 94, 0.75)';
                    }} else {{
                        stampLike.style.opacity = 0;
                        stampPass.style.opacity = 0;
                        card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
                    }}
                }}
            }});

            function endDrag() {{
                if (!isDragging || isDrawerOpen) return;
                isDragging = false;
                card.style.transition = 'transform 0.26s cubic-bezier(0.2, 0.8, 0.2, 1), opacity 0.26s ease';

                if (currentY <= SWIPE_UP_THRESHOLD && Math.abs(currentY) > Math.abs(currentX)) {{
                    openDrawer();
                }} else if (currentX >= SWIPE_X_THRESHOLD) {{
                    card.style.transform = `translate(650px, ${{currentY * 0.25}}px) rotate(28deg)`;
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('right'), 170);
                }} else if (currentX <= -SWIPE_X_THRESHOLD) {{
                    card.style.transform = `translate(-650px, ${{currentY * 0.25}}px) rotate(-28deg)`;
                    card.style.opacity = '0';
                    setTimeout(() => triggerStreamlitSwipe('left'), 170);
                }} else {{
                    card.style.transform = 'translate(0px, 0px) rotate(0deg)';
                    card.style.borderColor = 'rgba(255, 255, 255, 0.14)';
                    stampLike.style.opacity = 0;
                    stampPass.style.opacity = 0;
                    stampInfo.style.opacity = 0;
                }}
                currentX = 0;
                currentY = 0;
            }}

            card.addEventListener('pointerup', endDrag);
            card.addEventListener('pointercancel', endDrag);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=640, scrolling=False)


# ==============================================================================
# 6. ONBOARDING PERSONA MODAL DIALOG
# ==============================================================================
@st.dialog("🧭 Onboarding Persona Setup — AniMatch", width="large")
def open_onboarding_modal(df, anime_vectors, vectorizers, dims, all_genres, mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight):
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
            st.session_state.v_user = v_user_0
            st.session_state.excluded_ids = excluded_ids
            st.session_state.swiped_titles = []
            st.session_state.liked_ids = []
            st.session_state.passed_ids = []
            st.session_state.swipe_count = 0
            st.session_state.deck = generate_diverse_deck(
                df, anime_vectors, v_user_0, excluded_ids, [], 8,
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
            )
            st.session_state.onboarded = True
            st.session_state.show_onboarding_modal = False
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
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
            )
            st.session_state.onboarded = True
            st.session_state.show_onboarding_modal = False
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
                mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
            )
            st.session_state.onboarded = True
            st.session_state.show_onboarding_modal = False
            st.rerun()


# ==============================================================================
# 7. SESSION STATE & SWIPE HANDLER
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
        "dataset_signature": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_swipe(
    action: str,
    df: pd.DataFrame,
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
    anime_row = df[df["anime_id"] == current_aid].iloc[0]
    v_anime = anime_vectors[current_aid]

    # Dampen negative repulsion before the user has any positive Like
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
        quality_weight=quality_weight
    )


# ==============================================================================
# 8. MAIN APPLICATION
# ==============================================================================
def main():
    init_session_state()

    # Clean Top Bar (Auto-loads local CSV, no manual upload widget)
    top_c1, top_c2 = st.columns([3.5, 1.5])
    with top_c1:
        st.markdown("## 🔥 AniMatch")
    with top_c2:
        if st.button("🧭 Change Persona / Reset", use_container_width=True):
            st.session_state.show_onboarding_modal = True
            st.rerun()

    # Auto-detect .csv files in the same directory
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
            df = normalize_dataframe(pd.read_csv(selected_local_path))
            file_name = os.path.basename(selected_local_path)
            sig = f"local_{file_name}_{len(df)}"
            data_source_label = f"Local CSV ({file_name})"
        except Exception as e:
            st.error(f"Error reading `{os.path.basename(selected_local_path)}`: {e}")
            st.stop()
    else:
        df = normalize_dataframe(get_fallback_dataframe())
        sig = f"default_{len(df)}"
        data_source_label = "Built-in Sample Dataset"

    # Reset state cleanly if dataset changes
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

    # Sidebar Algorithm Hyperparameters
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

    V_i, anime_vectors, vectorizers, dims = build_multimodal_space(df)
    all_genres = sorted(list({
        g.strip()
        for val in (df["genres"] + "|" + df["themes"]).tolist()
        for g in re.split(r"[|,]", str(val))
        if g.strip()
    }))

    # Initialize or re-align user vector dimensions safely
    expected_dim = V_i.shape[1]
    if st.session_state.v_user is None or st.session_state.v_user.shape[0] != expected_dim:
        seeder = ContextualPriorSeeder(anime_vectors, vectorizers["genres"], W_G, dims["synopsis"], dims["studio"])
        v_user_0, excluded_ids = seeder.seed_cold_start("eyecandy_first", [])
        st.session_state.v_user = v_user_0
        st.session_state.excluded_ids = excluded_ids
        st.session_state.deck = generate_diverse_deck(
            df, anime_vectors, v_user_0, excluded_ids, [], 8,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
        )

    # Show Onboarding Modal
    if st.session_state.show_onboarding_modal:
        open_onboarding_modal(
            df, anime_vectors, vectorizers, dims, all_genres,
            mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
        )

    # Compact Status Bar
    st.caption(
        f"Source: **{data_source_label}** • "
        f"Persona: **{st.session_state.persona_mode}** • "
        f"Catalog: **{len(df)} anime** • "
        f"Swiped: **{st.session_state.swipe_count}** • "
        f"Liked: **{len(st.session_state.liked_ids)}**"
    )

    if st.session_state.last_action_msg:
        st.toast(st.session_state.last_action_msg)

    # ==========================================================================
    # MAIN TINDER SWIPE ARENA
    # ==========================================================================
    if not st.session_state.deck:
        st.warning("🎉 You have swiped through all available anime in the current filter! Click **Change Persona / Reset** or lower the Minimum Rating Score in the sidebar.")
    else:
        current_aid = st.session_state.deck[0]
        current_anime = df[df["anime_id"] == current_aid].iloc[0]
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

            # Hidden JS -> Streamlit event bridge buttons
            if st.button("BRIDGE_SWIPE_PASS", key="btn_bridge_pass"):
                process_swipe(
                    "pass", df, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.rerun()
            if st.button("BRIDGE_SWIPE_LIKE", key="btn_bridge_like"):
                process_swipe(
                    "like", df, anime_vectors, gamma, alpha, beta,
                    mmr_lambda, max_per_studio, franchise_thresh, min_score, quality_weight
                )
                st.rerun()

        with col_side:
            st.markdown("### 🧠 Real-Time Vector & Deck Queue")
            if st.session_state.last_action_msg:
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
                r = df[df["anime_id"] == u_aid].iloc[0]
                upcoming_rows.append({
                    "#": "👉 Active" if rank_idx == 1 else f"#{rank_idx}",
                    "Anime": r["title"],
                    "Score": r["score"],
                    "Studio": get_main_studio(r["studios"]),
                    "Genres": r["genres"]
                })
            st.dataframe(pd.DataFrame(upcoming_rows), use_container_width=True, hide_index=True)

    # ==========================================================================
    # LIKED ANIME & COMPARE VIEW EXPANDER
    # ==========================================================================
    with st.expander(f"💚 Liked Anime ({len(st.session_state.liked_ids)}) & ⚖️ Side-by-Side Compare View", expanded=False):
        if st.session_state.liked_ids:
            liked_df = df[df["anime_id"].isin(st.session_state.liked_ids)][
                ["anime_id", "title", "type", "genres", "studio", "score", "episodes", "year"]
            ]
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
                c2 = st.selectbox("Select Anime B:", options=labels_list, index=min(1, len(labels_list) - 1))

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