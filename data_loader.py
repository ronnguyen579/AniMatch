import json
import os
import re
import urllib.request
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    def calc_fuzz_ratio(s1: str, s2: str) -> float:
        return fuzz.ratio(s1, s2)
except ImportError:
    from difflib import SequenceMatcher
    def calc_fuzz_ratio(s1: str, s2: str) -> float:
        return SequenceMatcher(None, s1, s2).ratio() * 100.0

JIKAN_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jikan_tv_movie_cache.csv")

ANILIST_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($page: Int, $perPage: Int, $formats: [MediaFormat]) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(type: ANIME, format_in: $formats, sort: POPULARITY_DESC, isAdult: false) {
      idMal
      title {
        english
        romaji
      }
      format
      episodes
      seasonYear
      averageScore
      genres
      tags {
        name
        rank
      }
      description(asHtml: false)
      studios(isMain: true) {
        nodes {
          name
        }
      }
      coverImage {
        extraLarge
        large
      }
      trailer {
        id
        site
      }
    }
  }
}
"""

SEQUEL_PATTERNS = re.compile(
    r"\b(second season|third season|fourth season|sequel to|continuation of|"
    r"following the events of|resumes after|direct sequel)\b",
    re.IGNORECASE
)
PREQUEL_PATTERNS = re.compile(
    r"\b(prequel to|takes place before|set prior to|origin story of)\b",
    re.IGNORECASE
)
SPINOFF_PATTERNS = re.compile(
    r"\b(spin-?off|side story|parallel world|alternate setting|alternative retelling|"
    r"same universe as|gaiden)\b",
    re.IGNORECASE
)


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

def load_hybrid_catalog(csv_path: str | None, api_pages: int = 2) -> pd.DataFrame:
    frames = []

    # 1. Lấy 100 bộ thịnh hành nhất từ API (đặt lên đầu để ưu tiên cập nhật poster HD & trailer)
    api_df = fetch_jikan_catalog(pages_per_type=api_pages, force_refresh=False)
    if api_df is not None and len(api_df) > 16:
        frames.append(api_df)

    # 2. Đọc kho tĩnh 12.000 bộ từ CSV và lọc chỉ giữ TV & Movie
    if csv_path and os.path.exists(csv_path):
        csv_df = pd.read_csv(csv_path)
        if "type" in csv_df.columns:
            csv_df = csv_df[csv_df["type"].astype(str).str.upper().isin(["TV", "MOVIE"])].copy()

        # Đồng bộ cột mal_id và anime_id trước khi gộp để không bị dropna xóa nhầm
        if "mal_id" not in csv_df.columns and "anime_id" in csv_df.columns:
            csv_df["mal_id"] = csv_df["anime_id"]
        elif "anime_id" not in csv_df.columns and "mal_id" in csv_df.columns:
            csv_df["anime_id"] = csv_df["mal_id"]

        frames.append(csv_df)

    if not frames:
        return normalize_dataframe(get_fallback_dataframe())

    combined = pd.concat(frames, ignore_index=True)
    combined["mal_id"] = pd.to_numeric(combined["mal_id"], errors="coerce")
    combined = combined.dropna(subset=["mal_id"])
    combined["mal_id"] = combined["mal_id"].astype(int)
    combined["anime_id"] = combined["mal_id"]

    # Khử trùng lớp 1: Giữ bản từ API nếu trùng mal_id, giữ bản CSV cho ~11.900 bộ còn lại
    combined = combined.drop_duplicates(subset=["mal_id"], keep="first").reset_index(drop=True)

    # Khử trùng lớp 2: Chặn trùng tên chuẩn hóa + năm phát hành
    combined["_dedup_key"] = (
        combined["title"].apply(normalize_title) + "_" + combined["year"].astype(str)
    )
    combined = combined.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])

    return normalize_dataframe(combined.reset_index(drop=True))


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
            raise ValueError(f"Dataset is missing a required column: '{col}'")

    for candidate_col in ["trailer_url", "trailer_embed_url", "trailer", "trailer_youtube_id"]:
        if candidate_col in df.columns:
            df["trailer_url"] = df[candidate_col].fillna("").astype(str)
            break
    else:
        df["trailer_url"] = ""

    defaults = {
        "themes": "",
        "type": "TV",
        "score": "N/A",
        "episodes": "N/A",
        "year": "N/A",
        "image_url": "",
        "trailer_url": "",
        "artstyle_tags": "",
        "visual_tags_full": ""
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
    df["artstyle_tags"] = df["artstyle_tags"].fillna("").astype(str)
    df["visual_tags_full"] = df["visual_tags_full"].fillna("").astype(str)

    df = df.drop_duplicates(subset=["anime_id"]).reset_index(drop=True)
    df = enrich_franchise_metadata(df)
    return df


def clean_pipe_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("|", " ", regex=False).str.replace(",", " ", regex=False)


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


def extract_season_number(title: str) -> int:
    t = str(title).lower()
    if "final season" in t:
        return 99

    match_num = re.search(r"\b(?:season|part)\s*(\d+)\b", t)
    if match_num:
        return int(match_num.group(1))

    match_ord = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+season\b", t)
    if match_ord:
        return int(match_ord.group(1))

    roman_map = {"ii": 2, "iii": 3, "iv": 4, "v": 5}
    match_roman = re.search(r"\b(ii|iii|iv|v)\s*$", t)
    if match_roman:
        return roman_map[match_roman.group(1)]

    return 1


def get_franchise_root(title: str) -> str:
    base = normalize_title(title)
    base = re.sub(r"\b(ii|iii|iv|v|final)\s*$", "", base).strip()
    return base


def same_franchise(title_1: str, title_2: str, threshold: int = 90) -> bool:
    return calc_fuzz_ratio(normalize_title(title_1), normalize_title(title_2)) >= threshold


def extract_proper_nouns(text: str) -> set:
    words = re.findall(r"(?<!\. )\b[A-Z][a-z]{3,}\b", str(text))
    ignore = {"When", "After", "Before", "While", "During", "However", "Despite", "Their", "These", "Those", "With", "From", "Into", "Through"}
    return {w.lower() for w in words if w not in ignore}


def enrich_franchise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_year"] = pd.to_numeric(df["year"], errors="coerce").fillna(9999)
    df["is_tv"] = (df["type"].astype(str).str.upper() == "TV").astype(int)

    roles = []
    for _, row in df.iterrows():
        title = str(row["title"])
        syn_head = str(row["synopsis"])[:220]
        s_num = extract_season_number(title)

        if SPINOFF_PATTERNS.search(syn_head) or str(row["type"]).upper() in ("OVA", "SPECIAL", "ONA"):
            roles.append("SPIN_OFF")
        elif PREQUEL_PATTERNS.search(syn_head):
            roles.append("PREQUEL")
        elif s_num > 1 or SEQUEL_PATTERNS.search(syn_head):
            roles.append("SEQUEL")
        else:
            roles.append("ENTRY")

    df["franchise_role"] = roles

    df["prefix_root"] = (
        df["title"].astype(str)
        .str.split(r"[:;/\-–]").str[0]
        .apply(get_franchise_root)
    )
    df["proper_nouns"] = df["synopsis"].apply(extract_proper_nouns)
    df["main_studio"] = df["studios"].apply(get_main_studio)

    entry_candidates = df.sort_values(
        by=["is_tv", "num_year", "anime_id"],
        ascending=[False, True, True]
    )
    prefix_to_entry = {}
    for _, row in entry_candidates.iterrows():
        p_root = row["prefix_root"]
        if len(p_root) >= 4 and p_root not in prefix_to_entry:
            prefix_to_entry[p_root] = int(row["anime_id"])

    entry_id_map = {}
    studio_groups = df.groupby("main_studio")

    for _, row in df.iterrows():
        aid = int(row["anime_id"])
        p_root = row["prefix_root"]
        role = row["franchise_role"]
        resolved_entry = prefix_to_entry.get(p_root, aid)

        if role in ("SEQUEL", "PREQUEL") and resolved_entry == aid:
            studio = row["main_studio"]
            nouns_a = row["proper_nouns"]
            if studio and studio != "Unknown" and len(nouns_a) >= 2 and studio in studio_groups.groups:
                peers = studio_groups.get_group(studio)
                earlier_peers = peers[peers["num_year"] <= row["num_year"]]
                best_overlap = 1
                for _, p_row in earlier_peers.iterrows():
                    p_aid = int(p_row["anime_id"])
                    if p_aid == aid:
                        continue
                    overlap = len(nouns_a & p_row["proper_nouns"])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        resolved_entry = entry_id_map.get(p_aid, p_aid)

        entry_id_map[aid] = resolved_entry

    df["entry_anime_id"] = df["anime_id"].astype(int).map(entry_id_map)
    id_to_title = dict(zip(df["anime_id"].astype(int), df["title"].astype(str)))
    df["parent_title"] = df["entry_anime_id"].map(id_to_title).fillna("")
    return df


def fetch_jikan_catalog(pages_per_type: int = 4, force_refresh: bool = False) -> pd.DataFrame:
    """
    Kéo danh sách TV & Movie siêu tốc qua AniList GraphQL (50 bộ/trang),
    tự động map sang cấu trúc chuẩn MAL/AniMatch.
    """
    if not force_refresh and os.path.exists(JIKAN_CACHE_FILE):
        try:
            cached_df = pd.read_csv(JIKAN_CACHE_FILE)
            if not cached_df.empty:
                return normalize_dataframe(cached_df)
        except Exception:
            pass

    total_pages = max(2, pages_per_type)
    records = []
    seen_ids = set()

    for page in range(1, total_pages + 1):
        payload = json.dumps({
            "query": ANILIST_QUERY,
            "variables": {
                "page": page,
                "perPage": 50,
                "formats": ["TV", "MOVIE"]
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            ANILIST_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "AniMatchApp/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))

            page_data = (data.get("data") or {}).get("Page") or {}
            items = page_data.get("media") or []
            if not items:
                break

            for item in items:
                mal_id = item.get("idMal")
                if not mal_id or int(mal_id) in seen_ids:
                    continue
                mal_id = int(mal_id)
                seen_ids.add(mal_id)

                t_obj = item.get("title") or {}
                title = t_obj.get("english") or t_obj.get("romaji") or "Untitled Anime"
                fmt = "Movie" if item.get("format") == "MOVIE" else "TV"

                genres = "|".join(item.get("genres") or [])
                themes = "|".join([
                    tg["name"] for tg in (item.get("tags") or [])
                    if tg.get("name") and (tg.get("rank") or 0) >= 60
                ][:6])

                studio_nodes = ((item.get("studios") or {}).get("nodes")) or []
                studios = "|".join([s["name"] for s in studio_nodes if s.get("name")]) or "Unknown"

                raw_desc = str(item.get("description") or "")
                clean_syn = re.sub(r"<[^>]+>", " ", raw_desc).strip()

                avg_sc = item.get("averageScore")
                score_val = round(float(avg_sc) / 10.0, 2) if avg_sc else "N/A"

                eps_val = item.get("episodes")
                if eps_val is None:
                    eps_val = 1 if fmt == "Movie" else "N/A"

                cover = item.get("coverImage") or {}
                image_url = cover.get("extraLarge") or cover.get("large") or ""

                tr = item.get("trailer") or {}
                trailer_url = (
                    f"https://www.youtube.com/watch?v={tr['id']}"
                    if tr.get("site") == "youtube" and tr.get("id")
                    else ""
                )

                records.append({
                    "anime_id": mal_id,
                    "mal_id": mal_id,
                    "title": title,
                    "type": fmt,
                    "genres": genres,
                    "themes": themes,
                    "synopsis": clean_syn,
                    "studio": studios,
                    "studios": studios,
                    "score": score_val,
                    "episodes": eps_val,
                    "year": item.get("seasonYear") or "N/A",
                    "image_url": image_url,
                    "trailer_url": trailer_url,
                })

            if not (page_data.get("pageInfo") or {}).get("hasNextPage", False):
                break
        except Exception:
            continue

    if not records:
        return normalize_dataframe(get_fallback_dataframe())

    raw_df = pd.DataFrame(records)
    try:
        raw_df.to_csv(JIKAN_CACHE_FILE, index=False)
    except Exception:
        pass

    return normalize_dataframe(raw_df)
