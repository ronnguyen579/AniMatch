import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    import site
    for sp in site.getsitepackages():
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            for root, dirs, files in os.walk(nvidia_dir):
                if any(f.endswith(".dll") for f in files):
                    os.add_dll_directory(root)
                    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
import glob
import urllib.request
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# ==============================================================================
# CẤU HÌNH ĐẦU VÀO / ĐẦU RA
# ==============================================================================
INPUT_CSV = "anime_tv_movie_first_part.csv"  # Để trống để tự tìm file .csv đầu tiên trong thư mục, hoặc điền tên cụ thể (vd: "anime_data.csv")
OUTPUT_CSV = "anime_with_artstyle.csv"
REPO_ID = "SmilingWolf/wd-vit-tagger-v3"
CONFIDENCE_THRESHOLD = 0.35

# ==============================================================================
# BỘ LỌC DANBOORU TAGS CHO ARTSTYLE & THỊ GIÁC
# ==============================================================================
# 1. Hậu tố/tiền tố chỉ ngoại hình nhân vật & trang phục cần loại khỏi không gian thị giác chung
NON_STYLE_PREFIXES_SUFFIXES = (
    "_hair", "_eyes", "_skin", "_ears", "_tail", "_wings", "_horns",
    "_shirt", "_skirt", "_dress", "_jacket", "_uniform", "_hat", "_shoes",
    "_gloves", "_ribbon", "_bow", "_collar", "_choker", "_glasses",
    "girl", "girls", "boy", "boys", "solo", "multiple_", "couple", "1other"
)

# 2. Hậu tố cấu trúc đặc trưng của Danbooru cho phong cách & kỹ thuật vẽ
STYLE_STRUCTURAL_SUFFIXES = (
    "_style", "_artstyle", "_(style)", "_(medium)", "_media",
    "_shading", "_lighting", "_color", "_colors", "_coloring",
    "_lineart", "_lines", "_background", "_theme", "_aesthetic",
    "_angle", "_shot", "_focus", "_effect", "_filter", "_texture"
)

# 3. Các tag thẩm mỹ, chất liệu, thời kỳ, ánh sáng & bố cục cốt lõi
CORE_AESTHETIC_TAGS = {
    "monochrome", "greyscale", "sepia", "sketch", "rough_sketch", "lineart", "lineless",
    "watercolor_(medium)", "oil_painting_(medium)", "acrylic_paint_(medium)", "gouache_(medium)",
    "ink_(medium)", "colored_pencil_(medium)", "graphite_(medium)", "pastel_(medium)",
    "marker_(medium)", "airbrush", "impasto", "cel_shading", "flat_color", "painterly",
    "traditional_media", "digital_media", "mixed_media", "pixel_art", "3d", "cgi",
    "papercraft", "collage", "stippling", "hatching_(texture)", "crosshatching",
    "halftone", "screentones", "retro_artstyle", "1970s_(style)", "1980s_(style)",
    "1990s_(style)", "2000s_(style)", "classic", "vintage", "art_nouveau", "art_deco",
    "pop_art", "ukiyo-e", "sumi-e", "impressionism", "surrealism", "minimalism",
    "psychedelic", "abstract", "realism", "photorealistic", "semi-realistic",
    "chibi", "super_deformed", "moe", "gekiga", "noir", "cyberpunk", "steampunk",
    "gothic", "vaporwave", "synthwave", "high_contrast", "low_contrast", "chiaroscuro",
    "dramatic_lighting", "soft_lighting", "hard_lighting", "backlighting", "rim_lighting",
    "cinematic_lighting", "volumetric_lighting", "god_rays", "sunbeam", "dappled_sunlight",
    "lens_flare", "chromatic_aberration", "bloom", "glowing", "neon", "neon_lights",
    "silhouette", "reflection", "bokeh", "depth_of_field", "motion_blur", "film_grain",
    "vhs", "scanlines", "glitch", "vignetting", "light_particles", "sparkle",
    "vibrant", "muted_color", "pastel_colors", "pale_color", "dark", "bright",
    "warm_colors", "cool_colors", "limited_palette", "spot_color", "oversaturated",
    "dutch_angle", "fisheye", "wide_shot", "close-up", "extreme_close-up", "pov",
    "from_above", "from_below", "bird's-eye_view", "worm's-eye_view", "symmetrical",
    "poster", "cover", "key_visual", "official_art", "concept_art", "game_cg"
}


def is_visual_general_tag(tag: str) -> bool:
    """Giữ lại toàn bộ tag thị giác/bối cảnh/phong cách, loại bỏ tag đếm người & quần áo."""
    return not (tag.endswith(NON_STYLE_PREFIXES_SUFFIXES) or tag.startswith(NON_STYLE_PREFIXES_SUFFIXES))


def is_pure_artstyle_tag(tag: str) -> bool:
    """Chỉ giữ lại duy nhất các tag có mặt chính xác trong CORE_AESTHETIC_TAGS."""
    return tag in CORE_AESTHETIC_TAGS


# ==============================================================================
# TẢI MÔ HÌNH & HÀM XỬ LÝ ẢNH
# ==============================================================================
def load_wd14_model():
    print(f"⏳ Đang tải/nạp mô hình {REPO_ID} lên GPU...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.onnx")
    tags_path = hf_hub_download(repo_id=REPO_ID, filename="selected_tags.csv")

    tags_df = pd.read_csv(tags_path)
    tag_names = tags_df["name"].tolist()
    general_mask = (tags_df["category"] == 0).values

    # Ưu tiên chạy CUDA GPU, tự động lùi về CPU nếu thiếu driver
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"⚡ Đang chạy trên: {session.get_providers()[0]}")
    input_name = session.get_inputs()[0].name
    return session, input_name, tag_names, general_mask


def fetch_and_preprocess_image(url: str) -> np.ndarray | None:
    """Tải ảnh từ URL và chuẩn hóa về tensor (1, 448, 448, 3) BGR cho WD14."""
    url = str(url).strip()
    if not url or url.lower() == "nan":
        return None
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            img = Image.open(BytesIO(resp.read())).convert("RGBA")
            canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img)
            rgb = canvas.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
            return np.expand_dims(arr, axis=0)
    except Exception:
        return None


def main():
    # Xác định file CSV đầu vào
    csv_path = INPUT_CSV
    if not csv_path:
        candidates = [f for f in sorted(glob.glob("*.csv")) if f != OUTPUT_CSV]
        if not candidates:
            raise FileNotFoundError("Không tìm thấy file .csv nào trong thư mục hiện tại!")
        csv_path = candidates[0]

    print(f"📂 Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    if "image_url" not in df.columns:
        raise ValueError("File CSV thiếu cột 'image_url'!")

    session, input_name, tag_names, general_mask = load_wd14_model()
    urls = df["image_url"].fillna("").tolist()

    # Tải ảnh đa luồng (24 luồng) để theo kịp tốc độ xử lý của GPU
    print("🌐 Đang tải ảnh poster...")
    with ThreadPoolExecutor(max_workers=24) as executor:
        preprocessed_images = list(
            tqdm(executor.map(fetch_and_preprocess_image, urls), total=len(urls), desc="Tải ảnh")
        )

    BATCH_SIZE = 32
    ART_THRESHOLD = 0.15  # Ngưỡng nhạy cho nhóm CORE_AESTHETIC_TAGS

    artstyle_col = [""] * len(preprocessed_images)
    valid_items = [(i, arr[0]) for i, arr in enumerate(preprocessed_images) if arr is not None]

    print(f"🎨 Đang phân tích {len(valid_items)} ảnh bằng RTX 3060 (chỉ lấy CORE_AESTHETIC_TAGS)...")
    for start in tqdm(range(0, len(valid_items), BATCH_SIZE), desc="GPU Inference"):
        batch_slice = valid_items[start : start + BATCH_SIZE]
        batch_indices = [item[0] for item in batch_slice]
        batch_tensor = np.stack([item[1] for item in batch_slice], axis=0)

        batch_probs = session.run(None, {input_name: batch_tensor})[0]

        for orig_idx, probs in zip(batch_indices, batch_probs):
            pure_art = [
                (tag_names[idx], float(p))
                for idx, p in enumerate(probs)
                if general_mask[idx] and p >= ART_THRESHOLD and (tag_names[idx] in CORE_AESTHETIC_TAGS)
            ]
            pure_art.sort(key=lambda x: x[1], reverse=True)
            artstyle_col[orig_idx] = "|".join([t for t, _ in pure_art[:8]])

    df["artstyle_tags"] = artstyle_col
    # Xóa bỏ cột visual_tags_full nếu có từ lần chạy trước để không làm nhiễu TF-IDF
    if "visual_tags_full" in df.columns:
        df.drop(columns=["visual_tags_full"], inplace=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    success_count = sum(1 for x in artstyle_col if x)
    print(f"\n✅ Hoàn tất! Đã lưu file mới tại: {OUTPUT_CSV}")
    print(f"📊 Số anime có ít nhất 1 Core Aesthetic Tag: {success_count}/{len(df)}")

if __name__ == "__main__":
    main()