# AniMatch 2.2 — Franchise Graph, Trailer, Taste Expansion

**Mới:** xem [`UPGRADE_2_2_VI.md`](UPGRADE_2_2_VI.md) để áp dụng bản nâng cấp AniList PREQUEL, YouTube search/embed và Genres/Themes/Demographics.

# AniMatch 2.0 — Full-stack TypeScript rebuild

**React + Vite + TypeScript · Fastify (Node.js) · PostgreSQL · TypeScript recommendation engine**

**Trạng thái:** Bản source end-to-end để tiếp tục kiểm thử, **chưa được xác nhận production-ready**. Thiết kế dựa trên Streamlit AniMatch MVP và 12.156 anime của ZIP gốc. Không chứa Python runtime và không sử dụng Streamlit.

## 1. Chạy trên Windows / macOS / Linux

**Cần:** Node.js 22+ (nên dùng LTS được hỗ trợ), npm, Docker Desktop (hoặc PostgreSQL 17+ cài sẵn), PowerShell/terminal. Máy cần Internet để cài package lần đầu.

Tại thư mục `AniMatch-2.0`:

```bash
cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env

docker compose up -d
npm install
npm run db:init
npm run db:import
npm run dev
```

- Frontend: **http://localhost:5173**
- Backend: **http://localhost:3001**
- API health: **http://localhost:3001/api/health**

**Lưu ý thứ tự:** phải `db:init` + `db:import` trước khi khởi động API; backend load anime và lập vector khi khởi động.

**Nếu PostgreSQL chưa sẵn sàng:** kiểm tra `docker compose ps`, `docker compose logs postgres`, rồi chạy lại lệnh DB.

## 2. Những thứ có trong source

| Phần | Tính năng |
|---|---|
| Frontend | Onboarding, tìm kiếm anime, lọc Genre, Swipe Like/Pass/Watched, Undo, Watchlist, Library, sửa gu, trang Benchmark giới thiệu |
| Backend | REST API, xác thực bằng token ẩn danh, hồ sơ và lịch sử độc lập mỗi session, watchlist, ghi recommendation events |
| PostgreSQL | Catalogue anime, profile, interactions, watchlist, recommendation history |
| Engine TypeScript | Sparse TF-IDF (genre/theme, synopsis 1–2 grams, studio, artstyle), Rocchio, quality scoring, MMR, studio cap, franchise prerequisite gate |
| Importer | Parser CSV hỗ trợ quoted multiline, nhập dataset cũ, dùng franchise_sorted_mal_ids để tạo thứ tự |
| Benchmark | 6 kịch bản (2 Eye-candy, 2 Content-first, 2 Lapsed), nhiều seed/lượt, CSV từng anime |
| Test | Golden-style engine unit tests cơ bản; cần mở rộng parity tests và E2E |

## 3. Test & benchmark

```bash
npm run typecheck
npm test
SEEDS=10 STEPS=30 npm run benchmark
```

Trên Windows PowerShell:

```powershell
$env:SEEDS='10'; $env:STEPS='30'; npm run benchmark
```

Kết quả nằm trong `benchmarks/results/sequential.csv`, có từng lượt, tên anime, Genre, artstyle, phản hồi synthetic và các điểm chấm.

*Đây là benchmark mô phỏng*, không phải bằng chứng người dùng thật hài lòng. Lapsed User mô phỏng thời điểm trở lại, **chưa** có suy giảm theo thời gian thực trong engine.

## 4. Cấu trúc

```text
apps/web/                 React + Vite + CSS responsive
apps/api/                 Fastify REST API
packages/engine/          Sparse TF-IDF + Rocchio + MMR + guard
packages/shared/          Shared TypeScript contracts
database/schema.sql       PostgreSQL schema
scripts/import-anime.ts   CSV -> PostgreSQL
benchmarks/run.ts         Scenario simulation
benchmarks/results/       Generated CSV outputs
tests/                    Algorithm tests
data/anime.csv            Dataset gốc
```

## 5. Khác biệt quan trọng với engine Python gốc

1. **Không phải bản port tương đương bit-for-bit.** Scikit-learn TF-IDF cũ dùng stopwords, giới hạn `min_df`, `max_df` và `max_features=3500`; bản TypeScript dùng sparse TF-IDF viết lại, tokenization giản lược và giới hạn đoạn synopsis. Kết quả xếp hạng **có thể khác**.
2. Franchise relation được dựng từ `franchise_sorted_mal_ids` của CSV. CSV này không có toàn bộ trường `prereq_anime_id` đã enrich trong runtime Python. Cần so sánh và kiểm tra thứ tự riêng ở franchise phức tạp.
3. Auto-Pivot của Streamlit cũ **chưa port chính xác 1:1**: feedback Rocchio có, nhưng chính sách tự xoay gu theo lịch sử chưa có.
4. Engine hiện dùng một bộ dữ liệu vector dựng một lần vào memory, chạy thuật toán CPU tại Node.js process. Chưa có worker thread / search index chuyên dụng. Cần benchmark latency thực tế dưới tải đồng thời.
5. Auth hiện là **anonymous bearer session** lưu trong browser localStorage, **không phải** tài khoản có email/mật khẩu, không phù hợp dữ liệu nhạy cảm. Đổi trình duyệt hoặc mất token sẽ mất quyền truy cập hồ sơ cũ. Không có account recovery.
6. Không có chính sách HTTP rate limit / monitoring / deployment cloud production trong source hiện tại.
7. `artstyle_tags` vốn chưa là một subspace độc lập trong engine Streamlit; bản TypeScript **chủ động mở rộng** artstyle, do đó parity sẽ khác ở kịch bản Eye-candy.

## 6. Lộ trình kiểm thử trước final

- Chạy `npm install`, `npm run typecheck`, `npm test`, rồi chạy trên browser desktop/mobile.
- Chạy kiểm thử luồng onboarding → Like → refresh → Watched → Undo → Library → Watchlist.
- Chạy benchmark 6 kịch bản 10 seed, 30 lượt, đọc CSV và rà soát anime + genre.
- Bổ sung parity tests dùng cùng hồ sơ + history (Python MVP vs TypeScript) trước khi tuyên bố port đầy đủ.
- Không deploy trước khi xác nhận access control, rate limits, DB backups, origin config và latency thực tế.

## 7. Troubleshooting

- **`ECONNREFUSED 5432`**: PostgreSQL chưa chạy hoặc `.env` sai.
- **`relation anime does not exist`**: quên `npm run db:init`.
- **`Engine initialized: 0`**: quên `npm run db:import`, restart API sau khi import.
- **npm install EAI_AGAIN**: lỗi DNS/Internet, kiểm tra mạng rồi chạy lại.
- **Failed to fetch**: API chưa chạy hoặc `VITE_API_URL` / `WEB_ORIGIN` chưa đúng.

## 8. Scope và bảo mật

Phát triển cho MVP trong môi trường tin cậy. Các endpoint ngoại trừ `health`, `options` và tạo session yêu cầu Bearer token. Token được sinh bằng CSPRNG và chỉ lưu SHA-256 server-side. API không dùng global mutable user profile nên nhiều người dùng không chia sẻ history. Không có tài khoản đăng nhập thực; không dùng trong môi trường chứa dữ liệu nhạy cảm.

## AniMatch 2.1 — UX update

Mode-first onboarding, swipe gestures, live polygon vibe radar, searchable corpus-backed visual vocabulary, YouTube trailer embedding and on-demand larger posters. See [UPDATE_2_1_VI.md](UPDATE_2_1_VI.md).

## AniMatch 2.3 update

See [`UPGRADE_2_3_VI.md`](UPGRADE_2_3_VI.md). Requires re-running both `npm run db:init` and `npm run db:import` to populate release date, airing status, and episode count for existing anime. The UI now provides mode-specific onboarding, 2-tab signed taste radar, auto-pivot, and requeue/library transition actions.

## AniMatch 2.4: Full Detail & Recommendation Quality

Xem [`UPGRADE_2_4_VI.md`](./UPGRADE_2_4_VI.md) để biết cách cập nhật, bộ lọc số tập xuyên suốt, chính sách rating 8+, visual style radar và tính năng Pick 1 Tonight.
