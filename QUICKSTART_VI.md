# Chạy AniMatch 2.0 — Hướng dẫn nhanh

## Cài một lần

1. Cài Node.js 22+, Docker Desktop, Git và trình soạn thảo VS Code.
2. Giải nén ZIP vào thư mục project; mở Terminal tại thư mục gốc.
3. Tạo `.env` bằng cách sao chép `.env.example`.
4. Chạy từng lệnh theo đúng thứ tự:

```bash
docker compose up -d
npm install
npm run db:init
npm run db:import
npm run dev
```

5. Mở `http://localhost:5173` và chọn một trong ba mode Eye-Candy First / Content First / Lapsed User. Genre và artstyle có thể bỏ trống; hệ thống học dần từ lượt quẹt.

## Dùng các chức năng chính

- **Discover:** tìm kiếm anime, lọc genre, mở chi tiết, lưu watchlist.
- **Swipe:** kéo thẻ sang phải để Like, sang trái để Pass, lên trên để đánh dấu Watched. Có Undo, Vibe Radar cập nhật sau Like, và trailer YouTube khi nguồn có video. (Có thể dùng phím ←, →, ↑ thay cho thao tác kéo.)
- **Library:** danh sách Like / Watched / Watchlist.
- **Preferences:** sửa mode, genre / artstyle; tìm kiếm toàn bộ tag hình ảnh có hỗ trợ từ dataset.
- **Benchmark:** hướng dẫn chạy 6 kịch bản.

## Chạy benchmark (riêng, trong terminal khác)

```bash
npm run benchmark
```

Hoặc tùy chỉnh:

```powershell
$env:SEEDS='10'
$env:STEPS='30'
npm run benchmark
```

Kết quả tại `benchmarks/results/sequential.csv` gồm thông tin từng lượt: anime, genre, artstyle, phản hồi giả lập, các điểm chấm và gu mục tiêu.

## Chạy kiểm thử

```bash
npm run typecheck
npm test
```

**Giới hạn:** Codebase này chưa được cài npm và chưa được chạy end-to-end trong môi trường tạo file, do registry npm không thể truy cập. Nếu gặp lỗi khi chạy, cần gửi log để sửa thực nghiệm. Engine TypeScript là bản viết lại có chủ đích, không cam kết cho ra thứ hạng hoàn toàn giống scikit-learn cũ.

## Cập nhật từ bản 2.0 trước đó

Sao lưu `.env` và project, chép đè source mới, chạy `npm install`, `npm run db:init`, `npm run dev`. Script `db:init` có migration bổ sung cột mode. Xem `UPDATE_2_1_VI.md` cho API mới, lưu ý ảnh poster và giới hạn kiểm thử.
