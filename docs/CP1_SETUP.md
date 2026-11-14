# CP1 — Tái sử dụng dữ liệu, embedding và vector store Lab 6

## Kết quả thực nghiệm (2026-11-14)

- **Nguồn:** Allen B. Downey, *Think Python: How to Think Like a Computer Scientist*, 2nd ed.; <https://greenteapress.com/thinkpython2/thinkpython2.pdf>.
- **Quyền sử dụng:** CC BY-NC 3.0, chỉ dùng trong phạm vi cho phép và giữ thông tin nguồn, tác giả; <https://creativecommons.org/licenses/by-nc/3.0/>.
- **Corpus:** 35 trang nội dung hợp lệ, các chương 1–4; chọn chiến lược B paragraph-aware từ Lab 6 gồm 160 chunks, tối đa 650 ký tự/chunk. PDF gốc có 244 trang; **không** tuyên bố index toàn bộ cuốn sách.
- **Index:** PostgreSQL 16 + pgvector 0.8.7, bảng `day6_embeddings` (strategy='B'), 160 hàng, cosine similarity và metadata theo `chapter`, `page`, `source`, `document_id`.
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 chiều, CPU, normalize vector; truy vấn phải dùng đúng model/revision và chế độ normalize.
- **Kiểm chứng CP1:** 160/160 bản ghi có nội dung/metadata khớp JSONL Lab 6; hash PDF khớp manifest; 2 truy vấn smoke thật trả đủ top-3, truy vấn có filter chương 4 chỉ trả chunk chương 4.
- **Nguyên tắc:** Không chạm tới gold benchmark khi indexing/smoke, không ghi DB; cấu hình Day 6 giữ nguyên. Dữ liệu CP1 xuất JSON ở `starter/data/chunks.json` (local, Git-ignored).

## Tái lập trên Windows / Git Bash

1. Mở Docker Desktop. Khởi động database của Lab 6 (giữ nguyên Docker volume của Lab 6):

```bash
cd /e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter
docker compose -p day6_student_starter up -d --wait
```

2. Trở lại Lab 7, chạy Python trong virtualenv Lab 6, nơi các dependency đã được cài:

```bash
cd /e/code/Vin/lab7/K5-L3A-Day07-NguyenVanMinh-2A202603551-RetrievalContext
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp1_prepare.py --smoke
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

Nếu đặt thư mục khác, đặt `DAY6_ROOT` tới thư mục gốc của Day 6, hoặc truyền `--day6-root`. `DATABASE_URL` đọc từ biến môi trường hoặc `.env` **gốc** Day 6. Không sao chép `.env` vào repo Lab 7 hay commit credentials.

## Files

| File | Vai trò |
|---|---|
| `src/day6_adapter.py` | Load chunk, kiểm tra provenance + parity DB, và dense retriever thật. Tất cả truy vấn SQL chỉ đọc. |
| `scripts/cp1_prepare.py` | Xác thực dữ liệu, xuất JSON, chạy smoke tùy chọn. |
| `starter/data/chunks.json` | Corpus format Day 7, 160 chunks (generated, Git-ignored). |
| `starter/results/cp1_summary.json` | Kết quả kiểm tra và truy vấn thật (generated, Git-ignored). |
| `tests/test_day6_adapter.py` | Unit test adapter và validation. |

## Scope chưa làm (chặng tiếp theo)

- CP2 tạo benchmark Lab 7 tối thiểu 17 dev (15 answerable, 2 unanswerable) và 7 test mới (5 answerable, 2 unanswerable). Bộ test Day 6 đã lộ kết quả nên không dùng lại làm held-out test của Lab 7.
- CP3 kết nối controller LLM thật và tool; CP4 trở đi hoàn thiện Lexical/Hybrid, benchmark và audit.
- Không tính hai câu smoke là điểm retrieval/benchmark, cũng không tính mock unit test là bằng chứng agent thực tế.
