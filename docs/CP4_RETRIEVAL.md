# CP4 — Dense, Lexical BM25 và Hybrid RRF trên corpus Lab 6

**Trạng thái: hoàn thành triển khai và smoke thực tế (14/11/2026).** Chưa chạy benchmark đầy đủ trên dev; chưa chạy/chấm tập test khóa tại CP2. Số liệu smoke không được coi là điểm benchmark.

## 1. Ba phương pháp retrieval

Cả ba dùng cùng corpus *Think Python 2e*, chương 1–4, 35 trang nội dung và 160 chunk paragraph-B ở CP1, cùng `chunk_id`, `source_id`, URL, trang PDF và metadata chapter. Tuyệt đối không cấp `gold`, `expected_answer`, `absence_reason` cho retriever/Agent.

| Phương pháp | Kỹ thuật | Chỉ số/đầu ra | Tài nguyên |
|---|---|---|---|
| `dense` | `all-MiniLM-L6-v2`, 384 chiều, normalize; pgvector cosine trên index B Lab 6 | score = cosine similarity, rank theo score | PostgreSQL + CPU query embedding |
| `lexical` | BM25 (Robertson, `k1=1.5`, `b=0.75`), tokenizer Unicode NFC/casefold, in-memory trên 160 chunk gốc | `raw_score` BM25 và score transform `raw/(1+raw)` để phù hợp interface Agent; rank vẫn không đổi | Python standard library, không gọi embedding/DB |
| `hybrid` | Lấy candidate từ Dense và Lexical, mặc định mỗi bên 12; hợp nhất bằng Reciprocal Rank Fusion `k=60` | `rrf_score = Σ1/(60+rank)`; score chuẩn hóa so với hai rank-1, lưu rank và component score | Dense + BM25, không cộng score gốc của hai thang khác nhau |

**Lưu ý score:** score của các phương pháp dùng thang khác nhau, không so sánh trực tiếp trị số cosine/BM25/RRF giữa các cách. Muốn so sánh chất lượng phải dùng cùng benchmark và cutoff Hit@1/Evidence Hit@3 trên dev ở CP5. Hybrid dùng rank-based RRF nên không đòi hỏi tự ý cộng cosine và BM25.

Các phương pháp đều hỗ trợ `search(query, filters, top_k)`, tôn trọng chapter filter hợp lệ. Kết quả có `id`, `text`, `source_id`, `source`, `page`, `metadata`, `score`, phương pháp (Dense cung cấp qua CLI) và trace. Hybrid còn có `ranks`, `component_scores` và `rrf_score`. Các thao tác DB là SELECT/read-only.

## 2. Triển khai và kết nối Agent

- `src/retrieval.py`: BM25, Hybrid RRF, tokenizer, load corpus, strategy factory.
- `src/day6_adapter.py`: `Day6DenseRetriever` dùng lại không thay đổi index Lab 6.
- `scripts/cp3_run.py` đã có `--strategy dense|lexical|hybrid` (mặc định `dense`), `--candidate-k`, `--top-k`, `--max-searches`, `--context-chars`; vẫn **chỉ ad-hoc hoặc dev, không hỗ trợ test**.
- `src/agent.py` giữ nguyên cơ chế Gemini tool calling, context budget, citation/abstention và trace. `retrieval_strategy` được ghi ở trace CLI; Hybrid ghi thêm `rrf_k` và `rrf_candidate_k`.

## 3. Kết quả smoke trên truy vấn không thuộc benchmark

Chạy hai truy vấn mô tả Python expression và turtle graphics, mỗi truy vấn 3 phương pháp/top-3; truy vấn turtle có chapter filter. Trong `artifacts/cp4_smoke.json` lưu 6 lần retrieval thật, top-3 gồm ID, rank, score, trang, nguồn, filter và thời gian.

| Smoke case | Dense | Lexical BM25 | Hybrid RRF | Kiểm tra |
|---|---|---|---|---|
| SMOKE-01: evaluate numerical expression | Có top-3 | Có top-3 | Có top-3 | Source ID và trang hợp lệ |
| SMOKE-02: turtle move/turn, filter chapter 4 | Có top-3 | Có top-3 | Có top-3 | Cả 9 kết quả thuộc chapter 4 |

Ở SMOKE-02, Dense và Hybrid tìm các chunk liên quan đến turtle tại trang 52–53; kết quả đứng đầu của cả 3 phương pháp là cùng chunk `paragraph-6b3863c3810c03f5e917` ở trang 52. Kết quả thay đổi theo phương pháp là bình thường, chưa kết luận cái nào tốt hơn.

**Thời gian:** SMOKE-01 Dense mất khoảng 29,42 giây do cold model loading; Lexical 1,18 ms; Hybrid 48,57 ms sau khi Dense đã tải model. SMOKE-02 Dense 55,18 ms, Lexical 0,64 ms, Hybrid 47,25 ms. Đây là số đo tại máy ở các lượt smoke, không tách đầy đủ cold/warm hay đủ lặp để so sánh thống kê.

## 4. Smoke tích hợp Agent dùng Gemini thật

Dùng cùng câu dev D7-D06 (định nghĩa biến ở chương 2):

| Strategy | Status | Search calls | LLM calls | Wall time | Evidence |
|---|---|---:|---:|---:|---|
| `dense` (CP3) | `answered` | 1 | 2 | 15,88 s | PDF trang 31 và 36 |
| `lexical` | `answered` | 1 | 2 | 2,77 s | PDF trang 36 |
| `hybrid` | `answered` | 1 | 2 | 15,77 s | PDF trang 31 và 36 |

Các citation ID trong context đã được kiểm tra cú pháp và nguồn/trang, chưa thay thế việc đọc kiểm tra đầy đủ chất lượng answer/grounding. Không dùng ba lượt riêng biệt để tính tỷ lệ thắng hay hiệu năng tổng quát.

Trace có thể đối chiếu trong `artifacts/cp4_agent_smoke/lexical_D7-D06.json`, `artifacts/cp4_agent_smoke/hybrid_D7-D06.json` và trace Dense CP3 tại `artifacts/cp3_smoke/D7-D06.json`. File trace gốc của các lần chạy vẫn nằm trong `starter/results/cp4/` (Git-ignored); export bằng `scripts/cp4_export_agent_traces.py`.

## 5. Tái lập (Git Bash Windows)

```bash
# Khởi động PostgreSQL Day 6 nếu chưa bật
cd /e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter
docker compose -p day6_student_starter up -d --wait

cd /e/code/Vin/lab7/K5-L3A-Day07-NguyenVanMinh-2A202603551-RetrievalContext
# Real retrieval-only (2 câu smoke; không dùng LLM hoặc test)
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp4_smoke.py

# Real Gemini Agent, DEV ONLY. Dùng .env cục bộ của Lab 7.
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp3_run.py --strategy lexical --dev-id D7-D06
/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe scripts/cp3_run.py --strategy hybrid --dev-id D7-D06

PYTHONUTF8=1 py -m unittest discover -s tests -v
py scripts/cp2_validate.py
```

## 6. Giới hạn và bước tiếp theo

Tokenization BM25 theo Unicode đơn giản, không có stemming hoặc tách từ ngữ nghĩa tiếng Việt; corpus và câu hỏi hiện bằng tiếng Anh. RRF candidate_k=12 và các tham số khác là baseline, **chưa được tối ưu theo dev**. Chưa đánh giá Hit@1, Evidence Hit@3, context evidence, câu có/thiếu evidence và grounding trên toàn bộ benchmark. Trong CP5 phải chạy ít nhất ba phương pháp trên cùng 15 câu dev có evidence và hai câu thiếu evidence, tính metric theo từng cấu hình, lưu trace, đánh giá lỗi và thận trọng với sai lệch do các truy vấn được LLM viết lại. Chỉ chốt trên dev và sau đó mới chạy test tại CP7.
