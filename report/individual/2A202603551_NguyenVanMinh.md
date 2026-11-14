# Báo cáo cá nhân Day 7 — Nguyễn Văn Minh (2A202603551)

**Trạng thái:** CP1–CP3 hoàn thành triển khai; CP3 đã chạy 2 case dev thực tế. CP4+ và benchmark đầy đủ chưa thực hiện; các metric tổng hợp bên dưới vẫn để trống đến khi có số liệu thật.

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Allen B. Downey, *Think Python 2e*, https://greenteapress.com/thinkpython2/thinkpython2.pdf, CC BY-NC 3.0; 35 trang nội dung (chương 1–4), 160 chunk kiểu paragraph B (tối đa 650 ký tự), đối chiếu nguồn/metadata với DB 160/160 trong CP1.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16 + pgvector 0.8.7 (chiến lược B của Lab 6); sentence-transformers/all-MiniLM-L6-v2 (revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), 384 chiều, normalize, cosine similarity.
- Agent model, cách chạy và phân chia dev/test: CP3 đã kết nối Gemini `gemini-3.5-flash-lite` qua Gemini REST function calling + pgvector read-only, tham số mặc định top-k=3, tối đa 3 lượt search, context budget=3800 ký tự. CP2 khóa 17 dev (15 có đáp án, 2 thiếu evidence) và 7 test mới (5 có đáp án, 2 thiếu evidence); không dùng lại 5 câu test Lab 6 đã xem kết quả.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: 20 câu có evidence trích nguyên văn chunk nguồn/trang; 4 câu phủ định thuộc phạm vi không có trong 4 chương đã index, kiểm tra theo corpus và rà soát thủ công theo tài liệu CP2. Hit@1/Evidence Hit@3 chỉ tính 20 câu có đáp án theo split; Context Evidence Rate đo bằng chứng trong context; Agent đánh giá grounded 0/1/2, citation có hỗ trợ và abstention. Tất cả số liệu thực nghiệm agent chưa chạy.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | | | dev | | | | |
| Phương pháp 2 | | | dev | | | | |
| Phương pháp 3 | | | dev | | | | |
| Cải thiện 1 | | | dev | | | | |
| Cải thiện 2 | | | dev | | | | |
| Cấu hình đã chọn | | | test | | | | |
| Baseline | | | test | | | | |

Một cấu hình có thể đồng thời là phương pháp mới và cải thiện; giải thích rõ. Báo `n/N` và tỷ lệ, ghi đơn vị thời gian/chi phí, so sánh test sau khi chốt. Nêu hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: CP3 dùng `search_documents(query)`, bắt buộc search đầu tiên, tối đa 3 lượt; sắp xếp score/loại trùng để ghép context không cắt chunk, kiểm tra citation IDs với context; lỗi API/DB ghi `error`, không chấm thành abstention. Smoke dev D7-D06: `answered`, 1 search, 2 LLM calls, 15.88 giây, citation trang PDF 31 và 36. Dev D7-D16: `abstain`, 2 searches, 3 LLM calls, 17.08 giây. Đây là 2 case riêng lẻ, chưa phải benchmark hay đánh giá grounding đầy đủ.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: xem `docs/CP3_AGENT.md` và các file JSON có timestamp trong `starter/results/cp3/` (local, không commit). `api_cost_usd=null` vì chưa có giá đối chiếu; chưa tự nhận đã chấm nhãn grounding/citation thủ công cho toàn bộ dev/test.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:

## Thử thách nâng cao (nếu thực hiện)

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev:

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | | | |
| Ảnh, sơ đồ | | | |
| Biểu đồ | | | |
| Multi-hop | | | |
| Text | | | |
| Thiếu bằng chứng | | | |
| **Toàn bộ** | | | |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống:
- Giới hạn, cách tái lập và đường dẫn benchmark:

### CP3 — Minh chứng live dev (chưa phải full benchmark)

- `artifacts/cp3_smoke/D7-D06.json`: câu có evidence, 1 lần tìm, 2 lần gọi model, trạng thái `answered`, citation đến trang PDF 31 và 36.
- `artifacts/cp3_smoke/D7-D16.json`: câu ngoài corpus, 2 lần tìm, 3 lần gọi model, trạng thái `abstain`, không có citation giả.
- Các trace đầy đủ gồm query, tool calls, passages, context và mô hình; không chứa gold. Các thử nghiệm lỗi trước khi sửa parser/schema giữ trong thư mục local `starter/results/cp3/` (không đưa vào điểm thành công).
- Chưa kết luận tỷ lệ chất lượng retrieval/grounding trên toàn bộ dev/test; không tính unit test mock là số đo agent thực tế.

### CP4 — Triển khai ba phương pháp retrieval (chưa benchmark)

- `dense`: embedding MiniLM-L6-v2 384 chiều từ Lab 6, vector store PostgreSQL/pgvector (cosine, read-only).
- `lexical`: BM25 trên 160 chunks B, `k1=1.5`, `b=0.75`, tokenizer Unicode NFC/casefold. Trả ranking, normalized score, raw BM25 score.
- `hybrid`: RRF (`rrf_k=60`, candidate=12 mỗi nguồn), giữ ranks/components, không cộng trực tiếp raw score không cùng thang đo.
- Smoke retrieval thực tế `artifacts/cp4_smoke.json`: 2 câu hỏi không thuộc benchmark x 3 phương pháp, đủ top-3 với source, page, score và chapter filter chính xác.
- Smoke tích hợp Gemini thật trên dev D7-D06: lexical `answered`, 1 search/2 model calls, 2,77 giây, citation trang 36; hybrid `answered`, 1 search/2 model calls, 15,77 giây, citation trang 31/36. Trace nộp tại `artifacts/cp4_agent_smoke/`.
- Kết quả trên mới là kiểm tra triển khai, **không phải bảng điểm Hit@1/Hit@3 hoặc grounded answer trên toàn bộ dev/test**. Sẽ đo và phân tích ở CP5.
