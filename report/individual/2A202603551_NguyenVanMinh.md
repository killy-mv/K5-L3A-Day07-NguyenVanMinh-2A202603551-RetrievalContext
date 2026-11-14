# Báo cáo cá nhân Day 7 — Nguyễn Văn Minh (2A202603551)

**Trạng thái:** CP1–CP6 đã triển khai. CP5 có 51 retrieval + 51 Gemini Agent trên 17 dev x 3 phương pháp. CP6 đã thử E1 Weighted RRF (giảm hiệu quả), E2 Gemini completeness prompt (bị lỗi HTTP 503/timeout, không thể chấm), và E2b mở rộng Dense context (cải thiện một phần). **Cấu hình Hybrid RRF 1:1 đã được khóa trước CP7; chưa chạy test.** Nhãn grounding/citation CP5 vẫn là bản rà soát hỗ trợ AI, chưa có xác nhận độc lập từ người học.

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Allen B. Downey, *Think Python 2e*, https://greenteapress.com/thinkpython2/thinkpython2.pdf, CC BY-NC 3.0; 35 trang nội dung (chương 1–4), 160 chunk kiểu paragraph B (tối đa 650 ký tự), đối chiếu nguồn/metadata với DB 160/160 trong CP1.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16 + pgvector 0.8.7 (chiến lược B của Lab 6); sentence-transformers/all-MiniLM-L6-v2 (revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), 384 chiều, normalize, cosine similarity.
- Agent model, cách chạy và phân chia dev/test: CP3 đã kết nối Gemini `gemini-3.5-flash-lite` qua Gemini REST function calling + pgvector read-only, tham số mặc định top-k=3, tối đa 3 lượt search, context budget=3800 ký tự. CP2 khóa 17 dev (15 có đáp án, 2 thiếu evidence) và 7 test mới (5 có đáp án, 2 thiếu evidence); không dùng lại 5 câu test Lab 6 đã xem kết quả.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: 20 câu có evidence trích nguyên văn chunk nguồn/trang; 4 câu thiếu evidence được kiểm tra tự động theo phạm vi 4 chương đã index; kết luận vắng mặt theo ngữ nghĩa vẫn cần người học rà soát theo tài liệu CP2. Hit@1/Evidence Hit@3 chỉ tính 20 câu có đáp án theo split; Context Evidence Rate đo bằng chứng trong context; Agent đánh giá grounded 0/1/2, citation có hỗ trợ và abstention. CP2 audit xác minh gold theo source/page/chunk, nhưng độ đầy đủ ngữ nghĩa và các nhãn Agent CP5 vẫn cần người học xác nhận. Không đánh giá test để chọn cấu hình.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline Dense | MiniLM 384-D + pgvector cosine, top-3, budget=3.800 chars | Mốc so sánh CP5 | dev | 10/15 (66,7%) | 13/15 (86,7%) | 13/15 (86,7%) | Mean 473,381 ms; không dùng API embedding; chưa tính chi phí CPU |
| Phương pháp 2 — BM25 | BM25 k1=1.5, b=0.75, top-3, budget=3.800 chars | Tìm từ khóa chính xác tốt hơn semantic với tên hàm/keyword | dev | 12/15 (80,0%) | 14/15 (93,3%) | 14/15 (93,3%) | Mean 0,420 ms; không dùng API embedding; chưa tính chi phí CPU |
| Phương pháp 3 — Hybrid RRF | Dense+BM25; 12 candidates/nhánh, RRF k=60, top-3, budget=3.800 chars | Bù trừ lỗi Dense/lexical | dev | 12/15 (80,0%) | **15/15 (100%)** | **15/15 (100%)** | Mean 33,326 ms; không dùng API embedding; chưa tính chi phí CPU |
| CP6 E1 — Weighted RRF | Dense 0.7 / BM25 1.3, RRF k60, top-3, context3800 | Giả thuyết trọng số lexical lớn sẽ tốt hơn; **bị bác bỏ** trên dev | dev | 11/15 (73,3%) | 14/15 (93,3%) | 14/15 (93,3%) | Kém Hybrid 1:1 |
| CP6 E2b — Dense context expansion | Dense top-5, context5000 | Thêm gold vào context; Hit@5=14/15, nhưng Hit@3 không đổi | dev | 10/15 (66,7%) | 13/15 (86,7%) | 14/15 (93,3%) | Nhiều context hơn, chưa vượt Hybrid |
| **Đã khóa cho CP7** | Hybrid RRF 1:1, top-3, context3800, CP3 prompt | Quyết định theo dev, checksum cấu hình đã khóa | test | **Chưa chạy** | **Chưa chạy** | **Chưa chạy** | — |
| Baseline Dense | Chưa mở test | Chỉ chạy khi đã freeze lựa chọn | test | **Chưa chạy** | **Chưa chạy** | **Chưa chạy** | — |

Định nghĩa chỉ số: 15 câu có evidence ở dev làm mẫu số, không tính 2 câu unanswerable. `Evidence Hit@3` yêu cầu đúng source, trang PDF và đoạn nguyên văn trong một chunk thuộc top-3; `Context Evidence Rate` tính evidence nguyên vẹn trong context giới hạn 3.800 ký tự từ câu hỏi gốc. Thời gian mean retrieval có cold-load nên không phải phép đo cân bằng cold/warm. Đơn giá Gemini chưa đối chiếu, chi phí Agent báo `null`. Tham khảo [CP5 chi tiết](../../docs/CP5_EVALUATION.md).

**Lỗi dev quan sát thực:** D7-D07 và D7-D12 có gold nằm ngoài top-3 Dense nhưng có trong BM25/Hybrid; D7-D01 BM25 không tìm được gold top-3 còn Dense/Hybrid tìm được. Giả thuyết: BM25 có lợi cho keyword cụ thể và Dense có lợi cho diễn đạt khái niệm, Hybrid RRF kết hợp hai thế mạnh. Kết quả Hybrid 15/15 chỉ phản ánh **dev**, không khẳng định chất lượng test. Sau các thí nghiệm CP6 đã quyết định và **khóa Hybrid RRF 1:1** tại `artifacts/cp6/final_config_lock.json` trước CP7.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: CP3 dùng `search_documents(query)`, bắt buộc search đầu tiên, tối đa 3 lượt; sắp xếp score/loại trùng để ghép context không cắt chunk, kiểm tra citation IDs với context; lỗi API/DB ghi `error`, không chấm thành abstention. Smoke dev D7-D06: `answered`, 1 search, 2 LLM calls, 15.88 giây, citation trang PDF 31 và 36. Dev D7-D16: `abstain`, 2 searches, 3 LLM calls, 17.08 giây. Đây là 2 case riêng lẻ, chưa phải benchmark hay đánh giá grounding đầy đủ.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense / dev | 17/17 | 22 searches, 39 LLM calls | 0:0, 1:2, 2:13; **1,867/2 nháp AI** | 20/21 | 15/15 | 2/2 | 0/15 |
| BM25 / dev | 17/17 | 22 searches, 39 LLM calls | 0:0, 1:2, 2:13; **1,867/2 nháp AI** | 21/21 | 15/15 | 2/2 | 0/15 |
| Hybrid / dev | 17/17 | 19 searches, 36 LLM calls | 0:0, 1:2, 2:13; **1,867/2 nháp AI** | 20/20 | 15/15 | 2/2 | 0/15 |

Những nhãn `grounded answer`, `citation support` và abstention trên do AI hỗ trợ rà từng câu và nguồn, có giải thích trong `artifacts/cp5/agent_*_reviews_assisted.json`. Các file ghi **`human_confirmed=false`**. D7-D03 thiếu nhắc phép XOR `^` và D7-D14 thiếu đơn vị góc quay (độ), chấm bảo thủ 1/2. Dense D7-D04 có một citation chỉ hỗ trợ loại dữ liệu chung, không trực tiếp chứng minh hai giá trị cụ thể nên tính 20/21. Các chỉ số này phải được người học xác nhận nếu yêu cầu chấm độc lập.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| D7-D01 | Answerable, dev | BM25 query gốc không có gold top-3; Dense/Hybrid có | BM25 Agent viết lại query, trả định nghĩa program có nguồn | Phân biệt lỗi 1-turn retrieval với multi-turn Agent; xem trace |
| D7-D07 | Answerable, dev | Dense top-3 không có gold, BM25/Hybrid có | Agent Dense truy vấn lại, trả lời class là reserved keyword | Từ khóa `class` phù hợp lexical; Hybrid bù lỗi Dense |
| D7-D12 | Answerable, dev | Dense query gốc không có gold | Agent trả lời dot notation nhờ đoạn glossary/ví dụ khác | Exact quote thiếu chưa khẳng định answer sai; cần đọc nguồn |
| D7-D03 | Answerable, dev | Cả ba có gold | Agent nhắc `**` nhưng không giải thích `^` XOR | Nhãn nháp 1/2, thiếu chi tiết expected |
| D7-D14 | Answerable, dev | Cả ba có gold | Agent nêu bk/lt/rt, thiếu góc quay tính bằng độ | Nhãn nháp 1/2, thiếu chi tiết expected |
| D7-D16 và D7-D17 | Unanswerable, dev | Agent tìm 1–3 lượt rồi từ chối | Không tạo citation giả | Nhãn nháp abstention 2/2 mỗi phương pháp |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: `artifacts/cp5/agent_{dense,lexical,hybrid}_traces.json`, `agent_*_reviews_assisted.json`, `agent_*_summary.json`, `retrieval_*_traces.json`, `provider_error_audit.json`; lệnh tái lập tại `docs/CP5_EVALUATION.md`. Gemini tổng 97.972 tokens do provider báo cáo từ 51 lượt cuối (không gồm lượt lỗi 429), API billing USD chưa xác minh nên `api_cost_usd=null`.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn: Dense baseline Hit@3 13/15; BM25 14/15 và Hybrid 15/15 trên dev. 6 lượt BM25 ban đầu lỗi HTTP 429, đã retry và lưu trace lỗi riêng, không tính thành hallucination/abstention. Hai lỗi Dense D7-D07/D7-D12 và một lỗi BM25 D7-D01 được phân tích ở CP5. Model có query rewrite nên Agent context exact-gold có thể khác original-query retrieval. Grounding nháp AI cần xác nhận độc lập; test chưa chạy; CP6 đã thử E1/E2b, ghi nhận E2 Gemini prompt bị HTTP 503/timeout và khóa cấu hình Hybrid cân bằng trước CP7.

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


## CP6 — Hai giả thuyết cải thiện, phân tích lỗi và khóa cấu hình

1. **E1 Weighted RRF**: đặt trọng số Dense=0.7, BM25=1.3, giữ RRF k=60, candidate_k=12, top_k=3, context 3.800 ký tự. Trên 15 câu answerable dev, Hit@1 giảm **12/15 → 11/15**, Evidence Hit@3 giảm **15/15 → 14/15**, Context Evidence Rate giảm **15/15 → 14/15** so với Hybrid 1:1. Trường hợp D7-D01 (định nghĩa *program*) trượt gold top-3. Kết luận: bác bỏ giả thuyết lexical trọng số lớn cải thiện Hybrid trong tập dev này.
2. **E2 Gemini completeness prompt**: thiết kế và chạy prompt chung yêu cầu trả lời đầy đủ các vế/điều kiện và đơn vị trong câu hỏi, vẫn chỉ dựa trên evidence và abstain khi thiếu nguồn. Tuy nhiên bốn lượt dev được thử đều lỗi HTTP 503 hoặc timeout, không có answer hợp lệ. **Không có điểm grounded/citation cho E2**; lỗi provider được lưu minh bạch tại `artifacts/cp6/e2_prompt_provider_audit.json`.
3. **E2b bổ sung sau sự cố provider**: thay Dense top-3/context 3.800 bằng top-5/context 5.000. Hit@1 **10/15** không đổi; Hit@3 **13/15** không đổi; Hit@5 **14/15**; Context Evidence Rate tăng **13/15 → 14/15**, do D7-D07 có thêm bằng chứng trong context; D7-D12 vẫn thiếu. E2b là thử nghiệm retrieval bổ sung, **không phải kết quả của E2 prompt** và không có điểm Gemini Agent.
4. **Quyết định trước test:** chọn **Hybrid RRF cân bằng 1:1**, RRF k60, candidate_k12, top_k3, context budget3.800, Gemini `gemini-3.5-flash-lite` với prompt CP3 gốc, max_searches3. Cấu hình, mã nguồn và benchmark SHA256 đã khóa tại `artifacts/cp6/final_config_lock.json`. `py scripts/cp6_freeze.py` đã kiểm chứng lại mà không ghi đè. Không sử dụng test để quyết định.
5. **Đánh đổi và giới hạn:** tăng trọng số lexical có thể làm mất trường hợp biểu đạt khái niệm; tăng số chunk context giúp tìm thêm bằng chứng nhưng có thể tăng token và nhiễu. Kết quả trên dev n=15 nhỏ, chưa chứng minh tổng quát hóa. Nhãn grounded/citation CP5 là AI-assisted draft, chưa được xác nhận độc lập. Tài liệu và trace: `docs/CP6_IMPROVEMENTS.md`, `artifacts/cp6/`. **CP7 test vẫn chưa chạy**.
