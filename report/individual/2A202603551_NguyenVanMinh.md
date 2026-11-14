# Báo cáo cá nhân Lab 7 — Retrieval & Context

**MSSV:** 2A202603551
**Họ và tên:** Nguyễn Văn Minh
**Trạng thái:** Đã hoàn thành **CP1–CP7, phần cơ bản**; có benchmark dev/test, 3 retrieval methods, Gemini Agent thực tế, hai thí nghiệm cải tiến có số đo, lưu trace và kiểm thử. **Tập test cuối 7/7 thành công cho cả Dense và Hybrid** sau khi thay API key mới. Không thực hiện bonus nâng cao.

## 1. Dữ liệu, nguồn và benchmark (CP1–CP2)

- **Tài liệu:** Allen B. Downey, *Think Python: How to Think Like a Computer Scientist*, ấn bản 2. PDF: https://greenteapress.com/thinkpython2/thinkpython2.pdf. Giấy phép **CC BY-NC 3.0** (https://creativecommons.org/licenses/by-nc/3.0/), sử dụng cho mục đích học tập phi thương mại và ghi nguồn.
- **Phạm vi index:** Chương 1–4, **35 trang nội dung** trích từ tài liệu gốc, không nhận là đã index toàn bộ PDF. Chiến lược chunk B paragraph-aware của Lab 6 có **160 chunks**. Kiểm tra nguồn PDF theo SHA256 manifest và kiểm chứng **160/160 bản ghi** trùng nội dung/metadata trong PostgreSQL/pgvector. Metadata giữ `source_id`, `source`, `page`, `chapter`, `chunk_id`.
- **Vector store:** PostgreSQL 16 + pgvector 0.8.7, bảng `day6_embeddings`, strategy B, truy xuất bằng cosine; chỉ đọc, không ghi đè dữ liệu Lab 6.
- **Embedding:** `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 chiều, vector chuẩn hóa, chạy CPU.
- **Benchmark:** 24 câu hỏi: **dev 17** (15 có evidence, 2 thiếu) và **test 7** (5 có evidence, 2 thiếu). Reuse đúng 15 dev Lab6; tạo 7 test mới, không tái dùng 5 test Lab6 đã biết kết quả. Câu có evidence có đáp án mong đợi, `source_id`, trang PDF vật lý, `chunk_id`, quote đúng trong chunk. Câu thiếu evidence được xác định theo phạm vi bốn chương đã index. Phân chia dev/test có SHA256 khóa tại `benchmark/split_lock.json`, `artifacts/cp2_audit.json`.
- **Bảo vệ nhãn:** Agent chỉ nhận `id`, câu hỏi và filter hợp lệ; không đưa `gold`, `expected_answer` hoặc lý do absence vào prompt/tool. Benchmark chỉ dùng bởi evaluator sau retrieval/inference.

## 2. Thiết kế Agentic RAG (CP3–CP4)

Gemini **`gemini-3.5-flash-lite`** được chạy thật qua Gemini GenerateContent REST API, sử dụng native function calling `search_documents(query)`. Đây là **model thực tế trong cấu hình cuối, file `.env` hiện tại và toàn bộ trace benchmark CP5/CP7**. Trước đó từng thử Gemini 3.8 Flash như chẩn đoán nhưng không dùng vào điểm benchmark chính. Không lưu API key trong repo.

Agent bắt buộc tìm trước khi trả lời, có thể viết lại query và tìm tối đa ba lượt. Tool sử dụng một trong ba retriever trên cùng 160 chunks; các kết quả gồm id, score, text, nguồn, trang và metadata. Context loại trùng và ưu tiên score, đóng gói **nguyên chunk** đến giới hạn **3.800 ký tự**. Agent được yêu cầu trả lời dựa trên bằng chứng, trích dẫn dạng `[chunk-id]`, không tuân theo lệnh nhúng trong tài liệu; kiểm tra citation ID thuộc context trước khi chấp nhận. Khi evidence không đủ, Agent abstain; API/DB lỗi có `status=error`, không gán thành câu trả lời sai. Trace có query gốc, tool query, chunks, context, draft/final answer, citation, lượt gọi, latency, token, error class.

**Ba phương pháp retrieval:**

1. **Dense:** embedding MiniLM 384 chiều, pgvector cosine trên index B Lab6.
2. **Lexical:** BM25 in-memory, tokenizer Unicode NFC/casefold, k1=1.5, b=0.75, ranking theo BM25; score chuẩn hóa chỉ để thống nhất interface Agent.
3. **Hybrid:** lấy top-12 candidate từ mỗi nguồn, kết hợp thứ hạng RRF k=60 với trọng số Dense/BM25 1.0/1.0, trả top-3. Không cộng raw cosine và BM25 trên hai thang đo khác nhau.

## 3. Kết quả retrieval trên DEV (CP5)

| Phương pháp, tham số | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Mean retrieval |
|---|---|---:|---:|---:|---:|
| Dense, top-3, budget 3.800 | Dev | 10/15 = 66,7% | 13/15 = 86,7% | 13/15 = 86,7% | 473,381 ms |
| BM25, top-3, budget 3.800 | Dev | 12/15 = 80,0% | 14/15 = 93,3% | 14/15 = 93,3% | 0,420 ms |
| Hybrid RRF 1:1, top-3, budget 3.800 | Dev | 12/15 = 80,0% | **15/15 = 100%** | **15/15 = 100%** | 33,326 ms |

Hit@1 và Evidence Hit@3 kiểm tra đúng `source_id`, `page` và chuỗi evidence trong chunk trả về; Context Evidence Rate kiểm tra gold evidence nằm trọn trong packed context. Mẫu số **15 answerable**; 2 câu không có evidence không được tính vào các metric này. Latency chỉ là thời gian quan sát, Dense bị ảnh hưởng cold-start tải embedding model, không phải benchmark cold/warm công bằng.

**Agent trên Dev:** đã chạy đầy đủ 17 câu × 3 cấu hình = **51 lượt thực tế**. Cả ba đều trả lời 15/15 câu answerable, abstain đúng 2/2 câu unanswerable. Search calls lần lượt Dense/BM25/Hybrid: **22/22/19**; Gemini calls **39/39/36**. Review hỗ trợ AI chấm grounded trung bình **1,867/2** (13 câu grade 2 và 2 câu grade 1 trên mỗi phương pháp); citation hỗ trợ nháp tương ứng **20/21, 21/21, 20/20**. Những nhãn này có lý do chi tiết nhưng `human_confirmed=false`, chưa coi là đánh giá độc lập của người học.

**Phân tích lỗi Dev:** D7-D07 (keyword `class`) và D7-D12 (`dot notation`) Dense không tìm đủ gold trong top-3 query gốc; BM25 và Hybrid tìm được. Ngược lại D7-D01 (*what is a program*) BM25 bỏ sót gold trong top-3, Dense/Hybrid tìm được. D7-D03 Agent thiếu nêu `^` là XOR, D7-D14 thiếu nêu góc quay tính bằng độ; chấm nháp 1/2. Truy vấn Agent có thể được viết lại nên cần phân biệt kết quả query gốc với context Agent.

## 4. Hai thí nghiệm cải tiến trên DEV và khóa cấu hình (CP6)

Hai giả thuyết được ghi trước khi chạy tại `artifacts/cp6/experiment_plan.json`:

- **E1 Weighted RRF:** tăng trọng số BM25 lên 1.3 và giảm Dense về 0.7, giữ RRF k60, candidate12 và top-3. *Giả thuyết*: ưu tiên từ khóa đặc thù Python sẽ tăng Hit@1 và giữ Hit@3. **Kết quả âm tính:** Hit@1 **12/15 → 11/15**, Hit@3 và Context Evidence Rate **15/15 → 14/15**, D7-D01 trượt top-3. Bác bỏ E1.
- **E2 completeness prompt:** bổ sung yêu cầu chung để Gemini trả lời đủ từng ý/đơn vị được hỏi mà không phỏng đoán ngoài nguồn. Đã triển khai nhưng lúc đó API trả 503/timeout; **không bịa điểm hoặc tuyên bố thử nghiệm này cải thiện**.
- **E2b thay thế bổ sung (được công bố rõ, không phải kết quả E2):** Dense top-3/context3.800 thành **top-5/context5.000**. Hit@1 và Hit@3 không đổi (**10/15**, **13/15**), Hit@5 **14/15**, Context Evidence Rate **13/15 → 14/15**. Có thêm gold D7-D07 nhưng không vượt Hybrid, đồng thời tăng context/token.

**Quyết định chỉ dựa DEV:** giữ **Hybrid RRF cân bằng 1:1**: top-3, RRF k=60, 12 candidates/nhánh, budget=3.800; Gemini `gemini-3.5-flash-lite`, prompt CP3 gốc, temperature 0.1, tối đa ba searches. Toàn bộ lựa chọn được SHA256 khóa tại `artifacts/cp6/final_config_lock.json` **trước khi mở TEST**, script `scripts/cp6_freeze.py` đã xác minh không thay đổi.

## 5. Đánh giá TEST sau khi khóa (CP7)

**Đã chạy đủ 7 câu test cho cả Dense baseline và Hybrid đã chọn.** Các yêu cầu Gemini lỗi tạm thời trong quá trình chạy đã được retry sau khi thay key mới, **cùng model 3.5 và cấu hình**; trace lỗi cũ được lưu riêng, trace cuối đều hợp lệ. Không điều chỉnh model/prompt/retrieval dựa vào test.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate |
|---|---|---:|---:|---:|
| Dense baseline | Test | 3/5 = 60% | **5/5 = 100%** | **5/5 = 100%** |
| Hybrid RRF đã chọn | Test | 3/5 = 60% | **5/5 = 100%** | **5/5 = 100%** |

**Context grade 0/1/2 nháp theo gold source:** cả hai có **5/5 mức 2**, không có mức 0/1 trong năm câu answerable. Kết quả này dựa trên source/page/quote trùng đầy đủ và chưa thay thế independent human judgment.

| Agent metric cuối — 7 câu TEST | Dense | Hybrid |
|---|---:|---:|
| Lượt test hợp lệ | **7/7** | **7/7** |
| Có trả lời khi có evidence | **5/5** | **5/5** |
| Abstain đúng khi thiếu evidence | **2/2** | **2/2** |
| Từ chối nhầm | 0/5 | 0/5 |
| Có gọi tool retrieval | 7/7 | 7/7 |
| Số lượt search / gọi model | 10 / 17 | 10 / 17 |
| Grounded-answer grade nháp hỗ trợ AI | 5 câu mức 2/2 | 5 câu mức 2/2 |
| Có citation ở câu answerable | 5/5 | 5/5 |
| API error còn tồn tại trong trace cuối | **0** | **0** |
| Latency Agent trung bình (7 trace cuối) | 53,14 giây | 20,27 giây |
| Token Gemini báo cáo (7 trace cuối) | 15.528 | 15.705 |

**Đánh giá câu trả lời:** Năm câu answerable kiểm tra flow of execution, biến cục bộ, stack frame, interface của `circle(t,r)`, và quy trình encapsulation/generalization. Các câu trả lời có citation chunk/trang, đối chiếu nguồn phù hợp; hai câu ngoài corpus (`async def` và `yield from`) được abstain đúng. Review 0/1/2 và citation là **bản dự thảo do AI hỗ trợ**, có lý do và nguồn theo từng câu nhưng `human_confirmed=false`, chưa phải kết quả người chấm độc lập. Trên test nhỏ n=5, Hybrid không thắng Dense dù tốt hơn trên dev.

**Minh chứng:** `artifacts/cp7/retrieval_{dense,hybrid}_{summary,traces}.json`, `agent_{dense,hybrid}_{summary,traces,reviews_assisted}.json`, `context_{dense,hybrid}_review_assisted.json`, `docs/CP7_FINAL_EVALUATION.md`.

## 6. Thời gian, chi phí, độ tin cậy và tái lập

Trace lưu đầy đủ tool, model, câu hỏi, context, nguồn/trang, citation, latency và token. Sau khi thay key mới, ba lượt test từng gặp lỗi provider đã được gọi lại thành công **bằng cùng model 3.5 và cấu hình CP6 đã khóa**, có pacing tối đa 2 RPM theo từng request. Các lỗi cũ chỉ nằm trong lịch sử kiểm toán local, không tính thành hallucination hoặc abstention; trace cuối 7/7 hợp lệ.

Ước tính **giả định theo đơn giá trả phí** cho token ghi nhận ở CP5/CP7 khoảng **0,0531 USD**. Đây **không phải hóa đơn thực tế**, chưa cộng request lỗi không có usage, chi phí CPU/cache, ưu đãi hay free-tier. Đơn giá giả định, script tính và các giới hạn ở `artifacts/cp7/estimated_api_costs.json`; chi phí thực tế chưa xác minh.

Tái lập kết quả offline: `py scripts/cp2_validate.py`, `py scripts/cp6_freeze.py`, `py scripts/cp7_offline_summary.py`, `py scripts/cp7_context_review.py`, `PYTHONUTF8=1 py -m unittest discover -s tests -v`. Để chạy retrieval thực cần dựng PostgreSQL Lab6 theo `docs/CP1_SETUP.md`, không chia sẻ `.env`, mật khẩu hay API key.

**Giới hạn:** Chỉ index 35 trang và các câu hỏi Python cơ bản; chỉ 5 câu answerable test nên kết quả mẫu nhỏ, không suy ra production SLA. Nhãn grounded/citation/context cần người học kiểm tra độc lập nếu lớp yêu cầu. Không tune trên test. Nguồn CC BY-NC chỉ dùng cho mục đích phù hợp giấy phép và ghi đầy đủ tác giả/nguồn.

## 7. Thử thách nâng cao (tùy chọn)

Không tham gia bonus index tài liệu trên 100 trang với bảng, ảnh, biểu đồ. Bài làm này chỉ nộp **phần cơ bản**, không tự nhận điểm cộng 15 hoặc tạo kết quả giả cho bảng/ảnh/chart/multi-hop.

## 8. Kết luận

Đã hoàn thành pipeline Agentic RAG với Dense/BM25/Hybrid, Gemini tool calling, lựa chọn context, citation và abstention; chạy đủ benchmark dev, thử cải tiến có kết quả âm tính/dương tính cục bộ, khóa cấu hình trước test và hoàn tất **7/7 phản hồi test thực tế cho cả Dense và Hybrid**. Hybrid tăng độ phủ evidence trên dev, nhưng không vượt Dense ở test. Mã nguồn, benchmark, trace, phân tích lỗi, thời gian và token đều có thể đối chiếu; nhãn đánh giá hỗ trợ AI được công bố minh bạch để người học xác nhận cuối.
