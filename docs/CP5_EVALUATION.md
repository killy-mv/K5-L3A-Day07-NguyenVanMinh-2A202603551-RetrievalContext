# CP5 — Đánh giá toàn bộ dev: Dense vs BM25 vs Hybrid RRF

**Trạng thái:** hoàn thành thực nghiệm 17 dev x 3 retrieval = 51 truy vấn retrieval; hoàn thành 17 dev x 3 Agent = 51 lượt chạy Gemini thực tế. Có trace và nhãn đánh giá sơ bộ từng câu. **Chưa chạy test, chưa chốt phương pháp cuối (để CP6), và chưa có xác nhận nhãn độc lập từ người học.**

Ngày chạy: 14/11/2026. Corpus từ CP1: *Think Python 2e*, 35 trang nội dung chương 1–4, 160 chunk paragraph-B. Chỉ dùng câu hỏi từ `benchmark/dev.json` đã khóa ở CP2: 15 answerable + 2 unanswerable. Mọi lượt Agent chỉ nhận `id`, `question`, `filter` từ `agent_inputs('dev')`, không được xem `gold`/`expected_answer`/`absence_reason`. Người chấm dùng các nhãn sau khi đã chạy xong.

## 1. Cấu hình cố định để so sánh

| Cấu hình | Retrieval | Tham số | Index / nguồn | Context |
|---|---|---|---|---|
| Dense (baseline) | Embedding MiniLM 384-D + pgvector cosine | top-3 | Day 6 B, 160 chunk | 3.800 ký tự |
| Lexical | In-memory BM25 | k1=1.5, b=0.75, top-3 | Cùng 160 chunk | 3.800 ký tự |
| Hybrid | Dense + BM25, hợp nhất RRF | RRF k=60, 12 candidates mỗi nguồn, top-3 | Cùng corpus/index | 3.800 ký tự |

Agent trên cả ba: Gemini `gemini-3.5-flash-lite`, temperature=0.1, native tool calling `search_documents(query)`, ít nhất 1 lần tìm, tối đa 3 lần tìm, có thể viết lại truy vấn; context nguồn/trang/ID được trích dẫn theo `[chunk-id]`. Để so sánh ranking một cách công bằng, chạy **câu hỏi gốc** trực tiếp với retriever và đo kết quả top-3 riêng. Để đo Agent, chạy model thật với cùng câu hỏi gốc, cho phép tool query viết lại và lưu từng lượt tìm; không đánh đồng ranking một lượt với context của Agent.

## 2. Retrieval benchmark — query gốc, 15 câu answerable

| Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Mean retrieval time (ms) |
|---|---:|---:|---:|---:|
| Dense | 10/15 = 66,7% | 13/15 = 86,7% | 13/15 = 86,7% | 473,381 |
| BM25 | 12/15 = 80,0% | 14/15 = 93,3% | 14/15 = 93,3% | 0,420 |
| Hybrid RRF | 12/15 = 80,0% | **15/15 = 100%** | **15/15 = 100%** | 33,326 |

`Hit@1`: có ít nhất một gold evidence trong chunk top-1 đúng source/page. `Evidence Hit@3`: có gold evidence khớp trong ít nhất một chunk top-3. `Context Evidence Rate`: gold chunk nguyên vẹn (không cắt) có trong context giới hạn 3.800 ký tự tạo từ top-3. Tất cả đều chia **15 câu có đáp án**, không tính 2 câu thiếu evidence. Kết quả exact-match nguồn/trang/đoạn trích, **không phải** điểm semantic answer.

Mean time Dense chịu tác động model CPU load lần đầu; Hybrid chạy sau khi Dense đã load. Không so sánh latency này như một benchmark cold-start cân bằng, càng không cộng trực tiếp score cosine/BM25/RRF.

## 3. Agent thật — 17 dev mỗi phương pháp

| Chỉ số | Dense | BM25 | Hybrid |
|---|---:|---:|---:|
| Lượt dev hoàn thành | 17/17 | 17/17 | 17/17 |
| Có câu trả lời | 15 | 15 | 15 |
| Từ chối khi thiếu evidence | 2/2 | 2/2 | 2/2 |
| Từ chối nhầm câu answerable | 0/15 | 0/15 | 0/15 |
| Có gọi tool retrieval | 17/17 | 17/17 | 17/17 |
| Search calls (tổng) | 22 | 22 | 19 |
| Gemini calls (tổng) | 39 | 39 | 36 |
| Tokens provider báo cáo (tổng) | 34.692 | 35.180 | 28.100 |
| Mean agent latency (ms) | 3.835,03 | 2.854,25 | 3.553,45 |
| Exact gold nằm trong context cuối của Agent | 13/15 | 14/15 | 14/15 |
| Grounded answer grade **nháp hỗ trợ AI** | 0:0, 1:2, 2:13; 1,867/2 | 0:0, 1:2, 2:13; 1,867/2 | 0:0, 1:2, 2:13; 1,867/2 |
| Citation có nội dung hỗ trợ **nháp hỗ trợ AI** | 20/21 | 21/21 | 20/20 |

Với nhãn grounded 0/1/2: 0 = thiếu/sai evidence, 1 = câu trả lời đúng phần chính nhưng thiếu một điều kiện trong expected answer, 2 = đủ các ý thiết yếu và được source hỗ trợ. Hai câu dev D7-D03 (thiếu nêu phép XOR `^`) và D7-D14 (thiếu nêu góc quay tính bằng độ) được **chấm bảo thủ 1/2** trong bản review nháp trên cả ba phương pháp. Các câu còn lại được ghi nháp 2/2 sau khi đối chiếu câu trả lời và các passage trích dẫn. Ở Dense D7-D04, citation bổ sung vào glossary trang 29 mô tả loại dữ liệu nói chung, nhưng **không tự chứng minh giá trị cụ thể 2 và 42.0**, nên bản review bảo thủ chỉ ghi 20/21 citation có hỗ trợ đầy đủ. Đáp án chính vẫn được passage trang 26 xác nhận.

**Cảnh báo minh bạch:** Các nhãn grounded/citation/abstention vừa nêu do AI hỗ trợ đọc đối chiếu source, có lý do theo câu và citation, nhưng **`human_confirmed=false`**. Không được trình bày là 51 lần chấm độc lập của con người. Khi nộp nếu giảng viên yêu cầu human review, người học cần mở trace và xác nhận/đổi nhãn ở các file `agent_*_reviews_assisted.json`. `source_id`/`page`/`citation` tồn tại là điều kiện kỹ thuật, không tự động chứng minh grounded.

`Exact gold nằm trong context cuối` ở lượt Agent thấp hơn `Context Evidence Rate` trên truy vấn gốc ở một số trường hợp, do Agent dùng query viết lại và context sau nhiều lượt tìm. D7-D04 của cả ba và D7-D12 của Dense không mang đúng *chuỗi gold đã chọn* vào context, nhưng Agent vẫn có thể trả lời nhờ passage hỗ trợ tương đương. Phải phân biệt exact quote coverage với semantic support.

## 4. Lỗi dev, giả thuyết và cơ hội cải thiện

1. **Dense D7-D07:** Query gốc về vì sao `class` không được làm tên biến không lấy được gold vào top-3. BM25 và Hybrid tìm được đúng bằng chứng. *Giả thuyết:* truy vấn chứa keyword cụ thể nên lexical match thắng embedding semantic. *Hướng CP6:* thử kết hợp lexical / thay query.
2. **Dense D7-D12:** Query gốc về `dot notation` không lấy được gold vào top-3; BM25/Hybrid tìm được. *Giả thuyết:* từ khóa chuyên biệt trong phần glossary giúp BM25 hơn. *Hướng CP6:* Hybrid RRF hoặc truy vấn chứa từ khóa từ nguồn.
3. **BM25 D7-D01:** Query gốc “what is a program” không lấy đúng gold top-3, Dense/Hybrid lấy được. *Giả thuyết:* từ “program” phổ biến khiến lexical ưu tiên trang glossary/bài tập sai vị trí. *Hướng CP6:* hợp nhất Dense/BM25 hoặc thử lọc theo chapter khi được cung cấp hợp lệ.

Hai mức cải thiện quan sát được trên dev so với Dense: BM25 nâng Hit@1 từ 10/15 lên 12/15 và Evidence Hit@3 từ 13/15 lên 14/15; Hybrid nâng Hit@1 từ 10/15 lên 12/15 và Evidence Hit@3 lên 15/15. **Đây mới là so sánh phương pháp CP5, không tự nhận hai thí nghiệm cải tiến CP6 đã thực hiện.** Chọn/cố định cấu hình chính thức và hai thí nghiệm cải tiến riêng sẽ tiến hành tại CP6, trước khi xem kết quả test.

## 5. API error, latency và chi phí

BM25 gặp 6 HTTP 429 (rate/quota) trong lần chạy đầu, các trace lỗi được bảo quản trong `starter/results/cp5/agent/lexical/` dưới tên `*_previous_error_*.json` (ignored); sau khi giãn nhịp/retry đã chạy thành công 6/6 lượt. File `artifacts/cp5/provider_error_audit.json` ghi lịch sử này. Lỗi provider **không** bị chấm thành hallucination/abstention/grounding=0. Bảng Agent ở trên dùng 51 lượt cuối hoàn thành, không bao gồm các yêu cầu API lỗi trong số token/latency tính tổng, vì provider có thể không hoàn lại usage chính xác cho request lỗi.

Chi phí embedding/truy xuất là dùng máy cục bộ, không có phí API embedding nhưng có tài nguyên CPU/điện. Gemini usage có token input/output và total trong trace, **`api_cost_usd=null`**, vì chưa đối chiếu đơn giá và billing thật. Không được gán USD bằng 0 vì dùng gói free hoặc không thấy hóa đơn. Lần đầu tải model có chi phí cold-start; latency không đại diện cho SLA production.

## 6. File minh chứng và cách tái lập

| File | Vai trò |
|---|---|
| `artifacts/cp5/retrieval_comparison.json` | So sánh ba chiến lược trên query gốc, 15 answerable |
| `artifacts/cp5/retrieval_{dense,lexical,hybrid}_summary.json` | Mẫu số, tỉ lệ Hit/Context, latency |
| `artifacts/cp5/retrieval_{dense,lexical,hybrid}_traces.json` | 51 trace retrieval có top-3, score, text, context, gold hit label của evaluator |
| `artifacts/cp5/agent_comparison.json` | So sánh Gemini Agent trên 17 dev |
| `artifacts/cp5/agent_{dense,lexical,hybrid}_summary.json` | Tool use, call counts, grounded/citation **AI-assisted**, abstention |
| `artifacts/cp5/agent_{dense,lexical,hybrid}_traces.json` | 51 trace real agent có model/tool/query/result/context/citation/usage |
| `artifacts/cp5/agent_{dense,lexical,hybrid}_reviews_assisted.json` | Nhãn nháp chi tiết mỗi dev case/citation; `human_confirmed=false` |
| `artifacts/cp5/provider_error_audit.json` | Sáu lỗi HTTP 429 và trạng thái chạy lại |
| `artifacts/cp2_audit.json` | Hoàn thiện kiểm toán CP2: nguồn, evidence dev, absence, khóa split |

Trace Agent **không chứa gold**. Trace retrieval có `gold_comparison` do evaluator tạo *sau truy xuất*, không được đưa file này ngược vào Agent. File test khóa không được mở để chạy bất kỳ phương pháp nào tại CP5.

```bash
# Từ thư mục gốc Lab 7; cần Docker Day 6 + virtualenv Day 6
DAY6PY=/e/code/Vin/lab6/K5-L3A-Day06-NguyenVanMinh-2A202603551-DataIngestion/starter/.venv/Scripts/python.exe

# CP2 chỉ kiểm chứng, không thay đổi split
py scripts/cp2_validate.py
py scripts/cp2_audit.py

# CP5 retrieval: tự tiếp tục nếu case output đã tồn tại; không chạy test
$DAY6PY scripts/cp5_retrieval.py

# CP5 agent thực tế: có thể dừng/tiếp tục theo từng strategy; mỗi file 1 câu
$DAY6PY scripts/cp5_agent.py --strategy dense --pace-seconds 7
$DAY6PY scripts/cp5_agent.py --strategy lexical --pace-seconds 8 --retry-errors
$DAY6PY scripts/cp5_agent.py --strategy hybrid --pace-seconds 7

# Tổng hợp và xuất minh chứng có kiểm tra
py scripts/cp5_summarize.py
py scripts/cp5_error_audit.py
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

Chạy lại với `--retry-errors` phải giữ trace lỗi trước đó, không được xóa các lần thất bại hoặc lén sửa benchmark. Các JSON kết quả trong `artifacts/cp5/` có thể chia sẻ phục vụ đánh giá giáo dục theo CC BY-NC khi giữ nguồn/giấy phép. Tập test vẫn chỉ dùng **sau khi khóa cấu hình cuối ở CP6**.

## 7. CP5 hoàn thành, phần chưa xác nhận

- Đã làm: benchmark retrieval 3 phương pháp trên 17 dev; trace per-case, top-3, Context Evidence Rate, 51 lượt Gemini Agent, provider error audit, review nháp answer/citation/abstention, hai ví dụ lỗi dev trở lên và giả thuyết.
- Chưa làm: xác nhận độc lập của con người cho từng nhãn, chi phí USD đối chiếu billing, 2 thí nghiệm cải tiến định lượng của CP6, quyết định khóa cấu hình cuối, và bất kỳ đánh giá test nào.
- Chất lượng test/khả năng tổng quát hóa còn **chưa biết**. Tuyệt đối không dùng 15/15 của dev Hybrid để tuyên bố test cũng đạt 100%.
