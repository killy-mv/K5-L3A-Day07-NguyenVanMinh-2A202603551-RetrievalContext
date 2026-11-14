# Day 7 — Agentic RAG

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.

## Bài làm cá nhân — CP1

Đã kết nối dữ liệu/index Day 6 và chạy xác thực thực tế, xem [docs/CP1_SETUP.md](docs/CP1_SETUP.md). Script `scripts/cp1_prepare.py` kiểm tra checksum nguồn, khớp 160 chunks với pgvector và thử hai truy vấn ngoài benchmark; không ghi lại index. Tiếp theo: CP2 benchmark dev/test mới.

## Bài làm cá nhân — CP2

Đã tạo và khóa [benchmark CP2](docs/CP2_BENCHMARK.md): 17 câu dev (15 có evidence, 2 thiếu evidence) và 7 câu test mới (5 có evidence, 2 thiếu evidence), tổng 24 câu. Các gold câu có evidence đều đối chiếu được với chunk nguồn/trang. Chưa chạy test benchmark. Kiểm chứng: `py scripts/cp2_validate.py`.

## Bài làm cá nhân — CP3

[CP3 agent thực tế](docs/CP3_AGENT.md) đã triển khai Gemini function calling `search_documents` nối pgvector Lab 6 và kiểm chứng hai case dev thật: một câu có citation, một câu từ chối khi thiếu bằng chứng. Mã chạy ở `src/agent.py`, `src/gemini_client.py`, `scripts/cp3_run.py`; test ở `tests/test_cp3_agent.py`. Không dùng test/gold CP2 trong vòng chạy agent. Các trace thực tế ở `starter/results/cp3/` (Git-ignored), xem tài liệu CP3 để tái lập.

CP3 có hai trace thực tế dùng để nộp tại `artifacts/cp3_smoke/` (`D7-D06.json` và `D7-D16.json`). Có thể tạo lại bằng `py scripts/cp3_export_smoke.py` sau khi chạy live smoke. Mọi thay đổi vẫn chưa commit.

## Bài làm cá nhân — CP4

Đã hoàn thiện [CP4: ba retrieval methods](docs/CP4_RETRIEVAL.md): Dense pgvector, Lexical BM25, Hybrid RRF; hỗ trợ chọn bằng `scripts/cp3_run.py --strategy dense|lexical|hybrid`. Các smoke thực tế dùng dữ liệu gốc và không chạm tập test: `artifacts/cp4_smoke.json` (6 truy vấn retrieval với top-3), `artifacts/cp4_agent_smoke/` (2 trace Gemini dev có citation). Chưa thực hiện benchmark CP5.

## Bài làm cá nhân — CP5 và bổ sung CP2

Đã hoàn thiện [CP2 audit](docs/CP2_BENCHMARK.md) và [CP5 toàn bộ Dev](docs/CP5_EVALUATION.md). Thực nghiệm CP5 gồm 51 retrieval và 51 Gemini Agent trên cùng 17 dev, ba phương pháp Dense/BM25/Hybrid, có đầy đủ trace trong `artifacts/cp5/`. Hit@1: 10/15, 12/15, 12/15; Evidence Hit@3: 13/15, 14/15, 15/15; Context Evidence Rate cùng mức với Hit@3 cho query gốc. Grounded/citation nháp hỗ trợ AI được phân biệt rõ với đánh giá thủ công độc lập. Đã ghi nhận và phục hồi 6 lỗi quota HTTP 429; chưa chạy test hay CP6/CP7.

## Bài làm cá nhân — CP6

Đã thực hiện [CP6: thí nghiệm và khóa cấu hình](docs/CP6_IMPROVEMENTS.md). E1 Weighted RRF 0.7/1.3 giảm Evidence Hit@3 Hybrid từ 15/15 xuống 14/15; E2 completeness prompt đã gọi Gemini thật nhưng không chấm được do HTTP 503/timeout; E2b Dense top5/context5000 tăng context evidence từ 13/15 lên 14/15 nhưng vẫn thấp hơn Hybrid. **Đã khóa Hybrid RRF cân bằng 1:1, top3, context3800, Gemini prompt CP3** tại `artifacts/cp6/final_config_lock.json` trước khi chạy test. Không có kết quả test trong CP6.
