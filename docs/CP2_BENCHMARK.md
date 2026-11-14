# CP2 — Benchmark thiết kế, kiểm tra và khóa dev/test

## Mục tiêu và tình trạng

**Hoàn thành chuẩn bị benchmark (2026-11-14).** Chưa chạy retrieval/agent evaluation trên test. Tập câu hỏi dùng 160 paragraph chunks (B) và 35 trang nội dung của *Think Python 2e*, các chương 1–4, từ Lab 6 CP1.

| Split | Có evidence | Thiếu evidence | Tổng |
|---|---:|---:|---:|
| Dev | 15 | 2 | 17 |
| Test độc lập Lab 7 | 5 | 2 | 7 |
| **Tổng** | **20** | **4** | **24** |

- 15 câu dev xuất phát từ **dev Lab 6** (Q01–Q15); tuyệt đối không tái sử dụng Q16–Q20 của **test Lab 6** làm held-out test mới, vì kết quả đã được xem.
- 5 câu có evidence của test Lab 7 mới được tạo cho các chủ đề: flow of execution (PDF p43), phạm vi biến cục bộ (p44), stack diagram (p45), thiết kế interface (p55), development plan (p57). Những câu này không trùng đáp án trực tiếp với tập dev hoặc 5 câu test cũ.
- 4 câu thiếu evidence hỏi về cú pháp/phạm vi nằm ngoài 4 chương được index: list comprehension, class inheritance, async def, yield from. Bằng chứng phủ định được kiểm tra theo phạm vi 160 chunks, không dựa vào một kết quả top-k. Kiểm tra thêm search term là biện pháp hỗ trợ, **không tự chứng minh toàn bộ absence bằng ngữ nghĩa**; người đánh giá cần xác nhận lại phạm vi corpus khi chấm.
- Mỗi câu có đáp án có `source_id`, trang PDF vật lý, `chunk_id`, evidence nguyên văn trong một chunk, `expected_answer`. Source gốc: <https://greenteapress.com/thinkpython2/thinkpython2.pdf> (CC BY-NC 3.0, ghi rõ tác giả và chỉ chia sẻ phi thương mại theo giấy phép).

## Đặc biệt: alignment evidence của Lab 6

Day 6 kiểm tra evidence trên trang PDF theo chuẩn whitespace-normalization. Lab 7 kiểm chứng evidence có thể truy xuất từ **chính các chunk B đã index**. Script `scripts/cp2_build.py` đối chiếu chuỗi sau khi bỏ whitespace để tìm đoạn nguyên văn của cùng chunk mà không thay đổi đáp án. Ở câu Q10, evidence cũ trải trên hai chunks: gold Lab 7 sử dụng câu nguyên văn hỗ trợ trực tiếp kết luận ở chunk thứ hai và ghi `evidence_alignment_note`. Điều này không làm thay đổi source hay đáp án; có thể bổ sung cách chấm multi-chunk trong các CP sau nếu cần.

## Schema và quyền truy cập

- `benchmark/dev.json`: câu dev + nhãn và gold, **chỉ evaluator được đọc các trường gold/expected**.
- `benchmark/test.json`: 7 câu test mới và gold; **không đọc trong pipeline chọn cấu hình**, chỉ mở cho evaluation sau khi chốt tham số trên dev.
- `benchmark/split_lock.json`: SHA256 canonical của corpus và hai split, chính sách không dùng test để chỉnh; script báo lỗi nếu sửa benchmark sau freeze.
- `scripts/cp2_validate.py`: `validate_files()` xác thực toàn bộ khóa và evidence mà không chạy retrieval. `agent_inputs('dev')` chỉ trả `id`, `question`, `filter`; dùng cùng chuẩn cho test **chỉ khi chuyển sang CP đánh giá test**.
- `scripts/cp2_build.py`: chạy một lần để dựng bộ benchmark; cố tình từ chối ghi đè nếu file khóa đã tồn tại.

## Quy tắc sử dụng benchmark trong CP3–CP6

1. Development chỉ truy vấn và chấm **dev**. Cho agent thấy `question` và `filter` hợp lệ nếu có, không chuyển `gold`, `expected_answer`, `absence_reason`, hoặc thông tin đối chiếu nguồn vào prompt/tool.
2. Trên dev so sánh dense, lexical, hybrid cùng corpus/index/câu hỏi, với cùng định nghĩa Hit@1, Evidence Hit@3. 2 câu thiếu evidence **không tính** vào mẫu số Hit/Context Evidence Rate.
3. Sau khi chọn và ghi lại cấu hình cuối cùng, mới chạy một lần test cho baseline và cấu hình được chọn. Lưu cả lỗi, trace, citation và nhãn đánh giá có lý do.
4. Gold evidence được trích nguyên văn từ chunk index; **đây là kiểm tra vị trí bằng chứng, chưa phải thẩm định độc lập tất cả câu trả lời của người chấm**. Nếu phát hiện gold sai sau khi dùng test để điều chỉnh, hãy lập tập test mới và ghi công khai thay đổi.

## Tái lập (Git Bash, Windows)

```bash
# Từ thư mục gốc Lab 7, sau khi chạy CP1
py scripts/cp2_validate.py
PYTHONUTF8=1 py -m unittest discover -s tests -v
```

`py scripts/cp2_build.py` chỉ dành cho lần tạo mới, **không được chạy để ghi đè** split đã khóa. Để kiểm chứng lại với dữ liệu gốc Day 6 mà không tạo file mới: `py scripts/cp2_build.py --validate-only`.

## Chưa làm ở CP2

Không có kết quả Hit@1, Hit@3, grounding/citation/abstention/latency của agent; các chỉ số đó cần retrieval và agent thật ở các chặng sau. Không gán kết quả từ unit test thành benchmark chất lượng.
