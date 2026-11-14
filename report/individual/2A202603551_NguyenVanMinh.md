# Báo cáo cá nhân Day 7 — Nguyễn Văn Minh (2A202603551)

**Trạng thái:** CP1–CP2 hoàn thành chuẩn bị, CP3+ chưa thực hiện. Những chỉ số đánh giá bên dưới để trống cho đến khi chạy benchmark thật.

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: Allen B. Downey, *Think Python 2e*, https://greenteapress.com/thinkpython2/thinkpython2.pdf, CC BY-NC 3.0; 35 trang nội dung (chương 1–4), 160 chunk kiểu paragraph B (tối đa 650 ký tự), đối chiếu nguồn/metadata với DB 160/160 trong CP1.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16 + pgvector 0.8.7 (chiến lược B của Lab 6); sentence-transformers/all-MiniLM-L6-v2 (revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41), 384 chiều, normalize, cosine similarity.
- Agent model, cách chạy và phân chia dev/test: Model agent chưa chọn/kết nối (CP3). CP2 khóa 17 dev (15 có đáp án, 2 thiếu evidence) và 7 test mới (5 có đáp án, 2 thiếu evidence); không dùng lại 5 câu test Lab 6 đã xem kết quả.
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

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
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
