# Day 7 — Checklist nộp bài (phần cơ bản)

**Báo cáo cá nhân:** `report/individual/2A202603551_NguyenVanMinh.md`

## Đã chuẩn bị đầy đủ

- [x] Code Agent Gemini function calling và 3 retrieval: Dense pgvector, Lexical BM25, Hybrid RRF (`src/`).
- [x] Model embedding, vector store Lab6, nguồn tài liệu, giấy phép CC BY-NC 3.0, hướng dẫn cài đặt (`docs/CP1_SETUP.md`).
- [x] Benchmark 24 câu, 17 dev và 7 test, evidence theo source ID/trang/chunk, khóa SHA256 (`benchmark/`, `artifacts/cp2_audit.json`).
- [x] Ba phương pháp chạy cùng dev, kết quả truy vấn/trace từng câu và các metric Hit@1, Evidence Hit@3, Context Evidence Rate (`artifacts/cp5/`).
- [x] Gemini Agent thật trên dev, context, citation, abstention, tool use, token và latency (`artifacts/cp5/`).
- [x] Thí nghiệm E1/E2b có metric trước/sau, lý do và phân tích lỗi; E2 prompt được ghi nhận riêng không có điểm hợp lệ do provider (`artifacts/cp6/`, `docs/CP6_IMPROVEMENTS.md`).
- [x] Đã chốt Hybrid trên dev và khóa trước khi mở test (`artifacts/cp6/final_config_lock.json`).
- [x] Đã chạy đầy đủ Dense và Hybrid trên **7/7 câu test**: 5 answered, 2 đúng abstention mỗi phương pháp; tất cả trace cuối hợp lệ (`artifacts/cp7/`).
- [x] Bảng chấm context 0/1/2 cho 5 câu có evidence mỗi phương pháp; bản review có nguồn/trang/lý do (`artifacts/cp7/context_*`).
- [x] Bảng trace/model/citation, chi phí API giả định có ghi rõ không phải hóa đơn (`artifacts/cp7/estimated_api_costs.json`).
- [x] Báo cáo cá nhân đã điền kết quả CP1–CP7 (`report/individual/2A202603551_NguyenVanMinh.md`).
- [x] Unit test và hướng dẫn tái lập (`tests/`, `scripts/`, `README.md`, `docs/CP7_FINAL_EVALUATION.md`).

## Cần người học xác nhận khi nộp

- [ ] Xem lại và xác nhận độc lập các nhãn *grounding/citation/context 0/1/2* đang được gắn `human_confirmed=false`. Các review hiện là kết quả AI hỗ trợ đối chiếu nguồn, không thể tự nhận là do người học chấm.
- [ ] Xác nhận tài khoản giảng viên có quyền truy cập repo, commit/push và kiểm tra đúng link repo/nhánh sau khi nộp.
- [ ] Xác nhận điều kiện chia sẻ theo CC BY-NC và quy định của lớp; không đưa `.env`, API key hoặc dữ liệu mật vào Git.

## Thử thách nâng cao (tùy chọn)

- [ ] Tài liệu >=100 trang chứa bảng/ảnh/biểu đồ và benchmark đa dạng theo ADVANCED.md. **Không thực hiện bonus**; toàn bộ mục phần cơ bản ở trên đã được triển khai.

## Kiểm tra trước khi nộp

```bash
py scripts/cp2_validate.py
py scripts/cp6_freeze.py
py scripts/cp7_offline_summary.py
py scripts/cp7_context_review.py
PYTHONUTF8=1 py -m unittest discover -s tests -v
git status
```

**Kết quả cuối:** Dense và Hybrid đều 7/7 Agent test hợp lệ, Hit@3=5/5 và đúng 2/2 abstention; bằng chứng và giới hạn ở `docs/CP7_FINAL_EVALUATION.md`. Những lỗi provider trước khi thay API key đã được retry thành công bằng đúng Gemini 3.5 Flash-Lite và lưu trong lịch sử trace local, không ảnh hưởng kết quả cuối.
