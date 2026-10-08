# Federated Learning phân loại u não MRI — FedProx + DP-SGD, trao đổi bằng file

## Thay đổi so với bản gốc

| Vấn đề ở bản gốc | Xử lý |
|---|---|
| Socket TCP: `recv(4)` có thể trả về <4 byte, server treo nếu 1 client lỗi, phải mở cổng | Bỏ mạng. Client chạy lẻ, ghi **file cập nhật**; server đọc file |
| Server `torch.load(..., weights_only=False)` → file độc hại có thể chạy mã tuỳ ý | Mọi file đọc bằng `weights_only=True`; server kiểm tra round/shape/NaN |
| BatchNorm ở chế độ train, batch 16, dữ liệu 90/5/5 → thống kê BN lệch theo client, FedAvg trung bình cả `num_batches_tracked` | BN luôn eval (dùng thống kê ImageNet); chỉ gửi/hợp nhất các tham số được huấn luyện |
| Client gửi toàn bộ trọng số (~10 MB) | Chỉ gửi **delta** của phần được huấn luyện (`head` ≈ 2,4 MB) |
| Trọng số lớp nghịch đảo tần suất (tỉ lệ 18×) gây loss bất ổn | Balanced-Softmax (logit adjustment) có làm mượt Laplace |
| Không augmentation, đọc đĩa lại mỗi epoch | Augmentation nhẹ; cache ảnh 224×224 trong RAM; prefix đóng băng chạy `no_grad` (+fp16 trên GPU) |
| 3 lớp, không có validation/test, không seed | 4 lớp thật, validation toàn cục tách riêng, seed cố định |
| Không có script đánh giá | `evaluate.py` |

## Kịch bản Non-IID cực đoan (giữ nguyên tinh thần)
Gốc: mỗi bệnh viện 90% một lớp + 5% + 5%. Dữ liệu thật có 4 lớp nên chỉnh nhẹ thành
**90% lớp chủ đạo + ~3,3% mỗi lớp còn lại**. Mặc định: A chủ đạo `glioma`, B chủ đạo `pituitary`
(1200 ảnh/client, không trùng ảnh). Đổi bằng `--clients A:glioma B:notumor` hoặc `--dominant_frac`.

## Quy trình thật (mỗi bệnh viện chạy lẻ)
```bash
python server.py init --run_dir runs/exp1                       # server: tạo global_round_00.pt, gửi cho client
python client.py --client_id A --round 1 --run_dir runs/exp1 --data_dir ./hospital_A --dp   # client A
python client.py --client_id B --round 1 --run_dir runs/exp1 --data_dir ./hospital_B --dp   # client B
python server.py aggregate --run_dir runs/exp1 --round 1        # server: gộp -> global_round_01.pt
# lặp lại với --round 2, 3, ...
```
`--data_dir` có cấu trúc `<thư mục>/glioma|meningioma|notumor|pituitary/*.jpg`.

## Mô phỏng một máy
```bash
python run_simulation.py --data_root ./brain-tumor-mri-dataset --tag fedprox_dp --rounds 10 --mu 0.05 --dp
```
(`--data_root` chứa `Training/` và `Testing/`.) Script tự chia dữ liệu, chạy từng client bằng tiến trình riêng, hợp nhất, rồi gọi `evaluate.py`.

## Bộ thí nghiệm gợi ý (ablation cho báo cáo)
```bash
python run_simulation.py --data_root D --tag fedavg           --mu 0
python run_simulation.py --data_root D --tag fedprox          --mu 0.05
python run_simulation.py --data_root D --tag fedavg_dp        --mu 0    --dp --train_mode head
python run_simulation.py --data_root D --tag fedprox_dp       --mu 0.05 --dp --train_mode head
python run_simulation.py --data_root D --tag fedprox_dp_eps   --mu 0.05 --dp --train_mode head --noise_multiplier 0.7
```
Hãy so sánh các cấu hình **cùng `train_mode`**. Dùng `evaluate.py --split val` để chọn siêu tham số, chỉ chạy `--split test` cho kết quả cuối.

## DP-SGD
Per-sample gradient (`torch.func.vmap`) → cắt chuẩn `--max_grad_norm` → cộng nhiễu Gauss `σ·C` → chia cho batch kỳ vọng, với **Poisson sampling**.
Ngân sách ε tích luỹ qua các round được tính bằng RDP accountant tự cài đặt (`privacy.py`), lưu ở `client_state/`, và in ra mỗi round. `--max_epsilon` sẽ từ chối huấn luyện nếu dự toán vượt ngân sách.

Giới hạn cần nêu trung thực trong báo cáo:
* Đảm bảo DP ở mức **mẫu ảnh** cho từng client, trên phần tham số được huấn luyện; `num_samples` và phân bố nhãn (dùng cho Balanced-Softmax) **không** được bảo vệ bởi DP.
* `torch.randn` không phải bộ sinh ngẫu nhiên an toàn mật mã (cần `secure_mode` kiểu Opacus nếu triển khai thực).
* Phần backbone đóng băng là trọng số ImageNet công khai nên không tốn ngân sách quyền riêng tư.

## Tham số hay chỉnh
`--train_mode linear|head|last_blocks|all` (càng ít tham số, DP càng ít bị nhiễu át, nhưng độ chính xác trần thấp hơn) · `--mu` (FedProx) · `--noise_multiplier` · `--max_grad_norm` (nhìn "tỉ lệ bị clip" trong log; nếu ≈100% hãy tăng) · `--epochs` · `--lr`.
