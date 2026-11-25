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
| Không có script đánh giá | `src/evaluate.py` |

## Kịch bản Non-IID cực đoan (giữ nguyên tinh thần)
Gốc: mỗi bệnh viện 90% một lớp + 5% + 5%. Dữ liệu thật có 4 lớp nên chỉnh nhẹ thành
**90% lớp chủ đạo + ~3,3% mỗi lớp còn lại**. Mặc định có bốn client, mỗi client chủ đạo một lớp
khác nhau (`glioma`, `meningioma`, `notumor`, `pituitary`), giữ Non-IID tại từng client nhưng cân
bằng tổng số mẫu toàn cục (1200 ảnh/client, không trùng ảnh). Nếu chỉ dùng hai client chủ đạo
`glioma` và `pituitary`, tổng dữ liệu sẽ thiếu nghiêm trọng `meningioma` và `notumor`, khiến mô hình
dễ bỏ qua hai lớp đó. Đổi bằng `--clients ...` hoặc `--dominant_frac`.

## Quy trình thật (mỗi bệnh viện chạy lẻ)
```bash
python src/server.py init --run_dir runs/exp1                       # server: tạo global_round_00.pt, gửi cho client
python src/client.py --client_id A --round 1 --run_dir runs/exp1 --data_dir ./hospital_A --dp   # client A
python src/client.py --client_id B --round 1 --run_dir runs/exp1 --data_dir ./hospital_B --dp   # client B
python src/server.py aggregate --run_dir runs/exp1 --round 1        # server: gộp -> global_round_01.pt
# lặp lại với --round 2, 3, ...
```
`--data_dir` có cấu trúc `<thư mục>/glioma|meningioma|notumor|pituitary/*.jpg`.

## Mô phỏng một máy
```bash
python src/run_simulation.py --data_root ./brain-tumor-mri-dataset --tag fedprox_dp --rounds 10 --mu 0.05 --dp
```
(`--data_root` chứa `Training/` và `Testing/`.) Script tự chia dữ liệu, chạy từng client bằng tiến trình riêng, hợp nhất, rồi gọi `src/evaluate.py`.

Khi đổi cấu hình client, nên dùng tag mới và `--repartition` để tạo manifest mới. Script cũng tự
chia lại nếu thiếu manifest của một client:
```bash
python src/run_simulation.py --data_root ./brain-tumor-mri-dataset --tag fedprox_dp_4clients \
  --rounds 10 --mu 0.05 --dp --train_mode head --repartition
```

## Bộ thí nghiệm gợi ý (ablation cho báo cáo)
```bash
python src/run_simulation.py --data_root D --tag fedavg           --mu 0
python src/run_simulation.py --data_root D --tag fedprox          --mu 0.05
python src/run_simulation.py --data_root D --tag fedavg_dp        --mu 0    --dp --train_mode head
python src/run_simulation.py --data_root D --tag fedprox_dp       --mu 0.05 --dp --train_mode head
python src/run_simulation.py --data_root D --tag fedprox_dp_eps   --mu 0.05 --dp --train_mode head --noise_multiplier 0.7
```
Hãy so sánh các cấu hình **cùng `train_mode`**. Dùng `src/evaluate.py --split val` để chọn siêu tham số, chỉ chạy `--split test` cho kết quả cuối.

## Tuning để tăng accuracy
Thử trước một baseline không DP để đo mức accuracy có thể đạt được, với Non-IID nhẹ hơn và fine-tune
các block cuối của backbone:
```bash
python src/run_simulation.py --data_root ./brain-tumor-mri-dataset --tag tune_last_blocks \
  --rounds 20 --epochs 3 --batch_size 32 --lr 0.0001 --mu 0.01 \
  --train_mode last_blocks --dominant_frac 0.7 --repartition --skip_eval
```
Chọn round theo validation, rồi chỉ đánh giá checkpoint đã chọn trên test:
```bash
python src/evaluate.py --run_dir runs/tune_last_blocks --split val --partition_dir partitions \
  --all_rounds --device auto --no_plots
python src/evaluate.py --run_dir runs/tune_last_blocks --split test \
  --data_root ./brain-tumor-mri-dataset --server_rounds 8 --client_rounds "" \
  --device auto --no_plots
```
Thay `8` bằng round có Macro-F1 validation cao nhất. Đây là điểm khởi đầu để thử nghiệm, không phải
bảo đảm đạt 80–90%; sau khi chọn cấu hình tốt, chạy lại với `--dp` để đo trade-off accuracy/privacy.

## Benchmark đa seed và notebook tổng hợp
Mở `src/notebooks/ViSEF_complete.ipynb` trên Colab sau khi tải lên toàn bộ dự án (giữ nguyên thư mục `src/`) để chạy kiểm tra dữ liệu, benchmark 3 seed × 4 cấu hình,
chọn checkpoint bằng validation Macro-F1, đánh giá test, sweep epsilon, MIA và biểu đồ mean ± std.
Mặc định so sánh `linear` (chỉ classifier cuối, khoảng 4 nghìn tham số) để kiểm tra tác động của
giảm số chiều huấn luyện lên utility của DP. Notebook có thể resume và từ chối tính độ lệch chuẩn
nếu thiếu seed. Các notebook cũ được giữ trong `legacy/notebooks/`.

`python src/summarize_results.py` xuất bảng CSV vào `results/` khi đủ 3 seed cho cả 4 cấu hình. Chạy các lệnh từ thư mục gốc dự án. Dữ liệu MRI không được phân phối kèm dự án.

## DP-SGD
Per-sample gradient (`torch.func.vmap`) → cắt chuẩn `--max_grad_norm` → cộng nhiễu Gauss `σ·C` → chia cho batch kỳ vọng, với **Poisson sampling**.
Ngân sách ε tích luỹ qua các round được tính bằng RDP accountant tự cài đặt (`src/privacy.py`), lưu ở `client_state/`, và in ra mỗi round. `--max_epsilon` sẽ từ chối huấn luyện nếu dự toán vượt ngân sách.

Giới hạn cần nêu trung thực trong báo cáo:
* Mục tiêu phân tích là DP ở mức **mẫu ảnh** cho từng client, trên phần tham số được huấn luyện; bảo đảm chỉ hợp lệ nếu sampling, cơ chế nhiễu, clipping và accountant cùng thỏa các giả định đã nêu; `num_samples` và phân bố nhãn (dùng cho Balanced-Softmax) **không** được bảo vệ bởi DP.
* Poisson sampling dùng nguồn ngẫu nhiên hệ điều hành khi bật mặc định; Gaussian noise vẫn dùng PyTorch PRNG được khởi tạo từ entropy hệ điều hành, không phải luồng Gaussian mật mã đã được kiểm định. Cần `secure_mode`/bộ sinh đã kiểm định kiểu Opacus trước khi tuyên bố bảo đảm DP cho triển khai thực.
* Phần backbone đóng băng là trọng số ImageNet công khai nên không tốn ngân sách quyền riêng tư.

## Tham số hay chỉnh
`--train_mode linear|head|last_blocks|all` (càng ít tham số, DP càng ít bị nhiễu át, nhưng độ chính xác trần thấp hơn) · `--mu` (FedProx) · `--noise_multiplier` · `--max_grad_norm` (nhìn "tỉ lệ bị clip" trong log; nếu ≈100% hãy tăng) · `--batch_size` (tăng làm đổi sampling rate q và epsilon, cần tính lại accountant) · `--epochs` · `--lr`.
