# Nghiên cứu hệ thống tính toán song song phân tán cho huấn luyện và suy luận mô hình AI

Báo cáo Đồ án chuyên ngành I của sinh viên Nguyễn Hoàng, lớp KS-GenAI-K69, ngành Kỹ sư GenAI, Trường Công nghệ thông tin và Truyền thông (SOICT) - Đại học Bách khoa Hà Nội.

---

## 📌 Thông tin chung
* **Đề tài:** Nghiên cứu hệ thống tính toán song song phân tán cho huấn luyện và suy luận mô hình AI
* **Sinh viên thực hiện:** Nguyễn Hoàng (MSSV: `20250130E`, Email: `hoang.n250130e@sis.hust.edu.vn`)
* **Lớp:** `KS-GenAI-K69`
* **Ngành:** `Kỹ sư GenAI`
* **Giảng viên hướng dẫn (GVHD):** TS. Phạm Đăng Hải
* **Thời gian thực hiện:** Tháng 03/2026 – Tháng 06/2026
* **Hạn nộp báo cáo:** 15/06/2026

---

## 📂 Cấu trúc thư mục & các file chính

Dưới đây là sơ đồ tổ chức các file LaTeX nguồn và tài liệu trong project:

```text
├── main.tex                 # File cấu hình chính, định nghĩa package, format và nạp các phần khác
├── Cover page.tex           # Trang bìa chuẩn của Trường CNTT&TT (SOICT) - ĐHBKHN
├── Preface.tex              # Lời nói đầu của báo cáo
├── Pledge.tex               # Lời cam đoan của sinh viên
├── Abbreviations.tex        # Danh mục từ viết tắt (AI, DP, TP, PP, CP, NCCL, NVSHMEM, ZeRO, v.v.)
├── Abstract.tex             # Tóm tắt báo cáo bằng tiếng Việt
├── Chapter 1.tex            # Chương 1: Giới thiệu và cơ sở lý thuyết tính toán song song
├── Chapter 2.tex            # Chương 2: Lập trình song song trên CPU: OpenMP và MPI
├── Chapter 3.tex            # Chương 3: Lập trình GPU và giao tiếp liên GPU (CUDA, NCCL, NVSHMEM)
├── Chapter 4.tex            # Chương 4: Hệ thống huấn luyện và suy luận phân tán (PyTorch, DeepSpeed, Megatron Core)
├── Chapter 5.tex            # Chương 5: Thí nghiệm minh họa (Dự kiến thực nghiệm trên Kaggle 2 GPU T4)
├── Appendix.tex             # Phụ lục (mã nguồn/log chạy thí nghiệm)
├── References.bib           # File chứa các tài liệu tham khảo theo định dạng BibTeX (IEEEtran)
├── Figures/                 # Thư mục chứa hình ảnh minh họa cho báo cáo
│   ├── logoBK.png           # Logo Đại học Bách khoa Hà Nội
│   ├── vonNeumann1.png      # Cấu trúc von Neumann
│   ├── flynnsTaxonomy.png   # Phân loại Flynn
│   ├── shared_mem.png       # Mô hình Shared Memory
│   ├── distributed_mem.png  # Mô hình Distributed Memory
│   └── ...                  # Các hình vẽ và sơ đồ kiến trúc khác
├── mail.txt                 # Bản nháp chuỗi email báo cáo tiến độ (03/2026 - 06/2026)
├── plan.txt                 # Kế hoạch viết báo cáo (phác thảo ban đầu)
└── plan2.txt                # Kế hoạch viết báo cáo chi tiết (bản cập nhật đầy đủ 5 chương)
```

---

## 📖 Tóm tắt nội dung các chương

### Chương 1: Giới thiệu và cơ sở lý thuyết tính toán song song
* **Đặt vấn đề:** Sự bùng nổ của Generative AI và các mô hình ngôn ngữ lớn đòi hỏi tài nguyên tính toán vượt quá khả năng của một máy đơn lẻ.
* **Kiến trúc phần cứng cơ bản:** Tìm hiểu mô hình von Neumann truyền thống và phân loại Flynn (SIMD, MIMD).
* **Kiến trúc bộ nhớ:** Phân tích mô hình bộ nhớ chia sẻ (Shared Memory), bộ nhớ phân tán (Distributed Memory) và mô hình lai (NUMA - Hybrid).

### Chương 2: Lập trình song song trên CPU: OpenMP và MPI
* **OpenMP:** Mô hình lập trình cho hệ thống bộ nhớ chia sẻ (Shared Memory) dựa trên cơ chế Fork-Join và quản lý biến chia sẻ/riêng tư (`shared`/`private`).
* **MPI:** Chuẩn truyền thông điệp cho hệ thống bộ nhớ phân tán (Distributed Memory). Quản lý tiến trình qua Communicator và Rank; giao tiếp điểm-điểm (Point-to-Point) và tập thể (Collective Communication).
* **So sánh & kết hợp:** Kết hợp OpenMP (nội bộ node) và MPI (giữa các node) để tối ưu trên các siêu máy tính hiện đại.

### Chương 3: Lập trình GPU và giao tiếp liên GPU
* **Kiến trúc GPU:** Sử dụng mô hình SIMD tối ưu cho các phép tính ma trận trong Deep Learning.
* **Lập trình CUDA:** Quy trình giao tiếp Host (CPU) - Device (GPU) thông qua PCIe và quy trình 5 bước cơ bản:
  1. `cudaMalloc` (Cấp phát VRAM)
  2. `cudaMemcpy` (Sao chép Host ➔ Device)
  3. Kernel Launch (Tính toán song song trên GPU)
  4. `cudaMemcpy` (Sao chép Device ➔ Host)
  5. `cudaFree` (Giải phóng VRAM)
* **Phân cấp tính toán:** Quản lý logic từ **Grid ➔ Block ➔ Thread** tương ứng với cấu trúc phần cứng Streaming Multiprocessor (SM), Warp và L2 Cache.
* **Giao tiếp liên GPU:**
  * **NCCL:** Tối ưu hóa các phép toán giao tiếp tập thể (All-Reduce, All-Gather) trên phần cứng NVIDIA.
  * **NVSHMEM:** Dựa trên mô hình PGAS, giao tiếp truyền-nhận một phía (Put/Get) cho phép GPU truy cập thẳng vào VRAM của GPU khác với độ trễ tối thiểu.

### Chương 4: Hệ thống huấn luyện và suy luận phân tán
* **Framework:** PyTorch Distributed, DeepSpeed (Microsoft), Megatron-LM/Megatron Core (NVIDIA), và Ray Train.
* **Các chiến lược song song hóa:**
  * **DP (Data Parallelism) / FSDP (Fully Sharded Data Parallel):** Song song dữ liệu và tối ưu bộ nhớ ZeRO 1/2/3.
  * **TP (Tensor Parallelism):** Cắt dọc ma trận tính toán trong cùng một layer.
  * **PP (Pipeline Parallelism):** Cắt ngang mô hình theo các layer sâu, tối ưu bằng lịch trình 1F1B hoặc Interleaved.
  * **CP (Context Parallelism):** Băm nhỏ ngữ cảnh chuỗi đầu vào (Ring Attention, DeepSpeed Ulysses).
  * **EP (Expert Parallelism):** Sử dụng trong các mô hình MoE (Mixture of Experts) thông qua All-to-All.
* **So sánh Huấn luyện (Training) vs Suy luận (Inference):** So sánh chi tiết về yêu cầu tính toán, áp lực bộ nhớ (Activation Checkpointing vs KV Cache), giao tiếp mạng và các phương pháp tối ưu đặc thù (Quantization).
* **Tối ưu hóa suy luận (Inference Optimization):**
  * *Pha Prefill:* FlashAttention (tính toán SRAM), Chunked Prefill (tránh blocking), Prefix Caching (Radix Tree Matching).
  * *Quản lý KV Cache:* PagedAttention (tránh phân mảnh VRAM), Multi-head Latent Attention (MLA - nén latent), KV Compression/Quantization.
  * *Pha Decode:* Continuous Batching (iteration-level batching), Speculative Decoding (draft vs target models), Medusa (draft-less), EAGLE (feature-level speculation), Multi-Token Prediction (MTP).

### Chương 5: Thí nghiệm minh họa (Dự kiến)
* Khung thí nghiệm dự kiến nhằm kiểm chứng hiệu năng suy luận (Inference) trên mô hình lớn không vừa một GPU đơn lẻ.
* **Các chỉ số đánh giá:** Time to First Token (TTFT), thế hệ sinh token (Tokens/second), khả năng chịu tải của hệ thống.
* **Môi trường dự kiến:** Kaggle với 2 GPU NVIDIA T4.

---

## 🛠️ Hướng dẫn biên dịch tài liệu LaTeX

Để biên dịch file `main.tex` thành PDF đầy đủ phụ lục và tài liệu tham khảo trên Windows, bạn có thể thực hiện theo các bước sau trong PowerShell hoặc Command Prompt:

### 1. Chuẩn bị môi trường
Hãy đảm bảo bạn đã cài đặt một bản phân phối TeX (như **MiKTeX** hoặc **TeX Live**) và chương trình biên dịch tài liệu tham khảo **BibTeX**.

### 2. Các bước biên dịch thủ công
Chạy chuỗi lệnh sau tại thư mục gốc của project:

```powershell
# Bước 1: Biên dịch lần đầu để tạo các file phụ trợ (.aux, .out, ...)
pdflatex main.tex

# Bước 2: Biên dịch tài liệu tham khảo từ file References.bib
bibtex main

# Bước 3: Biên dịch lại để cập nhật các tham chiếu chéo và số trang
pdflatex main.tex

# Bước 4: Biên dịch lần cuối để hoàn thiện định dạng file PDF xuất ra
pdflatex main.tex
```

### 3. Biên dịch tự động bằng `latexmk` (khuyên dùng)
Nếu bản phân phối TeX của bạn có hỗ trợ `latexmk`, bạn chỉ cần chạy một lệnh duy nhất:
```powershell
latexmk -pdf main.tex
```
Lệnh này sẽ tự động phân tích các liên kết phụ thuộc và chạy `pdflatex`/`bibtex` số lần cần thiết để xuất ra file `main.pdf` hoàn chỉnh.

---

## 📝 Nhật ký & Kế hoạch Viết Báo cáo (Bối cảnh lịch sử)
* **Nguồn tài liệu gốc (`mail.txt`):** Tổng hợp từ các email báo cáo tiến độ hàng tuần từ tháng 3/2026 đến tháng 6/2026. Lịch trình bao gồm từ lý thuyết CPU song song ➔ CUDA/GPU ➔ Các kỹ thuật phân tán sâu hơn như DeepSpeed/Megatron.
* **Quy trình chuyển đổi văn phong:** Chuyển từ ngôn ngữ giao tiếp bán trang trọng trong email thành ngôn ngữ nghiên cứu khoa học, học thuật chuẩn hóa bằng tiếng Việt trong các chương báo cáo.
* **Chiến lược Thí nghiệm:** Do thời gian có hạn, bản báo cáo thiết lập sẵn một khung thí nghiệm dự kiến (Chương 5) để có thể cập nhật kết quả benchmark ngay khi có điều kiện thực nghiệm trên Kaggle/Colab mà không làm thay đổi cấu trúc chung của tài liệu.
