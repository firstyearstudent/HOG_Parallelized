# Parallel HOG Feature Extraction Benchmark

Dự án này thực hiện và so sánh hiệu năng của thuật toán trích xuất đặc trưng ảnh (HOG - Histogram of Oriented Gradients), cụ thể là bước tính toán Gradient (Sobel), sử dụng các kỹ thuật lập trình song song khác nhau.

Mục tiêu chính là đánh giá tốc độ tăng tốc (Speedup) khi chuyển đổi từ mã chạy tuần tự trên CPU sang đa luồng CPU (OpenMP) và tính toán trên GPU (CUDA, OpenCL).

## 🚀 Các phương pháp triển khai
Dự án bao gồm 4 phiên bản cài đặt của cùng một thuật toán tính Gradient:

1.  **Serial (Tuần tự):** Chạy trên 1 luồng CPU (làm cơ sở so sánh - Baseline).
2.  **OpenMP:** Song song hóa đa luồng trên CPU sử dụng chỉ thị `#pragma omp`.
3.  **CUDA:** Kernel chạy song song trên NVIDIA GPU.
4.  **OpenCL:** Kernel chạy trên GPU (hoặc CPU) thông qua chuẩn mở OpenCL.

> **Lưu ý:** Thư viện OpenCV chỉ được sử dụng để đọc/ghi ảnh (I/O). Toàn bộ phần tính toán toán học (Gradient Magnitude, Angle) được cài đặt thủ công bằng C/C++ và Kernel để đảm bảo tính công bằng trong so sánh.

## 📂 Cấu trúc Dự án

```text
HOG_Parallel_Project/
├── CMakeLists.txt          # Cấu hình build dự án
├── README.md               
├── images/                 # Thư mục chứa ảnh đầu vào
│   └── input_4k.jpg        # Ảnh kích thước lớn để test hiệu năng
├── output/                 # Thư mục chứa ảnh kết quả (để kiểm tra tính đúng đắn)
│   ├── gradient_cpu.jpg
│   └── gradient_cuda.jpg
├── include/                # Các file Header
│   ├── timing.hpp          # Class đo thời gian chính xác cao 
│   └── common.h            # Các định nghĩa chung
├── kernels/                # Mã nguồn Kernel chạy trên GPU
│   ├── hog_cuda.cu         # CUDA Kernel 
│   └── hog_opencl.cl       # OpenCL Kernel 
├── src/                    # Mã nguồn C++
│   ├── main.cpp            # Chương trình chính điều khiển luồng chạy
│   ├── hog_serial.cpp      # Cài đặt thuật toán tuần tự
│   ├── hog_omp.cpp         # Cài đặt OpenMP
│   ├── hog_opencl.cpp      # Thiết lập môi trường OpenCL (Host setup)
│   └── timing.cpp          # Cài đặt bộ đếm thời gian
└── scripts/
    └── utilityFuncs.py     # Script Python vẽ biểu đồ so sánh 
