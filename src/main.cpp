#include "common.h"
#include "timing.hpp" // Class đo thời gian của project cũ
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  // 1. Load ảnh (nhập đường dẫn ảnh từ dòng lệnh hoặc mặc định)
  string imagePath =
      (argc > 1) ? argv[1] : "../images/2025_Moscow_Victory_Day_Parade_55.jpg";
  Mat src = imread(imagePath, IMREAD_GRAYSCALE);

  if (src.empty()) {
    cerr << "[ERROR] Khong tim thay anh: " << imagePath << endl;
    return -1;
  }

  // Resize ảnh to lên (4K) để thấy rõ sự chênh lệch tốc độ
  // Nếu ảnh quá nhỏ, OpenMP có thể chậm hơn Serial do chi phí tạo luồng
  resize(src, src, Size(3840, 2160));
  cout << "[INFO] Kich thuoc anh: " << src.cols << "x" << src.rows << endl;

  int width = src.cols;
  int height = src.rows;
  int num_pixels = width * height;

  // 2. Cấp phát bộ nhớ cho kết quả (Output)
  // Dùng vector để tự động quản lý bộ nhớ
  vector<float> mag_serial(num_pixels);
  vector<float> ang_serial(num_pixels);
  vector<float> mag_omp(num_pixels);
  vector<float> ang_omp(num_pixels);

  // Lấy con trỏ dữ liệu thô từ OpenCV
  const unsigned char *input_ptr = src.data;

  // 3. Chạy SERIAL
  Timing timer;
  cout << "------------------------------------------------" << endl;
  cout << "Dang chay Serial..." << endl;

  timer.start();
  compute_gradients_serial(input_ptr, mag_serial.data(), ang_serial.data(),
                           width, height);
  timer.end();

  double time_serial = timer.get_elapse();
  cout << "Serial Time: " << time_serial << " seconds" << endl;

  // 4. Chạy OPENMP
  int n_iter = 1000000; // Lặp 1000 lần để kịp nhìn btop
  cout << "Dang chay OpenMP (" << omp_get_max_threads() << " luong) trong "
       << n_iter << " lan lap..." << endl;
  cout << ">> HAY CHUYEN QUA CUA SO BTOP NGAY!" << endl;

  timer.start();
  // Vòng lặp stress test
  for (int k = 0; k < n_iter; k++) {
    compute_gradients_omp(input_ptr, mag_omp.data(), ang_omp.data(), width,
                          height);
  }
  timer.end();

  double time_omp = timer.get_elapse();
  // Tính trung bình 1 lần chạy
  cout << "Tong thoi gian: " << time_omp << "s" << endl;
  cout << "Trung binh OpenMP: " << time_omp / n_iter << " seconds/lan" << endl;

  // Speedup tính trên 1 lần chạy trung bình so với Serial 1 lần
  cout << ">> Speedup: " << time_serial / (time_omp / n_iter) << "x" << endl;
  cout << "------------------------------------------------" << endl;
  // 5. Lưu ảnh kết quả để kiểm tra (Validation)
  // Chuyển từ float (magnitude) về dạng ảnh 8-bit để xem
  Mat out_serial(height, width, CV_8UC1);
  Mat out_omp(height, width, CV_8UC1);

#pragma omp parallel for
  for (int i = 0; i < num_pixels; i++) {
    // Cắt giá trị > 255 về 255
    out_serial.data[i] = (unsigned char)min(255.0f, mag_serial[i]);
    out_omp.data[i] = (unsigned char)min(255.0f, mag_omp[i]);
  }

  imwrite("output_serial.jpg", out_serial);
  imwrite("output_omp.jpg", out_omp);
  cout << "[INFO] Da luu anh ket qua: output_serial.jpg, output_omp.jpg"
       << endl;

  return 0;
}
