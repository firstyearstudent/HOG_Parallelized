#include <math.h>
#include <omp.h> // Thư viện OpenMP
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Hàm tính Gradient song song trên CPU sử dụng OpenMP
 * Logic toán học giống hệt bản Serial, nhưng chạy trên nhiều luồng.
 */
void compute_gradients_omp(const unsigned char *input_img, float *grad_mag,
                           float *grad_ang, int width, int height) {

// #pragma omp parallel for: Chỉ thị tạo vùng song song và chia vòng lặp for
// collapse(2): Gộp 2 vòng lặp (y và x) thành 1 không gian lặp lớn để chia việc
// đều hơn schedule(static): Chia khối lượng công việc cố định cho các luồng
// (tốt cho xử lý ảnh vì mỗi pixel tính toán như nhau)
#pragma omp parallel for collapse(2) schedule(static)
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {

      int idx = y * width + x;

      // 1. Tính Gx
      float gx = (float)input_img[y * width + (x + 1)] -
                 (float)input_img[y * width + (x - 1)];

      // 2. Tính Gy
      float gy = (float)input_img[(y + 1) * width + x] -
                 (float)input_img[(y - 1) * width + x];

      // 3. Tính Magnitude
      grad_mag[idx] = sqrtf(gx * gx + gy * gy);

      // 4. Tính Angle
      grad_ang[idx] = atan2f(gy, gx) * (180.0f / M_PI);
    }
  }
}
