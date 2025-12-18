#include <math.h>
#include <stdio.h>
// Định nghĩa PI nếu chưa có
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Hàm tính Gradient (Sobel) tuần tự trên CPU
 * * @param input_img: Mảng 1 chiều chứa dữ liệu ảnh xám (Grayscale)
 * @param grad_mag:  Mảng đầu ra chứa độ lớn Gradient (Magnitude)
 * @param grad_ang:  Mảng đầu ra chứa góc Gradient (Angle) - Đơn vị Độ
 * @param width:     Chiều rộng ảnh
 * @param height:    Chiều cao ảnh
 */
void compute_gradients_serial(const unsigned char *input_img, float *grad_mag,
                              float *grad_ang, int width, int height) {

  // Duyệt qua từng pixel của ảnh
  // Lưu ý: Bỏ qua viền ngoài cùng (padding 1 pixel) để tránh lỗi truy cập bộ
  // nhớ biên
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {

      // Tính chỉ số (index) trong mảng 1 chiều
      // Công thức: index = hàng * chiều_rộng + cột
      int idx = y * width + x;

      // 1. Tính đạo hàm theo phương X (Gx)
      // Công thức: Pixel bên phải - Pixel bên trái
      // img(x+1, y) - img(x-1, y)
      float gx = (float)input_img[y * width + (x + 1)] -
                 (float)input_img[y * width + (x - 1)];

      // 2. Tính đạo hàm theo phương Y (Gy)
      // Công thức: Pixel bên dưới - Pixel bên trên
      // img(x, y+1) - img(x, y-1)
      float gy = (float)input_img[(y + 1) * width + x] -
                 (float)input_img[(y - 1) * width + x];

      // 3. Tính độ lớn (Magnitude)
      // M = sqrt(Gx^2 + Gy^2)
      grad_mag[idx] = sqrtf(gx * gx + gy * gy);

      // 4. Tính góc (Angle) - Tùy chọn cho HOG đầy đủ
      // Góc = arctan(Gy / Gx) * (180 / PI)
      grad_ang[idx] = atan2f(gy, gx) * (180.0f / M_PI);
    }
  }
}
