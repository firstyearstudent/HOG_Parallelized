#ifndef INCLUDE_COMMON_H_
#define INCLUDE_COMMON_H_

// Khai báo hàm (Header declaration)
void compute_gradients_serial(const unsigned char *input_img, float *grad_mag,
                              float *grad_ang, int width, int height);

void compute_gradients_omp(const unsigned char *input_img, float *grad_mag,
                           float *grad_ang, int width, int height);

#endif // INCLUDE_COMMON_H_
