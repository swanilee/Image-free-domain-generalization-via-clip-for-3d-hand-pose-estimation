#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

template <typename scalar_t>
__global__ void dilation_cuda_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ kernel,
        scalar_t* __restrict__ output,
        int input_rows,
        int input_cols,
        int kernel_rows,
        int kernel_cols,
        int output_rows,
        int output_cols,
        int stride_rows,
        int stride_cols,
        int rate_rows,
        int rate_cols,
        int pad_top,
        int pad_left) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int h_beg = y * stride_rows - pad_top;
    const int w_beg = x * stride_cols - pad_left;

    auto cur_val = -99999.0f;
    for (int h = 0; h < kernel_rows; h++) {
        const int h_in = h_beg + h * rate_rows;
        if (h_in >= 0 && h_in < input_rows) {
            for (int w = 0; w < kernel_cols; w++) {
                const int w_in = w_beg + w * rate_cols;
                if (w_in >= 0 && w_in < input_cols) {
                    auto val =
                        input[h_in * input_cols + w_in] + kernel[h * kernel_cols + w];
                    if (val > cur_val) {
                        cur_val = val;
                    }
                }
            }
        }
    }
    output[y * output_cols + x] = cur_val;
}

at::Tensor dilation_cuda(
        at::Tensor input,
        at::Tensor kernel,
        int stride_rows,
        int stride_cols,
        int rate_rows,
        int rate_cols,
        int pad_top,
        int pad_left,
        int output_height,
        int output_width) {

    const auto batch_size = input.size(0);

    const int input_rows = input.size(1);
    const int input_cols = input.size(2);
    const int kernel_rows = kernel.size(0);
    const int kernel_cols = kernel.size(1);

    const int req_thread = min(output_height, 16);

    const dim3 threads(req_thread, req_thread);
    const int im_thread = (output_height / req_thread);
    const dim3 blocks(im_thread, im_thread);

    auto output = at::zeros({batch_size, output_height, output_width},
            input.type());

    for (int i = 0; i < batch_size; i++) {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "dilation_cuda", ([&] {
                    dilation_cuda_kernel<scalar_t><<<blocks, threads>>>(
                            input[i].data<scalar_t>(),
                            kernel.data<scalar_t>(),
                            output[i].data<scalar_t>(),
                            input_rows,
                            input_cols,
                            kernel_rows,
                            kernel_cols,
                            output_height,
                            output_width,
                            stride_rows,
                            stride_cols,
                            rate_rows,
                            rate_cols,
                            pad_top,
                            pad_left);
                    }));
    }

    gpuErrorcheck(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    return output;
}
