#include <iostream>

#include <torch/torch.h>


/* template<typename T> */
/* struct dilation_op { */
/*     void CPU(const at::Type & t, at::Tensor &input_, at::Tensor &kernel_, */
/*                    int stride_rows, int stride_cols, int rate_rows, */
/*                    int rate_cols, int pad_top, int pad_left, */
/*                    at::Tensor &output_) { */
/*     }; */
/* }; */

at::Tensor dilation2d(at::Tensor input, at::Tensor kernel, int stride_rows,
                int stride_cols, int rate_rows, int rate_cols, int pad_top,
                int pad_left, int output_height, int output_width) {
    // Tensor accessors
    at::Tensor output = at::zeros({input.size(0), input.size(1), output_height, output_width},
            torch::CPU(at::kFloat));
    auto input_a = input.accessor<float,4>();
    auto kernel_a = kernel.accessor<float,3>();
    auto output_a = output.accessor<float,4>();

    const int batch = input.sizes()[0];
    const int depth = input.sizes()[1];
    const int input_rows = input.sizes()[2];
    const int input_cols = input.sizes()[3];

    const int kernel_rows = kernel.sizes()[1];
    const int kernel_cols = kernel.sizes()[2];

    const int output_rows = output.sizes()[2];
    const int output_cols = output.sizes()[3];

    for (int b = 0; b < batch; b++) {
        for (int h_out = 0; h_out < output_rows; h_out++) {
            int h_beg = h_out * stride_rows - pad_top;
            for (int w_out = 0; w_out < output_cols; w_out++) {
                int w_beg = w_out * stride_cols - pad_left;
                for (int d = 0; d < depth; d++) {
                    auto cur_val = -99999.0; // TODO(alex): Use Eigen
                    for (int h = 0; h < kernel_rows; h++) {
                        const int h_in = h_beg + h * rate_rows;
                        if (h_in >= 0 && h_in < input_rows) {
                            for (int w = 0; w < kernel_cols; w++) {
                                const int w_in = w_beg + w * rate_cols;
                                if (w_in >= 0 && w_in < input_cols) {
                                    auto val = input_a[b][d][h_in][w_in] + kernel_a[d][h][w];
                                    if (val > cur_val) {
                                        cur_val = val;
                                    }
                                }
                            }
                        }
                    }
                    output_a[b][d][h_out][w_out] = cur_val;
                }
            }
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dilation2d", &dilation2d, "Dilation2D");
}
