#include <torch/torch.h>

#include <vector>

// CUDA forward declarations
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
        int output_width);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor dilation(
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
    CHECK_INPUT(input);
    CHECK_INPUT(kernel);

    return dilation_cuda(input, kernel, stride_rows, stride_cols, rate_rows,
            rate_cols, pad_top, pad_left, output_height, output_width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dilation2d", &dilation, "Dilation (CUDA)");
}
