import math

import torch
import torch.nn.functional as F

from ..utils.general import calculate_padding


class TFConv2D(torch.nn.Module):
    """TFConv2D implements the padding strategy used by Tensorflow."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding="SAME", dilation=1, groups=1, bias=True):
        super(TFConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels,
                                                      in_channels,
                                                      kernel_size,
                                                      kernel_size))

        if bias is True:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_normal_(self.weight.data)
        stdv = 1. / math.sqrt(self.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def pad_input(self, x):
        """Pads the input using the padding strategy defined in Tensorflow.

        https://www.tensorflow.org/api_guides/python/nn#Notes_on_SAME_Convolution_Padding

        Args:
            x - Tensor(N x C x H x W): Input.

        Returns:
            pad_x - Tensor(N x C x H + p_H x W x p_W): Padded output.
        """
        pad_h = calculate_padding(x.shape[2],
                                  self.weight.data.shape[2],
                                  self.stride)

        pad_w = calculate_padding(x.shape[3],
                                  self.weight.data.shape[3],
                                  self.stride)

        padding = (pad_h[0], pad_h[1], pad_w[0], pad_w[1])

        return F.pad(x, padding, "constant", 0)

    def forward(self, x):
        """Since the padding is based on Tensorflow's strategy, padding=0 is
        used as a parameter to the original PyTorch call."""

        if self.padding == "SAME":
            x = self.pad_input(x)

        return F.conv2d(x, self.weight, self.bias, self.stride, padding=0,
                        dilation=self.dilation, groups=self.groups)
