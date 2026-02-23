import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange

# Utility functions
def squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def np_to_th(x, requires_grad=False, dtype=None):
    if dtype is None:
        return torch.from_numpy(x).requires_grad_(requires_grad)
    else:
        return torch.from_numpy(x).type(dtype).requires_grad_(requires_grad)

class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x

class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

class AvgPool2dWithConv(nn.Module):
    def __init__(self, kernel_size, stride, dilation=1, padding=0):
        super(AvgPool2dWithConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self._pool_weights = None

    def forward(self, x):
        in_channels = x.size()[1]
        weight_shape = (in_channels, 1, self.kernel_size[0], self.kernel_size[1])
        if self._pool_weights is None or (
            (tuple(self._pool_weights.size()) != tuple(weight_shape))
            or (self._pool_weights.is_cuda != x.is_cuda)
            or (self._pool_weights.data.type() != x.data.type())
        ):
            n_pool = np.prod(self.kernel_size)
            weights = np_to_th(np.ones(weight_shape, dtype=np.float32) / float(n_pool))
            weights = weights.type_as(x)
            if x.is_cuda:
                weights = weights.cuda()
            self._pool_weights = weights

        pooled = F.conv2d(
            x,
            self._pool_weights,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            groups=in_channels,
        )
        return pooled

class Deep4Net(nn.Module):
    def __init__(
        self,
        n_chans,
        n_outputs,
        input_window_samples,
        final_conv_length="auto",
        n_filters_time=25,
        n_filters_spat=25,
        filter_time_length=10,
        pool_time_length=3,
        pool_time_stride=3,
        n_filters_2=50,
        filter_length_2=10,
        n_filters_3=100,
        filter_length_3=10,
        n_filters_4=200,
        filter_length_4=10,
        first_pool_mode="max",
        later_pool_mode="max",
        drop_prob=0.5,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        stride_before_pool=False,
    ):
        super().__init__()

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.split_first_layer = split_first_layer
        self.batch_norm = nn.BatchNorm2d(n_filters_4)
        self.dropout = nn.Dropout(drop_prob)
        self.stride_before_pool = stride_before_pool

        self.rearrange = Rearrange('b w c l -> (b w) c l 1')

        if self.stride_before_pool:
            self.conv_stride = pool_time_stride
            self.pool_stride = 1
        else:
            self.conv_stride = 1
            self.pool_stride = pool_time_stride

        pool_class_dict = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)
        first_pool_class = pool_class_dict[first_pool_mode]
        later_pool_class = pool_class_dict[later_pool_mode]

        self.layers = nn.Sequential()
        if self.split_first_layer:
            self.layers.add_module('conv_time', nn.Conv2d(n_chans, n_filters_time, (filter_time_length, 1), stride=(self.conv_stride, 1), padding=(filter_time_length // 2, 0)))
            self.layers.add_module('conv_spat', nn.Conv2d(n_filters_time, n_filters_spat, (1, 1), bias=not batch_norm))
            n_filters_conv = n_filters_spat
        else:
            self.layers.add_module('conv_time', nn.Conv2d(n_chans, n_filters_time, (filter_time_length, 1), stride=(self.conv_stride, 1), padding=(filter_time_length // 2, 0), bias=not batch_norm))
            n_filters_conv = n_filters_time
        if batch_norm:
            self.layers.add_module('bnorm', nn.BatchNorm2d(n_filters_conv, momentum=batch_norm_alpha, affine=True, eps=1e-5))
        self.layers.add_module('conv_nonlin', nn.ELU())
        self.layers.add_module('pool', first_pool_class(kernel_size=(pool_time_length, 1), stride=(self.pool_stride, 1)))

        def add_conv_pool_block(n_filters_before, n_filters, filter_length, block_nr):
            suffix = f'_{block_nr}'
            self.layers.add_module('drop' + suffix, nn.Dropout(p=drop_prob))
            self.layers.add_module('conv' + suffix, nn.Conv2d(n_filters_before, n_filters, (filter_length, 1), stride=(self.conv_stride, 1), padding=(filter_length // 2, 0), bias=not batch_norm))
            if batch_norm:
                self.layers.add_module('bnorm' + suffix, nn.BatchNorm2d(n_filters, momentum=batch_norm_alpha, affine=True, eps=1e-5))
            self.layers.add_module('nonlin' + suffix, nn.ELU())
            self.layers.add_module('pool' + suffix, later_pool_class(kernel_size=(pool_time_length, 1), stride=(self.pool_stride, 1)))

        add_conv_pool_block(n_filters_conv, n_filters_2, filter_length_2, 2)
        add_conv_pool_block(n_filters_2, n_filters_3, filter_length_3, 3)
        add_conv_pool_block(n_filters_3, n_filters_4, filter_length_4, 4)

        self.eval()
        if self.final_conv_length == 'auto':
            self.final_conv_length = self.get_output_shape()[2]

        self.final_conv = nn.Conv2d(n_filters_4, n_outputs, (self.final_conv_length, 1), bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

    def get_output_shape(self):
        with torch.no_grad():
            dummy_input = torch.zeros((1, self.n_chans, self.input_window_samples, 1), dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
            return self.layers(dummy_input).shape

    def forward(self, x):
        # x shape: (batch_size, n_windows, n_channels, window_samples)
        batch_size, n_windows, _, _ = x.shape

        x = self.rearrange(x)

        x = self.layers(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.final_conv(x)

        x = x.squeeze(-1)

        if x.shape[-1] == 1:
            x = x.squeeze(-1)

        # Reshape back to (batch_size, n_windows, n_outputs)
        x = rearrange(x, '(b w) o -> b w o', b=batch_size, w=n_windows)

        # Average over windows
        x = x.mean(dim=1)

        return x
