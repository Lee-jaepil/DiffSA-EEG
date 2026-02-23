import warnings
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils import parametrizations as weight_norm
from typing import Optional

def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time dimension"""
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

def safe_log(x, eps=1e-6):
    """Prevents log(0) by using log(max(x, eps))."""
    return torch.log(torch.clamp(x, min=eps))

def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

class EEGModuleMixin:
    def __init__(
        self,
        n_outputs: Optional[int] = None,
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
        add_log_softmax: Optional[bool] = False,
    ):
        self._n_outputs = n_outputs
        self._n_chans = n_chans
        self._n_times = n_times
        self._add_log_softmax = add_log_softmax

    @property
    def n_outputs(self):
        if self._n_outputs is None:
            raise ValueError("n_outputs not specified.")
        return self._n_outputs

    @property
    def n_chans(self):
        if self._n_chans is None:
            raise ValueError("n_chans not specified.")
        return self._n_chans

    @property
    def n_times(self):
        if self._n_times is None:
            raise ValueError("n_times not specified.")
        return self._n_times

    @property
    def add_log_softmax(self):
        if self._add_log_softmax:
            warnings.warn(
                "LogSoftmax final layer will be removed! "
                + "Please adjust your loss function accordingly (e.g. CrossEntropyLoss)!"
            )
        return self._add_log_softmax

# TCN model class
class TCN(EEGModuleMixin, nn.Module):
    def __init__(
        self,
        n_outputs,
        n_chans,
        n_times,
        n_filters=30,
        n_blocks=3,
        kernel_size=5,
        drop_prob=0.5,
        add_log_softmax=False,
    ):
        EEGModuleMixin.__init__(
            self,
            n_outputs=n_outputs,
            n_chans=n_chans,
            n_times=n_times,
            add_log_softmax=add_log_softmax,
        )
        nn.Module.__init__(self)

        self.ensuredims = Ensure4d()
        t_blocks = nn.Sequential()
        for i in range(n_blocks):
            n_inputs = n_chans if i == 0 else n_filters
            dilation_size = 2 ** i
            t_blocks.add_module(
                f"temporal_block_{i}",
                _TemporalBlock(
                    n_inputs=n_inputs,
                    n_outputs=n_filters,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    drop_prob=drop_prob,
                ),
            )
        self.temporal_blocks = t_blocks
        self.final_layer = _FinalLayer(
            in_features=n_filters,
            out_features=self.n_outputs,
            add_log_softmax=add_log_softmax,
        )
        self.min_len = 1
        for i in range(n_blocks):
            dilation = 2 ** i
            self.min_len += 2 * (kernel_size - 1) * dilation

    def forward(self, x):
        x = self.ensuredims(x)
        x = x.permute(0, 2, 3, 1)  # (batch, sequence, channels, time) -> (batch, channels, time, sequence)
        x = x.reshape(x.size(0), x.size(1), -1)  # (batch, channels, time*sequence)
        x = self.temporal_blocks(x)
        x = x.mean(dim=2)  # Average over time dimension
        out = self.final_layer(x)  # (batch, n_outputs)
        return out

class _TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        drop_prob,
    ):
        super().__init__()
        self.conv1 = weight_norm.weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout1d(drop_prob)

        self.conv2 = weight_norm.weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout1d(drop_prob)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        init.normal_(self.conv1.weight, 0, 0.01)
        init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class _FinalLayer(nn.Module):
    def __init__(self, in_features, out_features=2, add_log_softmax=False):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.out_fun = nn.LogSoftmax(dim=1) if add_log_softmax else nn.Identity()

    def forward(self, x):
        fc_out = self.fc(x)
        out = self.out_fun(fc_out)
        return out
