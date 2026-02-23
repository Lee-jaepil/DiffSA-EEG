import torch
from torch import nn
from einops.layers.torch import Rearrange

class Ensure4d(nn.Module):
    def forward(self, x):
        if x.dim() == 3:
            return x.unsqueeze(-1)
        elif x.dim() == 4:
            return x
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor")

class Expression(nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

def squeeze_final_output(x):
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNetv4(nn.Module):
    def __init__(
        self,
        n_outputs=2,
        n_chans=21,
        n_times=250,
        final_conv_length="auto",
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
        drop_prob=0.25,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        self.n_chans = n_chans
        self.n_times = n_times
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.third_kernel_size = third_kernel_size
        self.drop_prob = drop_prob

        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        self.ensuredims = Ensure4d()

        self.conv_temporal = nn.Conv2d(
            1, self.F1, (1, self.kernel_length), stride=1, bias=False,
            padding=(0, self.kernel_length // 2)
        )
        self.bnorm_temporal = nn.BatchNorm2d(
            self.F1, momentum=0.01, affine=True, eps=1e-3
        )
        self.conv_spatial = Conv2dWithConstraint(
            self.F1, self.F1 * self.D, (self.n_chans, 1), max_norm=1,
            stride=1, bias=False, groups=self.F1, padding=(0, 0)
        )
        self.bnorm_1 = nn.BatchNorm2d(
            self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3
        )
        self.elu_1 = nn.ELU()
        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        self.drop_1 = nn.Dropout(p=self.drop_prob)

        self.conv_separable_depth = nn.Conv2d(
            self.F1 * self.D, self.F1 * self.D, (1, 16), stride=1, bias=False,
            groups=self.F1 * self.D, padding=(0, 16 // 2)
        )
        self.conv_separable_point = nn.Conv2d(
            self.F1 * self.D, self.F2, (1, 1), stride=1, bias=False, padding=(0, 0)
        )
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.elu_2 = nn.ELU()
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        self.permute_back = Rearrange("batch x y z -> batch x z y")
        self.squeeze = Expression(squeeze_final_output)

        self.conv_classifier = None  # Initialized lazily in forward pass

    def _initialize_classifier(self, x):
        with torch.no_grad():
            dummy_output = self.forward_features(x)
            n_out_virtual_chans, n_out_time = dummy_output.shape[2:]

        if self.final_conv_length == "auto":
            self.final_conv_length = n_out_time

        self.conv_classifier = nn.Conv2d(
            self.F2, self.n_outputs,
            (n_out_virtual_chans, self.final_conv_length),
            bias=True
        ).to(x.device)

    def forward_features(self, x):
        # Adjust input shape
        if x.dim() == 4:  # (batch, seq_len, channels, time)
            x = x.permute(0, 2, 3, 1)  # (batch, channels, time, seq_len)
        elif x.dim() == 3:  # (batch, channels, time)
            x = x.unsqueeze(-1)  # (batch, channels, time, 1)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = x.reshape(x.size(0), x.size(1), -1)  # (batch, channels, time*seq_len)
        x = x.unsqueeze(1)  # (batch, 1, channels, time*seq_len)

        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_1(x)
        x = self.elu_1(x)
        x = self.pool_1(x)
        x = self.drop_1(x)
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bnorm_2(x)
        x = self.elu_2(x)
        x = self.pool_2(x)
        x = self.drop_2(x)
        return x

    def forward(self, x):
        if self.conv_classifier is None:
            self._initialize_classifier(x)

        x = self.forward_features(x)
        x = self.conv_classifier(x)
        x = self.permute_back(x)
        x = self.squeeze(x)
        return x
