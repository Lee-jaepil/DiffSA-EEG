# Faithful PyTorch port of ATCNet (Altaheri et al., IEEE Trans. Ind. Informatics 2023).
# Reference (Keras/TF): https://github.com/Altaheri/EEG-ATCNet  (models.py::ATCNet_)
#
# Architecture preserved exactly:
#   1) EEGNet-style Conv block: temporal conv -> depthwise spatial conv -> separable conv,
#      with two average-pooling stages that reduce the time axis.
#   2) Sliding-window Attention + TCN: the conv feature sequence (length T') is split into
#      `n_windows` overlapping windows; each window -> multi-head self-attention (MHA) block
#      -> temporal convolutional network (TCN, residual dilated causal convs); the last TCN
#      timestep of each window feeds a Dense(n_classes); the per-window logits are averaged.
#
# Deviations from the reference (documented; all values scaled for our LONG input, not
# structural changes to the network):
#   - Input adaptation: harness feeds (B, 120, 21, 250); we reshape to a single 21-ch trial of
#     30000 samples (see forward), exactly like the other baselines in this repo.
#   - Pooling factors: reference used ~1125-sample trials with pool1=8, pool2(poolSize)=7,
#     giving T'~=20. Our trial is 30000 samples, so we keep pool1=8 and raise pool2 8->20 so the
#     feature-sequence length before the sliding-window attention stays tractable (a few hundred).
#       T' = floor(floor(30000/8)/20) = floor(3750/20) = 187 steps.  n_windows=5 (unchanged) ->
#       per-window length = T' - n_windows + 1 = 183.
#   - Keras kernel/depthwise max-norm & L2 weight-decay constraints are not ported (they are
#     regularizers, not architecture); everything else (F1=16, D=2, F2=32, kernLength=64,
#     MHA key_dim=8/num_heads=2, TCN depth=2/kernel=4/filters=32, ELU, average fusion) matches.

import math
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """EEGNet-style feature extractor (Conv_block_ in the reference)."""

    def __init__(self, n_chans=21, F1=16, D=2, kern_length=64,
                 pool1=8, pool2=20, dropout=0.3):
        super().__init__()
        F2 = F1 * D
        # Temporal conv: kernel (1, kern_length), 'same' padding preserves the time length.
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_length),
                                       padding='same', bias=False)
        self.bn_temporal = nn.BatchNorm2d(F1)
        # Depthwise spatial conv over the channel axis (valid, collapses n_chans -> 1).
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1),
                                      groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout)
        # Separable conv = depthwise (1,16) 'same' + pointwise (1,1) -> F2.
        self.conv_sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                        groups=F1 * D, padding='same', bias=False)
        self.conv_sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn_sep = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        # x: (B, 1, n_chans, n_times)
        x = self.bn_temporal(self.conv_temporal(x))
        x = self.elu(self.bn_spatial(self.conv_spatial(x)))
        x = self.drop1(self.pool1(x))
        x = self.conv_sep_point(self.conv_sep_depth(x))
        x = self.elu(self.bn_sep(x))
        x = self.drop2(self.pool2(x))
        return x  # (B, F2, 1, T')


class MHABlock(nn.Module):
    """Vanilla multi-head self-attention block (mha_block in attention_models.py).

    Faithful to Keras semantics: LayerNorm(eps=1e-6) -> MHA(x,x) with independent
    key_dim per head (8) and value_dim==key_dim -> output projection back to `dim`
    -> dropout -> residual add. PyTorch's nn.MultiheadAttention ties head_dim to
    embed_dim/num_heads, so a small custom MHA is used to keep key_dim=8 exactly.
    """

    def __init__(self, dim, key_dim=8, num_heads=2, attn_dropout=0.5, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        inner = num_heads * key_dim
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.q = nn.Linear(dim, inner)
        self.k = nn.Linear(dim, inner)
        self.v = nn.Linear(dim, inner)
        self.proj = nn.Linear(inner, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, dim)
        res = x
        y = self.ln(x)
        B, L, _ = y.shape
        H, kd = self.num_heads, self.key_dim

        def split(t):
            return t.view(B, L, H, kd).transpose(1, 2)  # (B, H, L, kd)

        q, k, v = split(self.q(y)), split(self.k(y)), split(self.v(y))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(kd)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))
        out = attn @ v                                   # (B, H, L, kd)
        out = out.transpose(1, 2).reshape(B, L, H * kd)  # (B, L, inner)
        out = self.drop(self.proj(out))
        return res + out


class TCNBlock(nn.Module):
    """Residual dilated causal TCN (TCN_block_ in the reference).

    Operates on (B, L, C). depth residual stages; stage i uses dilation 2**i, each
    stage is two causal Conv1d -> BN -> activation -> dropout, added to a (possibly
    1x1-projected) residual, then activated.
    """

    def __init__(self, input_dim, depth=2, kernel_size=4, filters=32,
                 dropout=0.3, activation='elu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.act = nn.ELU() if activation == 'elu' else nn.ReLU()
        self.depth = depth
        self.convs1, self.convs2 = nn.ModuleList(), nn.ModuleList()
        self.bns1, self.bns2 = nn.ModuleList(), nn.ModuleList()
        self.drops1, self.drops2 = nn.ModuleList(), nn.ModuleList()
        for i in range(depth):
            in_ch = input_dim if i == 0 else filters
            self.convs1.append(nn.Conv1d(in_ch, filters, kernel_size))
            self.bns1.append(nn.BatchNorm1d(filters))
            self.drops1.append(nn.Dropout(dropout))
            self.convs2.append(nn.Conv1d(filters, filters, kernel_size))
            self.bns2.append(nn.BatchNorm1d(filters))
            self.drops2.append(nn.Dropout(dropout))
        # 1x1 projection for the first residual if channel count changes.
        self.res_proj = None
        if input_dim != filters:
            self.res_proj = nn.Conv1d(input_dim, filters, 1)

    def _causal(self, conv, x, dilation):
        pad = (self.kernel_size - 1) * dilation
        x = F.pad(x, (pad, 0))
        conv.dilation = (dilation,)
        return conv(x)

    def forward(self, x):
        # x: (B, L, C) -> work in (B, C, L)
        x = x.transpose(1, 2)
        out = None
        for i in range(self.depth):
            dilation = 2 ** i
            src = x if i == 0 else out
            h = self.drops1[i](self.act(self.bns1[i](
                self._causal(self.convs1[i], src, dilation))))
            h = self.drops2[i](self.act(self.bns2[i](
                self._causal(self.convs2[i], h, dilation))))
            if i == 0:
                res = self.res_proj(x) if self.res_proj is not None else x
            else:
                res = out
            out = self.act(h + res)
        return out.transpose(1, 2)  # (B, L, filters)


class ATCNet(nn.Module):
    """PyTorch ATCNet for the clinical-EEG harness. Returns raw logits (B, n_classes)."""

    def __init__(self, n_chans=21, n_classes=2, n_windows=5,
                 F1=16, D=2, kern_length=64, pool1=8, pool2=20, eegn_dropout=0.3,
                 tcn_depth=2, tcn_kernel_size=4, tcn_filters=32, tcn_dropout=0.3,
                 mha_key_dim=8, mha_heads=2, **kwargs):
        super().__init__()
        self.n_windows = n_windows
        F2 = F1 * D
        self.conv_block = ConvBlock(n_chans=n_chans, F1=F1, D=D,
                                    kern_length=kern_length, pool1=pool1,
                                    pool2=pool2, dropout=eegn_dropout)
        # One attention + TCN + head per sliding window (weights are per-window, as in ref).
        self.attn = nn.ModuleList([
            MHABlock(F2, key_dim=mha_key_dim, num_heads=mha_heads)
            for _ in range(n_windows)])
        self.tcn = nn.ModuleList([
            TCNBlock(F2, depth=tcn_depth, kernel_size=tcn_kernel_size,
                     filters=tcn_filters, dropout=tcn_dropout, activation='elu')
            for _ in range(n_windows)])
        self.heads = nn.ModuleList([
            nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)])

    def forward(self, x):
        # Harness contract: x is (B, 120, 21, 250). Reshape like the other baselines to a
        # single 21-channel trial of 30000 samples, then to (B, 1, n_chans, n_times).
        x = x.permute(0, 2, 3, 1)                       # (B, 21, 250, 120)
        x = x.reshape(x.size(0), x.size(1), -1)         # (B, 21, 30000)
        x = x.unsqueeze(1)                              # (B, 1, 21, 30000)

        feat = self.conv_block(x)          # (B, F2, 1, T')
        feat = feat.squeeze(2)             # (B, F2, T')  -- ref does block1[:,:,-1,:]
        feat = feat.transpose(1, 2)        # (B, T', F2)
        T = feat.size(1)

        logits = []
        for i in range(self.n_windows):
            st, end = i, T - self.n_windows + i + 1
            w = feat[:, st:end, :]         # (B, win_len, F2)
            w = self.attn[i](w)
            w = self.tcn[i](w)
            w = w[:, -1, :]                # last timestep -> (B, tcn_filters)
            logits.append(self.heads[i](w))
        return torch.stack(logits, dim=0).mean(dim=0)  # average fusion -> (B, n_classes)
