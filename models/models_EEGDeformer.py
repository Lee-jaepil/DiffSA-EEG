# EEG-Deformer: A Dense Convolutional Transformer for Brain-Computer Interfaces
# Ding et al., 2024.  Faithful re-implementation adapted from the official code:
#   https://github.com/yi-ding-cs/EEG-Deformer/blob/main/models/EEGDeformer.py
#
# The submodules (FeedForward, Attention, Transformer with the coarse-grained
# self-attention branch + fine-grained dense-conv branch + info-enriched
# "get_info" features, Conv2dWithConstraint shallow encoder, dense-connection
# classification head) are kept structurally identical to the reference.
#
# ---------------------------------------------------------------------------
# Deviations from the reference (required for our much longer input) and why:
#
#   * Input length. The reference was validated on short trials (hundreds to
#     ~1000 samples). Our harness feeds a single-trial 21-channel EEG of
#     30000 samples (120 windows x 250 timesteps, concatenated). In this
#     architecture the transformer's *sequence length* is num_kernel (=64,
#     fixed), while the temporal feature dimension `dim = 0.5 * num_time` is
#     the quantity that scales with input length. Left unchanged, dim would be
#     15000 -> the pos-embedding, Conv1d dense branch and Linear heads would be
#     enormous. We therefore add initial temporal downsampling in the shallow
#     conv encoder so the temporal dim entering the first transformer stage is
#     ~1500 (see `temporal_stride` and `initial_pool` below). Everything else
#     (depth, heads, dim_head, mlp_dim, temporal_kernel, the per-layer halving)
#     is left at the reference values.
#
#   * temporal_stride=10 on the first temporal conv, plus the reference's
#     MaxPool2d((1,2)) (initial_pool=2) -> total /20 downsampling:
#     30000 -> 3000 (strided conv) -> 1500 (pool). So the temporal feature
#     dim at the first transformer stage is 1500 (<= the ~1500 token budget).
#     We prefer a *strided conv* for the bulk of the downsampling (learned)
#     over one giant max-pool (lossy). Reference used stride 1 + pool /2.
#
#   * `dim` is computed dynamically via a dry run through the conv encoder
#     rather than hard-coded to int(0.5*num_time), so the pos-embedding and
#     head sizes stay exactly consistent with the strided/pooled encoder.
#
#   * dropout default 0.25 (reference default 0.0; its experiments used 0.5).
#
# I/O contract for this repo's harness:
#   forward receives x of shape (B, 120, 21, 250); it is reshaped to
#   (B, 21, 30000) exactly like the other baselines, then fed to Deformer.
#   Returns raw logits (B, n_classes) -- no softmax.
# ---------------------------------------------------------------------------

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)              # coarse-grained: downsample then self-attention
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)                    # fine-grained: dense conv branch
            x_info = self.get_info(x_fg)     # info-enriched feature (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth  (dense connections)
        x = x.view(x.size(0), -1)                   # b, in_chan*d_hidden_last_layer
        emd = torch.cat((x, x_dense), dim=-1)
        return emd

    def get_info(self, x):
        # x: b, k, l  -> log power across time (purity / info-enriched feature)
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    """Faithful EEG-Deformer, with an initial-downsampling knob for long signals."""

    def cnn_block(self, out_chan, kernel_size, num_chan, temporal_stride, initial_pool):
        return nn.Sequential(
            # strided temporal conv carries the bulk of the downsampling (learned)
            Conv2dWithConstraint(1, out_chan, kernel_size, stride=(1, temporal_stride),
                                 padding=self.get_padding(kernel_size[-1]), max_norm=2),
            # spatial conv collapses the channel axis to 1 (reference behaviour)
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, initial_pool), stride=(1, initial_pool))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.,
                 temporal_stride=10, initial_pool=2):
        super().__init__()

        self.cnn_encoder = self.cnn_block(
            out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan,
            temporal_stride=temporal_stride, initial_pool=initial_pool
        )

        # Derive the temporal embedding size from the actual encoder output so
        # the pos-embedding and head stay consistent with strided/pooled conv.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_chan, num_time)
            enc = self.cnn_encoder(dummy)  # (1, num_kernel, 1, dim)
        dim = enc.shape[-1]
        self.embed_dim = dim  # temporal tokens entering the first transformer stage

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, num_classes)
        )

    def forward(self, eeg):
        # eeg: (b, chan, time)
        eeg = torch.unsqueeze(eeg, dim=1)   # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)           # (b, num_kernel, 1, dim)
        x = self.to_patch_embedding(x)      # (b, num_kernel, dim)
        x = x + self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


class EEGDeformer(nn.Module):
    """Harness wrapper matching the repo's baseline I/O contract.

    forward expects x of shape (B, 120, 21, 250) and returns raw logits (B, n_classes).
    """

    def __init__(self, n_chans=21, n_classes=2, n_times=30000,
                 temporal_kernel=11, num_kernel=64, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.25,
                 temporal_stride=10, initial_pool=2, **kwargs):
        super().__init__()
        self.model = Deformer(
            num_chan=n_chans, num_time=n_times, temporal_kernel=temporal_kernel,
            num_kernel=num_kernel, num_classes=n_classes, depth=depth, heads=heads,
            mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout,
            temporal_stride=temporal_stride, initial_pool=initial_pool,
        )

    def forward(self, x):
        # (B, 120, 21, 250) -> (B, 21, 30000), exactly as the other baselines do.
        # NB: done in two steps (like the baselines) so size(1) refers to the
        # channel axis *after* the permute. Fusing into one expression would
        # capture the original size(1)=120 and mis-reshape.
        x = x.permute(0, 2, 3, 1)                       # (B, 21, 250, 120)
        x = x.reshape(x.size(0), x.size(1), -1)         # (B, 21, 30000)
        return self.model(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
