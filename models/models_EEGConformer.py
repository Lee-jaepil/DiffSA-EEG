"""EEG-Conformer: Convolutional Transformer for EEG Decoding and Visualization.

Faithful PyTorch re-implementation of Song et al., IEEE TNSRE 2023.
Architecture (temporal conv -> spatial conv -> token projection -> N transformer
encoder blocks with multi-head self-attention -> classification head) is based on
the official implementation:
    https://github.com/eeyhsong/EEG-Conformer  (conformer.py)

Adapted here as a single nn.Module for a clinical-EEG harness whose inputs are
(B, 120, 21, 250) and are reshaped to a single 21-channel, 30000-sample trial.
"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Long-signal hyperparameter adaptation (documented deviations from reference)
# ---------------------------------------------------------------------------
# Reference used ~1000-sample inputs and produced ~61 tokens. Our reshaped input
# is 30000 samples (250 Hz x 120 s). Keeping the reference temporal kernel (1,25)
# stride 1 and pooling stride 15 would yield ~1995 tokens, which is too many for
# self-attention on one 48 GB GPU at batch 32.
#
# Only the average-pooling STRIDE is scaled; every other block is left structurally
# identical to the reference. The reference produced ~61 tokens; over our 30000-sample
# signal we target a comparable token *scale* (~300, each token spanning ~0.4 s of the
# 120 s recording) rather than the raw reference stride, which is both closer in spirit
# to the reference's token regime AND keeps the N x N self-attention tractable in memory
# (stride 25 -> 1197 tokens needs >32 GB and OOMs at batch 32; stride 100 -> ~300 tokens
# is a few GB). This yields:
#   time after temporal conv (1,25) stride 1 : 30000 - 25 + 1         = 29976
#   time after AvgPool (1,75) stride 100      : (29976-75)//100 + 1    = 300 tokens
POOL_KERNEL = 75
POOL_STRIDE = 100         # reference: 15  -> widened to 100 for long-signal input (~300 tokens)
TEMPORAL_KERNEL = 25      # reference value, unchanged
EMB_SIZE = 40             # reference value, unchanged
DEPTH = 6                 # reference value, unchanged
NUM_HEADS = 10            # reference value, unchanged


class PatchEmbedding(nn.Module):
    """Temporal conv + spatial conv + projection to a token sequence."""

    def __init__(self, n_chans=21, emb_size=EMB_SIZE):
        super().__init__()
        # Shallow conv net: temporal filtering then spatial (across electrodes).
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, TEMPORAL_KERNEL), (1, 1)),
            nn.Conv2d(40, 40, (n_chans, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, POOL_KERNEL), (1, POOL_STRIDE)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x):
        # x: (B, 1, n_chans, n_times)
        x = self.shallownet(x)
        x = self.projection(x)
        return x  # (B, N_tokens, emb_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=EMB_SIZE, num_heads=NUM_HEADS, dropout=0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)
        if mask is not None:
            fill_value = torch.finfo(energy.dtype).min
            energy.masked_fill_(~mask, fill_value)
        # Reference scales by emb_size**0.5 (not head_dim); kept for faithfulness.
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.5):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=EMB_SIZE, num_heads=NUM_HEADS, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=DEPTH, emb_size=EMB_SIZE):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Module):
    """Flatten all tokens then a 3-layer MLP, exactly as the reference `fc` head.

    The reference hardcodes the flattened dim (2440 for its 61 tokens). Here it is
    supplied at construction (`flatten_dim`) since our token count differs.
    """

    def __init__(self, flatten_dim, n_classes=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return self.fc(x)


class EEGConformer(nn.Module):
    """EEG-Conformer wrapper matching the clinical-EEG harness I/O contract.

    Input : (B, 120, 21, 250) float32
    Output: (B, n_classes) raw logits (no softmax)
    """

    def __init__(self, n_chans=21, n_classes=2, n_times=30000,
                 emb_size=EMB_SIZE, depth=DEPTH, **kwargs):
        super().__init__()
        self.n_chans = n_chans
        self.patch_embedding = PatchEmbedding(n_chans=n_chans, emb_size=emb_size)
        self.transformer = TransformerEncoder(depth=depth, emb_size=emb_size)

        # Determine the flattened token dimension with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            tokens = self.patch_embedding(dummy)
            self.n_tokens = tokens.shape[1]
            flatten_dim = tokens.shape[1] * tokens.shape[2]
        self.classification_head = ClassificationHead(flatten_dim, n_classes)

    def forward(self, x):
        # Harness reshape: (B,120,21,250) -> (B,21,30000), same as the baselines.
        # Done in two steps (like models_EEGNet) so size(1) is the channel dim (21).
        x = x.permute(0, 2, 3, 1)                       # (B, 21, 250, 120)
        x = x.reshape(x.size(0), x.size(1), -1)         # (B, 21, 30000)
        # -> (B, 1, n_chans, n_times) as the Conformer expects.
        x = x.unsqueeze(1)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classification_head(x)
        return x
