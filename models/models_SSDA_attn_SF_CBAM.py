import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class WeightStandardizedConv1d(nn.Conv1d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x)
        x_out = self.spatial_attention(x_out)
        return x_out

class ResidualConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size, stride=1, gn=8):
        super().__init__()
        self.same_channels = inc == outc
        num_groups = gn if outc % gn == 0 else 1
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, kernel_size, stride, get_padding(kernel_size)),
            nn.GroupNorm(num_groups, outc),
            nn.PReLU(),
        )
        self.cbam = CBAM(outc)  # Add CBAM

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.cbam(x1)  # Apply CBAM
        if self.same_channels:
            out = (x + x1) / 2
        else:
            out = x1
        return out

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetDown, self).__init__()
        self.pool = nn.MaxPool1d(factor)
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        B, T, C, L = x.size()
        x = x.view(B * T, C, L)
        x = self.layer(x)
        x = self.pool(x)
        new_L = x.size(-1)
        x = x.view(B, T, -1, new_L)
        return x

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, gn=8, factor=2):
        super(UnetUp, self).__init__()
        self.pool = nn.Upsample(scale_factor=factor, mode="nearest")
        self.layer = ResidualConvBlock(in_channels, out_channels, kernel_size, gn=gn)

    def forward(self, x):
        B, T, C, L = x.size()
        x = x.view(B * T, C, L)
        x = self.pool(x)
        x = self.layer(x)
        new_C = x.size(1)
        new_L = x.size(2)
        x = x.view(B, T, new_C, new_L)
        return x

class SelfAttention(nn.Module):
    def __init__(self, channels, reduction_factor=8):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.query = nn.Conv1d(channels, channels // reduction_factor, 1)
        self.key = nn.Conv1d(channels, channels // reduction_factor, 1)
        self.value = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = torch.bmm(q.transpose(1, 2), k)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))
        return out + x

class SparseEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseEncoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SparseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseDecoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class SpatialFilterLayer(nn.Module):
    def __init__(self, n_channels, n_components, regularization=1e-4):
        super(SpatialFilterLayer, self).__init__()
        # Set basic parameters
        self.n_channels = n_channels          # Number of input channels
        self.n_components = n_components      # Number of output components
        self.regularization = regularization  # Regularization strength

        # SVD-based weight initialization
        self.spatial_filter = nn.Parameter(
            self._initialize_weights(n_channels, n_components)
        )

    def _initialize_weights(self, n_channels, n_components):
        # Orthogonal initialization using SVD
        U, _, _ = torch.svd(torch.randn(n_channels, n_channels))
        # Select only the first n_components orthogonal vectors
        return U[:, :n_components]

    def forward(self, x):
        # Decompose input tensor dimensions
        B, T, C, L = x.size()  # Batch, Time, Channels, Length
        x = x.view(B * T, C, L)

        # Filter normalization: normalize each component vector to a unit vector
        normalized_filter = F.normalize(self.spatial_filter, dim=0)

        # Apply spatial filtering
        x_filtered = torch.matmul(normalized_filter.t(), x)

        # Compute regularization loss only during training
        if self.training:
            # L2 norm-based regularization loss
            reg_loss = self.regularization * torch.norm(self.spatial_filter)
            self.reg_loss = reg_loss  # Save for later addition to total loss

        # Reconstruct result tensor
        return x_filtered.view(B, T, self.n_components, L)

class Encoder(nn.Module):
    def __init__(self, in_channels, dim=64, n_spatial_components=8):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.e1_out = dim
        self.e2_out = dim
        self.e3_out = dim

        self.spatial_filter = SpatialFilterLayer(in_channels, n_spatial_components)

        self.down1 = UnetDown(n_spatial_components, self.e1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.e2_out, self.e3_out, 1, gn=8, factor=2)

        self.ssda_layers = nn.ModuleList([
            SparseEncoder(self.e3_out, self.e3_out),
            SparseEncoder(self.e3_out, self.e3_out),
            SparseEncoder(self.e3_out, self.e3_out)
        ])

        self.attn1 = SelfAttention(self.e1_out)
        self.attn2 = SelfAttention(self.e2_out)
        self.attn3 = SelfAttention(self.e3_out)

        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.act = nn.Tanh()

    def add_noise(self, x, noise_factor=0.05):
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def forward(self, x0):
        B, T, C, L = x0.size()

        # Apply Spatial Filtering
        x0 = self.spatial_filter(x0)

        dn1 = self.down1(x0)
        dn1 = self.attn1(dn1.view(B * T, -1, dn1.shape[-1])).view(B, T, -1, dn1.shape[-1])

        dn2 = self.down2(dn1)
        dn2 = self.attn2(dn2.view(B * T, -1, dn2.shape[-1])).view(B, T, -1, dn2.shape[-1])

        dn3 = self.down3(dn2)
        dn3 = self.attn3(dn3.view(B * T, -1, dn3.shape[-1])).view(B, T, -1, dn3.shape[-1])

        x = dn3.view(B * T, self.e3_out, -1)
        for layer in self.ssda_layers:
            x = self.add_noise(x)
            x = layer(x)

        z = self.avg_pooling(x).view(B, T, self.e3_out).mean(dim=1)

        down = (dn1, dn2, dn3)
        out = (down, z)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=128, encoder_dim=64, n_classes=2, n_spatial_components=8):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes
        self.e1_out = encoder_dim
        self.e2_out = encoder_dim
        self.e3_out = encoder_dim
        self.d1_out = n_feat
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 4
        self.u1_out = n_feat * 2
        self.u2_out = n_feat
        self.u3_out = n_spatial_components

        self.ssda_layers = nn.ModuleList([
            SparseDecoder(self.e3_out, self.e3_out),
            SparseDecoder(self.e3_out, self.e3_out),
            SparseDecoder(self.e3_out, self.e3_out)
        ])

        self.up1 = UnetUp(self.d3_out + self.e3_out, self.u1_out, 1, gn=8, factor=2)
        self.up2 = UnetUp(self.u1_out + self.d2_out, self.u2_out, 1, gn=8, factor=2)

        # Calculate the total number of channels after concatenation
        total_channels = self.u2_out + self.d1_out + n_spatial_components * 2

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=1, mode="nearest"),
            nn.Conv1d(total_channels, self.u3_out * 2, 1, 1, 0),
            nn.Conv1d(self.u3_out * 2, self.u3_out * 2, 1, 1, 0)
        )

        self.attn1 = SelfAttention(self.u1_out)
        self.attn2 = SelfAttention(self.u2_out)
        self.attn3 = SelfAttention(self.u3_out * 2)

        # Inverse Spatial Filter layer to reconstruct original channels
        self.inverse_spatial_filter = nn.Linear(n_spatial_components * 2, in_channels)
        self.spatial_filter = SpatialFilterLayer(in_channels, n_spatial_components)

    def forward(self, x0, encoder_out, diffusion_out):
        if x0.dim() == 3:
            x0 = x0.unsqueeze(1)
        B, T, C, L = x0.size()

        down, z = encoder_out
        dn1, dn2, dn3 = [d.view(B, T, -1, d.size(-1)) for d in down]

        x_hat, down_ddpm, up, t = diffusion_out
        dn11, dn22, dn33 = [d.view(B, T, -1, d.size(-1)) for d in down_ddpm]

        x = dn3.view(B * T, self.e3_out, -1)
        for layer in self.ssda_layers:
            x = layer(x)

        x = x.view(B, T, self.e3_out, -1)

        up1 = self.up1(torch.cat([x, dn33.detach()], 2))
        up1 = self.attn1(up1.view(B * T, -1, up1.shape[-1])).view(B, T, -1, up1.shape[-1])

        up2 = self.up2(torch.cat([up1, dn22.detach()], 2))
        up2 = self.attn2(up2.view(B * T, -1, up2.shape[-1])).view(B, T, -1, up2.shape[-1])

        up2 = F.interpolate(up2.view(B * T, -1, up2.size(-1)), size=L, mode='linear', align_corners=False)
        up2 = up2.view(B, T, -1, L)

        dn11_interp = F.interpolate(dn11.view(B * T, -1, dn11.size(-1)), size=L, mode='linear', align_corners=False)
        dn11_interp = dn11_interp.view(B, T, -1, L)

        # Transform x0 and x_hat to Spatial Filter space
        x0_sf = self.spatial_filter(x0)
        x_hat_sf = self.spatial_filter(x_hat)

        concat_tensor = torch.cat([
            x0_sf.view(B * T, -1, L),
            x_hat_sf.view(B * T, -1, L).detach(),
            up2.view(B * T, -1, L),
            dn11_interp.view(B * T, -1, L)
        ], 1)

        out = self.up3(concat_tensor)
        out = self.attn3(out)

        # Reshape tensor to correct form
        B, C, L = out.shape
        out = out.view(B // T, T, C, L)

        # Apply inverse Spatial Filter to reconstruct original channels
        out = out.permute(0, 1, 3, 2)  # [B, T, L, n_spatial_components * 2]
        out = out.reshape(B // T * T * L, self.u3_out * 2)  # [B * T * L, n_spatial_components * 2]
        out = self.inverse_spatial_filter(out)  # [B * T * L, in_channels]
        out = out.view(B // T, T, L, self.in_channels)  # [B, T, L, in_channels]
        out = out.permute(0, 1, 3, 2)  # [B, T, in_channels, L]

        # Adjust output size to match input
        out = F.interpolate(out.view(B // T * T, self.in_channels, L), size=x0.shape[-1], mode='linear', align_corners=False)
        out = out.view(B // T, T, self.in_channels, x0.shape[-1])

        return out

class DiffE(nn.Module):
    def __init__(self, encoder, decoder, fc):
        super(DiffE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc = fc

    def forward(self, x0, ddpm_out):
        encoder_out = self.encoder(x0)
        decoder_out = self.decoder(x0, encoder_out, ddpm_out)
        fc_out = self.fc(encoder_out[1])
        return decoder_out, fc_out

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, latent_dim, emb_dim):
        super().__init__()
        self.linear_out = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=latent_dim),
            nn.GroupNorm(4, latent_dim),
            nn.PReLU(),
            nn.Linear(in_features=latent_dim, out_features=emb_dim),
        )

    def forward(self, x):
        x = self.linear_out(x)
        return x

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def ddpm_schedules(beta1, beta2, T):
    beta_t = cosine_beta_schedule(T, s=0.008).float()
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)

    return {
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device

    def forward(self, x):
        B, T, C, L = x.size()
        _ts = torch.randint(1, self.n_T, (B, T)).to(self.device)
        noise = torch.randn_like(x)

        sqrtab = self.sqrtab[_ts].view(B, T, 1, 1)
        sqrtmab = self.sqrtmab[_ts].view(B, T, 1, 1)

        x_t = sqrtab * x + sqrtmab * noise
        times = _ts / self.n_T

        output, down, up = self.nn_model(x_t, times)

        output = F.interpolate(output.view(B * T, -1, output.shape[-1]), size=x.shape[-1])
        output = output.view(B, T, C, -1)

        return output, down, up, noise, times

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat=32):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 4

        self.u1_out = n_feat * 2
        self.u2_out = n_feat
        self.u3_out = n_feat
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=self.adjust_gn(self.d1_out), factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=self.adjust_gn(self.d2_out), factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=self.adjust_gn(self.d3_out), factor=2)

        self.attn1 = SelfAttention(self.d2_out)
        self.attn2 = SelfAttention(self.d3_out)

        self.up2 = UnetUp(self.d3_out, self.u1_out, 1, gn=self.adjust_gn(self.u1_out), factor=2)
        self.up3 = UnetUp(self.u1_out + self.d2_out, self.u2_out, 1, gn=self.adjust_gn(self.u2_out), factor=2)
        self.up4 = UnetUp(self.u2_out + self.d1_out, self.u3_out, 1, gn=self.adjust_gn(self.u3_out), factor=2)

        self.attn3 = SelfAttention(self.u1_out)

        self.out = nn.Conv1d(self.u3_out + in_channels, in_channels, 1)

    def adjust_gn(self, out_channels, gn=8):
        return gn if out_channels % gn == 0 else 1

    def forward(self, x, t):
        B, T, C, L = x.size()

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down2 = self.attn1(down2.view(B * T, -1, down2.shape[-1])).view(B, T, -1, down2.shape[-1])

        down3 = self.down3(down2)
        down3 = self.attn2(down3.view(B * T, -1, down3.shape[-1])).view(B, T, -1, down3.shape[-1])

        t = t.view(B * T)
        temb = self.sin_emb(t).view(B, T, self.n_feat, 1)

        up1 = self.up2(down3)
        up1 = self.attn3(up1.view(B * T, -1, up1.shape[-1])).view(B, T, -1, up1.shape[-1])

        up1 = F.interpolate(up1.view(B * T, -1, up1.shape[-1]), size=L).view(B, T, -1, L)
        down2_resized = F.interpolate(down2.view(B * T, -1, down2.shape[-1]), size=L).view(B, T, -1, L)

        up2 = self.up3(torch.cat([up1, down2_resized], 2))

        up2 = F.interpolate(up2.view(B * T, -1, up2.shape[-1]), size=L).view(B, T, -1, L)
        down1_resized = F.interpolate(down1.view(B * T, -1, down1.shape[-1]), size=L).view(B, T, -1, L)

        up3 = self.up4(torch.cat([up2, down1_resized], 2))

        up3 = up3.view(B * T, -1, up3.size(-1))
        up3 = F.interpolate(up3, size=L, mode='linear', align_corners=False)
        up3 = up3.view(B, T, -1, L)

        out = self.out(torch.cat([up3.view(B * T, -1, L), x.view(B * T, -1, L)], 1))

        return out.view(B, T, C, L), (down1, down2, down3), (up1, up2, up3)
