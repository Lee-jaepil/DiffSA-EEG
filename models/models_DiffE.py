import math
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# Swish activation function
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
    """Weight Standardization for Conv1d.
    Standardizes weights (zero mean, unit variance) before convolution,
    which works synergistically with GroupNorm.
    Reference: https://arxiv.org/abs/1903.10520
    """
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


class ResidualConvBlock(nn.Module):
    def __init__(self, inc: int, outc: int, kernel_size: int, stride=1, gn=8):
        super().__init__()
        """Standard ResNet style convolutional block"""
        self.same_channels = inc == outc
        self.ks = kernel_size
        num_groups = gn if outc % gn == 0 else 1
        self.conv = nn.Sequential(
            WeightStandardizedConv1d(inc, outc, self.ks, stride, get_padding(self.ks)),
            nn.GroupNorm(gn, outc),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv(x)
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
        new_L = x.size(-1)  # Length reduced by factor
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


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """Generic one-layer FC NN for embedding"""
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.PReLU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels, n_feat=64):
        super(ConditionalUNet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.d1_out = n_feat * 1
        self.d2_out = n_feat * 2
        self.d3_out = n_feat * 4

        self.u1_out = n_feat * 2
        self.u2_out = n_feat * 1
        self.u3_out = n_feat * 1
        self.u4_out = in_channels

        self.sin_emb = SinusoidalPosEmb(n_feat)

        self.down1 = UnetDown(in_channels, self.d1_out, 1, gn=self.adjust_gn(self.d1_out), factor=2)
        self.down2 = UnetDown(self.d1_out, self.d2_out, 1, gn=self.adjust_gn(self.d2_out), factor=2)
        self.down3 = UnetDown(self.d2_out, self.d3_out, 1, gn=self.adjust_gn(self.d3_out), factor=2)

        self.up2 = UnetUp(self.d3_out, self.u1_out, 1, gn=self.adjust_gn(self.u1_out), factor=2)
        self.up3 = UnetUp(self.u1_out + self.d2_out, self.u2_out, 1, gn=self.adjust_gn(self.u2_out), factor=2)
        self.up4 = UnetUp(self.u2_out + self.d1_out, self.u3_out, 1, gn=self.adjust_gn(self.u3_out), factor=2)
        self.out = nn.Conv1d(self.u3_out + in_channels, in_channels, 1)

    def adjust_gn(self, out_channels, gn=8):
        return gn if out_channels % gn == 0 else 1

    def forward(self, x, t):
        B, T, C, L = x.size()

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        t = t.view(B * T)
        temb = self.sin_emb(t).view(B, T, self.n_feat, 1)

        up1 = self.up2(down3)

        # Interpolate all tensors to original input size
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


class Encoder(nn.Module):
    def __init__(self, in_channels, dim=64):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.e1_out = dim
        self.e2_out = dim
        self.e3_out = dim

        self.down1 = UnetDown(in_channels, self.e1_out, 1, gn=8, factor=2)
        self.down2 = UnetDown(self.e1_out, self.e2_out, 1, gn=8, factor=2)
        self.down3 = UnetDown(self.e2_out, self.e3_out, 1, gn=8, factor=2)

        self.avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.max_pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.act = nn.Tanh()

    def forward(self, x0):
        B, T, C, L = x0.size()

        dn1 = self.down1(x0)
        dn2 = self.down2(dn1)
        dn3 = self.down3(dn2)

        # Average over the time dimension
        z = self.avg_pooling(dn3.view(B * T, self.e3_out, -1)).view(B, T, self.e3_out).mean(dim=1)

        down = (dn1, dn2, dn3)
        out = (down, z)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, n_feat=128, encoder_dim=64, n_classes=2):
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
        self.u3_out = in_channels

        self.up1 = UnetUp(self.d3_out + self.e3_out, self.u1_out, 1, gn=8, factor=2)
        self.up2 = UnetUp(self.u1_out + self.d2_out, self.u2_out, 1, gn=8, factor=2)

        # Final up block with additional Conv1d layers
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(self.u2_out + self.d1_out + in_channels * 2, self.u3_out * 2, 1, 1, 0),
            nn.Conv1d(self.u3_out * 2, self.u3_out * 2, 1, 1, 0)
        )

    def forward(self, x0, encoder_out, diffusion_out):
        if x0.dim() == 3:
            x0 = x0.unsqueeze(1)
        B, T, C, L = x0.size()

        # Encoder output
        down, z = encoder_out
        dn1, dn2, dn3 = [d.view(B, T, -1, d.size(-1)) for d in down]

        # DDPM output
        x_hat, down_ddpm, up, t = diffusion_out
        dn11, dn22, dn33 = [d.view(B, T, -1, d.size(-1)) for d in down_ddpm]

        # Up sampling
        up1 = self.up1(torch.cat([dn3, dn33.detach()], 2))
        up2 = self.up2(torch.cat([up1, dn22.detach()], 2))

        # Interpolate all tensors to original input size
        up2 = F.interpolate(up2.view(B * T, -1, up2.size(-1)), size=L, mode='linear', align_corners=False)
        up2 = up2.view(B, T, -1, L)

        dn11_interp = F.interpolate(dn11.view(B * T, -1, dn11.size(-1)), size=L, mode='linear', align_corners=False)
        dn11_interp = dn11_interp.view(B, T, -1, L)

        concat_tensor = torch.cat([
            x0.view(B * T, C, L),
            x_hat.view(B * T, C, L).detach(),
            up2.view(B * T, -1, L),
            dn11_interp.view(B * T, -1, L)
        ], 1)

        out = self.up3(concat_tensor)

        # Adjust channels to C and length to L
        out = out[:, :C, :L]
        out = out.view(B, T, C, L)

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
        B, T, C, L = x.size()  # Batch, Time, Channels, Length
        _ts = torch.randint(1, self.n_T, (B, T)).to(self.device)
        noise = torch.randn_like(x)

        sqrtab = self.sqrtab[_ts].view(B, T, 1, 1)
        sqrtmab = self.sqrtmab[_ts].view(B, T, 1, 1)

        x_t = sqrtab * x + sqrtmab * noise  # Add Gaussian noise at timestep _ts
        times = _ts / self.n_T

        output, down, up = self.nn_model(x_t, times)  # Denoise via U-Net

        output = F.interpolate(output.view(B * T, -1, output.shape[-1]), size=x.shape[-1])
        output = output.view(B, T, C, -1)

        return output, down, up, noise, times
