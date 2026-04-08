"""
Diffusion-based action policy head for VLA.

Implements a conditional DDPM (Denoising Diffusion Probabilistic Model) that
generates action sequences conditioned on 3D observation features and language
embeddings. Supports both single-step and multi-step (horizon) action prediction.

The diffusion formulation naturally handles multi-modal action distributions,
which is critical for language-conditioned manipulation where a single
instruction may correspond to multiple valid execution strategies.

References:
    - 3D Diffusion Policy (DP3): Ke et al., RSS 2024
    - Diffusion Policy: Chi et al., RSS 2023
    - DDPM: Ho et al., NeurIPS 2020
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal, ICML 2021).

    Produces smoother noise levels compared to linear schedule,
    which is beneficial for low-dimensional action spaces.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0, 0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule (Ho et al., NeurIPS 2020)."""
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResidualBlock(nn.Module):
    """1D residual block with time and condition injection.

    Applies time embedding via addition and condition via FiLM modulation.
    This design allows the denoiser to adapt its behavior based on both
    the diffusion step and the observation context.

    Args:
        in_channels: input channels
        out_channels: output channels
        time_dim: time embedding dimension
        cond_dim: condition embedding dimension
    """

    def __init__(self, in_channels, out_channels, time_dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels),
        )
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )

        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb, cond_emb):
        """
        Args:
            x: (B, C, T) noisy action sequence
            time_emb: (B, time_dim) timestep embedding
            cond_emb: (B, cond_dim) observation condition
        Returns:
            out: (B, C_out, T)
        """
        h = self.conv1(x)

        # Time injection via addition
        h = h + self.time_mlp(time_emb).unsqueeze(-1)

        # Condition injection via FiLM
        scale, shift = self.cond_mlp(cond_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = self.conv2(h)
        return h + self.residual_proj(x)


class ActionDenoiserUNet(nn.Module):
    """Conditional 1D U-Net for denoising action sequences.

    Architecture follows Diffusion Policy (Chi et al.) adapted for
    point cloud conditioned action generation.

    The U-Net operates on the temporal dimension of action sequences,
    using multi-scale processing to capture both local dynamics and
    trajectory-level coherence.

    Args:
        action_dim: dimension of each action step
        horizon: number of action steps to predict
        cond_dim: observation condition dimension
        base_dim: base channel width
        dim_mults: channel multipliers per level
        time_dim: timestep embedding dimension
    """

    def __init__(
        self,
        action_dim,
        horizon=1,
        cond_dim=512,
        base_dim=128,
        dim_mults=(1, 2, 4),
        time_dim=128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Condition projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(action_dim, base_dim, kernel_size=1)

        # Encoder (downsampling path)
        dims = [base_dim * m for m in dim_mults]
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_dim = base_dim
        for dim in dims:
            self.encoder_blocks.append(
                ConditionalResidualBlock(in_dim, dim, time_dim, time_dim)
            )
            self.downsample.append(
                nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
                if horizon > 1
                else nn.Identity()
            )
            in_dim = dim

        # Bottleneck
        self.mid_block = ConditionalResidualBlock(dims[-1], dims[-1], time_dim, time_dim)

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, dim in enumerate(reversed(dims[:-1])):
            up_in = dims[-(i + 1)]
            self.upsample.append(
                nn.ConvTranspose1d(up_in, up_in, kernel_size=4, stride=2, padding=1)
                if horizon > 1
                else nn.Identity()
            )
            self.decoder_blocks.append(
                ConditionalResidualBlock(up_in + dim, dim, time_dim, time_dim)
            )

        out_dim = dims[0] if len(dims) > 1 else dims[-1]
        if horizon > 1 and len(dims) > 1:
            self.final_upsample = nn.ConvTranspose1d(out_dim, out_dim, kernel_size=4, stride=2, padding=1)
        else:
            self.final_upsample = nn.Identity()

        # Output projection back to action_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(out_dim, action_dim, kernel_size=1),
        )

    def forward(self, noisy_action, timestep, condition):
        """
        Args:
            noisy_action: (B, action_dim, horizon) noisy action sequence
            timestep: (B,) diffusion timestep
            condition: (B, cond_dim) observation feature
        Returns:
            noise_pred: (B, action_dim, horizon) predicted noise
        """
        t_emb = self.time_embed(timestep)
        c_emb = self.cond_proj(condition)

        x = self.input_proj(noisy_action)

        # Encoder with skip connections
        skips = []
        for block, down in zip(self.encoder_blocks, self.downsample):
            x = block(x, t_emb, c_emb)
            skips.append(x)
            x = down(x)

        x = self.mid_block(x, t_emb, c_emb)

        # Decoder with skip connections
        for i, (block, up) in enumerate(zip(self.decoder_blocks, self.upsample)):
            x = up(x)
            skip = skips[-(i + 2)]
            # Handle size mismatch from stride-2 ops
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb, c_emb)

        x = self.final_upsample(x)
        if x.shape[-1] != self.horizon:
            x = F.interpolate(x, size=self.horizon, mode="nearest")

        return self.output_proj(x)


class DiffusionActionHead(nn.Module):
    """DDPM-based action generation head.

    During training, adds noise to ground truth actions and learns to denoise.
    During inference, starts from pure noise and iteratively denoises to
    generate actions.

    Supports DDPM and DDIM sampling strategies.

    Args:
        action_dim: dimension of action space
        horizon: number of action steps to predict (1 for reactive, >1 for trajectory)
        cond_dim: conditioning feature dimension
        num_timesteps: number of diffusion steps
        schedule: noise schedule type ("cosine" or "linear")
        base_dim: U-Net base channel width
        clip_sample: whether to clip denoised samples to [-1, 1]
    """

    def __init__(
        self,
        action_dim,
        horizon=1,
        cond_dim=512,
        num_timesteps=100,
        schedule="cosine",
        base_dim=128,
        clip_sample=True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_timesteps = num_timesteps
        self.clip_sample = clip_sample

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            betas = linear_beta_schedule(num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-compute diffusion coefficients
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # Denoiser network
        self.denoiser = ActionDenoiserUNet(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=cond_dim,
            base_dim=base_dim,
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion: add noise to clean actions.

        Args:
            x_start: (B, action_dim, horizon) clean actions
            t: (B,) timestep
            noise: optional pre-generated noise
        Returns:
            x_noisy: (B, action_dim, horizon) noisy actions
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def compute_loss(self, actions, condition):
        """Compute diffusion training loss (simplified MSE on noise prediction).

        Args:
            actions: (B, action_dim) or (B, horizon, action_dim) ground truth actions
            condition: (B, cond_dim) observation features
        Returns:
            loss: scalar training loss
        """
        B = actions.shape[0]

        # Reshape to (B, action_dim, horizon)
        if actions.dim() == 2:
            x_start = actions.unsqueeze(-1)  # (B, action_dim, 1)
        else:
            x_start = actions.permute(0, 2, 1)  # (B, action_dim, horizon)

        noise = torch.randn_like(x_start)
        t = torch.randint(0, self.num_timesteps, (B,), device=actions.device, dtype=torch.long)

        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.denoiser(x_noisy, t, condition)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """Single reverse diffusion step (DDPM).

        Args:
            x: (B, action_dim, horizon) current noisy sample
            t: scalar timestep
            condition: (B, cond_dim)
        Returns:
            x_prev: (B, action_dim, horizon) denoised one step
        """
        B = x.shape[0]
        t_batch = torch.full((B,), t, device=x.device, dtype=torch.long)

        noise_pred = self.denoiser(x, t_batch, condition)

        # Posterior mean
        coef1 = self.posterior_mean_coef1[t]
        coef2 = self.posterior_mean_coef2[t]
        mean = coef1 * (
            (x - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred)
            / torch.sqrt(1.0 - self.betas[t])
        )
        # Simplified: use direct mean formula
        mean = (
            self.sqrt_recip_alphas[t]
            * (x - self.betas[t] / self.sqrt_one_minus_alphas_cumprod[t] * noise_pred)
        )

        if t == 0:
            return mean

        variance = self.posterior_variance[t]
        noise = torch.randn_like(x)
        return mean + torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, condition, num_samples=1):
        """Generate action sequences via iterative denoising.

        Args:
            condition: (B, cond_dim) observation features
            num_samples: number of action samples per observation (for best-of-N)
        Returns:
            actions: (B, action_dim) or (B, horizon, action_dim)
        """
        B = condition.shape[0]
        device = condition.device

        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)

        x = torch.randn(condition.shape[0], self.action_dim, self.horizon, device=device)

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, condition)

        if self.clip_sample:
            x = x.clamp(-1, 1)

        if num_samples > 1:
            x = x.view(B, num_samples, self.action_dim, self.horizon)
            x = x[:, 0]  # Take first sample (or implement best-of-N selection)

        if self.horizon == 1:
            return x.squeeze(-1)  # (B, action_dim)
        return x.permute(0, 2, 1)  # (B, horizon, action_dim)

    @torch.no_grad()
    def ddim_sample(self, condition, ddim_steps=10, eta=0.0):
        """DDIM sampling for faster inference.

        Reduces the number of denoising steps while maintaining quality.
        With eta=0, produces deterministic outputs.

        Args:
            condition: (B, cond_dim)
            ddim_steps: number of DDIM steps (< num_timesteps)
            eta: DDIM stochasticity parameter (0 = deterministic)
        Returns:
            actions: (B, action_dim) or (B, horizon, action_dim)
        """
        B = condition.shape[0]
        device = condition.device

        # Sub-sample timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(B, self.action_dim, self.horizon, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.denoiser(x, t_batch, condition)

            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

            # Predict x_0
            x0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            if self.clip_sample:
                x0_pred = x0_pred.clamp(-1, 1)

            # Direction pointing to x_t
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred

            noise = torch.randn_like(x) if eta > 0 and i + 1 < len(timesteps) else 0.0
            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt + sigma * noise

        if self.horizon == 1:
            return x.squeeze(-1)
        return x.permute(0, 2, 1)


class DeterministicActionHead(nn.Module):
    """Simple deterministic MLP action head for comparison / ablation.

    Provides a non-diffusion baseline to isolate the contribution
    of the diffusion formulation vs. the 3D encoder.

    Args:
        cond_dim: input condition dimension
        action_dim: output action dimension
        hidden_dim: MLP hidden dimension
    """

    def __init__(self, cond_dim=512, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, condition):
        return self.net(condition)

    def compute_loss(self, actions, condition):
        predicted = self.forward(condition)
        return F.smooth_l1_loss(predicted, actions)

    @torch.no_grad()
    def sample(self, condition, **kwargs):
        return self.forward(condition)
