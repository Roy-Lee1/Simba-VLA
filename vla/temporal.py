"""
Temporal observation aggregation for multi-step VLA.

Provides modules to aggregate information across multiple observation
timesteps, which is critical for tasks requiring motion history or
velocity estimation. Without temporal context, the policy can only
make reactive decisions; with it, the policy can reason about dynamics.

References:
    - Diffusion Policy (Chi et al., RSS 2023): observation horizon concept
    - ACT (Zhao et al., RSS 2023): action chunking with transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalTransformerAggregator(nn.Module):
    """Transformer-based temporal aggregation of observation features.

    Processes a sequence of observation features (from multiple timesteps)
    using self-attention, then pools into a single conditioning vector.

    Each observation feature includes its 3D point cloud encoding, text
    encoding, and optional proprioceptive state (e.g., joint angles,
    end-effector pose).

    Args:
        feature_dim: per-timestep feature dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        max_horizon: maximum observation history length
        dropout: attention dropout
    """

    def __init__(self, feature_dim=512, num_layers=2, num_heads=8, max_horizon=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_horizon = max_horizon

        # Learnable temporal position embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_horizon, feature_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Aggregation: learnable query token or mean pool
        self.agg_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, features, mask=None):
        """
        Args:
            features: (B, T, D) observation features across T timesteps
            mask: (B, T) boolean, True = valid timestep
        Returns:
            aggregated: (B, D) temporally aggregated feature
        """
        B, T, D = features.shape
        assert T <= self.max_horizon, f"T={T} exceeds max_horizon={self.max_horizon}"

        # Add temporal position encoding
        features = features + self.temporal_pos_embed[:, :T, :]

        # Prepend aggregation token
        agg = self.agg_token.expand(B, -1, -1)
        tokens = torch.cat([agg, features], dim=1)  # (B, 1+T, D)

        # Adjust mask for agg token (always valid)
        if mask is not None:
            agg_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([agg_mask, mask], dim=1)
            # TransformerEncoder expects src_key_padding_mask where True = IGNORE
            padding_mask = ~mask
        else:
            padding_mask = None

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        aggregated = encoded[:, 0, :]  # Take the agg token output

        return self.output_norm(aggregated)


class ProprioceptionEncoder(nn.Module):
    """Encode proprioceptive state (joint angles, EE pose, gripper state).

    Maps low-dimensional robot state to the same feature space as
    visual observations for fusion.

    Args:
        proprio_dim: input proprioceptive state dimension
            (e.g., 7 for joint angles, or 7+6 for joints + EE pose)
        output_dim: output feature dimension
    """

    def __init__(self, proprio_dim=13, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state):
        """
        Args:
            state: (B, proprio_dim) or (B, T, proprio_dim)
        Returns:
            features: same shape with last dim = output_dim
        """
        return self.net(state)


class ObservationFusion(nn.Module):
    """Fuse multi-modal observation into a single feature vector.

    Combines 3D visual features, text features, and optional
    proprioceptive features into a unified observation representation.

    When temporal context is available, aggregates across timesteps
    before producing the final conditioning vector.

    Args:
        visual_dim: 3D point cloud feature dimension
        text_dim: text feature dimension
        proprio_dim: proprioceptive state dimension (0 to disable)
        output_dim: fused output dimension
        temporal_horizon: number of observation history steps (1 = no temporal)
    """

    def __init__(
        self,
        visual_dim=512,
        text_dim=256,
        proprio_dim=0,
        output_dim=512,
        temporal_horizon=1,
    ):
        super().__init__()
        total_dim = visual_dim + text_dim
        self.use_proprio = proprio_dim > 0
        self.use_temporal = temporal_horizon > 1

        if self.use_proprio:
            self.proprio_encoder = ProprioceptionEncoder(proprio_dim, output_dim=128)
            total_dim += 128

        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

        if self.use_temporal:
            self.temporal_agg = TemporalTransformerAggregator(
                feature_dim=output_dim,
                num_layers=2,
                num_heads=8,
                max_horizon=temporal_horizon,
            )

    def forward(self, visual_feature, text_feature, proprio_state=None, temporal_mask=None):
        """
        Args:
            visual_feature: (B, visual_dim) or (B, T, visual_dim)
            text_feature: (B, text_dim) or (B, T, text_dim)
            proprio_state: (B, proprio_dim) or (B, T, proprio_dim), optional
            temporal_mask: (B, T) boolean, optional
        Returns:
            fused: (B, output_dim)
        """
        components = [visual_feature, text_feature]

        if self.use_proprio and proprio_state is not None:
            components.append(self.proprio_encoder(proprio_state))

        fused = torch.cat(components, dim=-1)
        fused = self.fusion_mlp(fused)

        if self.use_temporal and fused.dim() == 3:
            fused = self.temporal_agg(fused, mask=temporal_mask)

        return fused
