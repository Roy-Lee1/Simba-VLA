"""
Advanced 3D point cloud encoders for VLA.

Provides PointNet++-style set abstraction encoders with hierarchical grouping
and optional cross-attention conditioning on language features.
Designed to replace the simple MLP encoder when richer 3D representations
are needed for policy learning.

References:
    - PointNet++: Deep Hierarchical Feature Learning on Point Sets (Qi et al., NeurIPS 2017)
    - 3D Diffusion Policy (DP3): Generalizable 3D Manipulation (Ke et al., RSS 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Compute pairwise squared Euclidean distance between two point sets.

    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    return (
        torch.sum(src ** 2, dim=-1, keepdim=True)
        + torch.sum(dst ** 2, dim=-1, keepdim=True).transpose(-2, -1)
        - 2.0 * torch.matmul(src, dst.transpose(-2, -1))
    )


def farthest_point_sample(xyz, npoint):
    """Farthest Point Sampling.

    Args:
        xyz: (B, N, 3)
        npoint: number of points to sample
    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def index_points(points, idx):
    """Index into points tensor using idx.

    Args:
        points: (B, N, C)
        idx: (B, S) or (B, S, K)
    Returns:
        indexed: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    return points[batch_indices, idx, :]


def knn_query(k, xyz, new_xyz):
    """K nearest neighbor query.

    Args:
        k: number of neighbors
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, K) indices of K nearest neighbors
    """
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, dim=-1, largest=False, sorted=False)
    return group_idx


def ball_query(radius, nsample, xyz, new_xyz):
    """Ball query: find all points within radius around query points.

    Args:
        radius: search radius
        nsample: max number of points in each ball
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
    Returns:
        group_idx: (B, S, nsample) indices of grouped points
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    dist = square_distance(new_xyz, xyz)
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[dist > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1).values[:, :, :nsample]

    first_point = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = first_point[mask]

    return group_idx


class SetAbstractionLayer(nn.Module):
    """PointNet++ Set Abstraction layer with FPS + ball query + PointNet.

    Downsamples the point cloud and extracts local features.

    Args:
        npoint: number of output centroids (None for global aggregation)
        radius: ball query radius
        nsample: max points per group
        in_channel: input feature dimension (including xyz if concat)
        mlp_channels: list of MLP output dimensions
        use_knn: use KNN instead of ball query (for variable-density clouds)
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels, use_knn=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_knn = use_knn

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, N, C) point features (can be None)
        Returns:
            new_xyz: (B, npoint, 3) downsampled coordinates
            new_features: (B, npoint, D) output features
        """
        B, N, _ = xyz.shape

        if self.npoint is None:
            # Global aggregation
            new_xyz = torch.zeros(B, 1, 3, device=xyz.device)
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            if features is not None:
                grouped_features = features.unsqueeze(1)  # (B, 1, N, C)
                grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

            if self.use_knn:
                group_idx = knn_query(self.nsample, xyz, new_xyz)
            else:
                group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.unsqueeze(2)  # local coordinates

            if features is not None:
                grouped_features = index_points(features, group_idx)
                grouped = torch.cat([grouped_xyz, grouped_features], dim=-1)
            else:
                grouped = grouped_xyz

        # (B, S, K, C) -> (B, C, K, S) for Conv2d
        grouped = grouped.permute(0, 3, 2, 1).contiguous()

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped = F.gelu(bn(conv(grouped)))

        new_features = grouped.max(dim=2).values  # (B, D, S)
        new_features = new_features.permute(0, 2, 1).contiguous()  # (B, S, D)

        if self.npoint is None:
            new_features = new_features.squeeze(1)  # (B, D) for global
            new_xyz = new_xyz.squeeze(1)

        return new_xyz, new_features


class PointNetPPEncoder(nn.Module):
    """PointNet++ encoder with hierarchical set abstraction.

    Progressively downsamples and extracts multi-scale 3D features.
    Outputs a global feature vector suitable for policy conditioning.

    Args:
        output_dim: final output feature dimension
        use_knn: use KNN instead of ball query
    """

    def __init__(self, output_dim=512, use_knn=True):
        super().__init__()
        self.sa1 = SetAbstractionLayer(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp_channels=[64, 64, 128],
            use_knn=use_knn,
        )
        self.sa2 = SetAbstractionLayer(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp_channels=[128, 128, 256],
            use_knn=use_knn,
        )
        self.sa3 = SetAbstractionLayer(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp_channels=[256, 512, output_dim],
            use_knn=use_knn,
        )

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            global_feature: (B, output_dim)
            multi_scale: dict with intermediate features for skip connections
        """
        l1_xyz, l1_features = self.sa1(xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        _, l3_features = self.sa3(l2_xyz, l2_features)

        multi_scale = {
            "l1": (l1_xyz, l1_features),
            "l2": (l2_xyz, l2_features),
        }
        return l3_features, multi_scale


class ConditionedPointEncoder(nn.Module):
    """Point cloud encoder with language-conditioned feature modulation.

    Combines PointNet++ hierarchical features with FiLM conditioning
    from text embeddings, plus an optional cross-attention layer for
    more expressive text-geometry interaction.

    Args:
        output_dim: output feature dimension
        text_dim: text feature dimension for conditioning
        use_cross_attention: whether to use cross-attention fusion
        num_heads: number of attention heads (if cross-attention)
    """

    def __init__(self, output_dim=512, text_dim=256, use_cross_attention=True, num_heads=8):
        super().__init__()
        self.backbone = PointNetPPEncoder(output_dim=output_dim)

        # FiLM modulation: scale and shift features based on text
        self.film_generator = nn.Sequential(
            nn.Linear(text_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * 2),
        )

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(output_dim)
            self.text_proj = nn.Linear(text_dim, output_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, xyz, text_feature):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            text_feature: (B, text_dim) text embedding
        Returns:
            features: (B, output_dim) conditioned 3D features
        """
        global_feature, multi_scale = self.backbone(xyz)

        # FiLM conditioning
        gamma, beta = self.film_generator(text_feature).chunk(2, dim=-1)
        conditioned = global_feature * (1.0 + gamma) + beta

        # Cross-attention: geometry attends to text
        if self.use_cross_attention:
            text_kv = self.text_proj(text_feature).unsqueeze(1)  # (B, 1, D)
            query = conditioned.unsqueeze(1)  # (B, 1, D)
            attn_out, _ = self.cross_attn(query, text_kv, text_kv)
            conditioned = self.cross_norm(conditioned + attn_out.squeeze(1))

        return self.output_proj(conditioned)


class DualBranchEncoder(nn.Module):
    """Dual-branch 3D encoder for partial and completed point clouds.

    Encodes partial and completed point clouds with shared or separate
    backbones, then fuses them with text features.

    The dual-branch design allows the model to leverage both the raw
    partial observation (which preserves real sensor characteristics)
    and the completed geometry (which provides fuller state estimation).

    Args:
        feature_dim: per-branch output dimension
        text_dim: text feature dimension
        share_backbone: whether to share weights between branches
    """

    def __init__(self, feature_dim=512, text_dim=256, share_backbone=False):
        super().__init__()
        self.partial_encoder = ConditionedPointEncoder(
            output_dim=feature_dim, text_dim=text_dim,
        )
        if share_backbone:
            self.completed_encoder = self.partial_encoder
        else:
            self.completed_encoder = ConditionedPointEncoder(
                output_dim=feature_dim, text_dim=text_dim,
            )

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, partial_xyz, completed_xyz, text_feature):
        """
        Args:
            partial_xyz: (B, N, 3)
            completed_xyz: (B, M, 3) or None
            text_feature: (B, text_dim)
        Returns:
            fused: (B, feature_dim) fused observation feature
        """
        partial_feat = self.partial_encoder(partial_xyz, text_feature)

        if completed_xyz is None:
            return partial_feat

        completed_feat = self.completed_encoder(completed_xyz, text_feature)
        return self.fusion(torch.cat([partial_feat, completed_feat], dim=-1))
