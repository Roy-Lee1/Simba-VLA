import importlib.util
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from utils.config import cfg_from_yaml_file


def resolve_device(device_name):
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def sample_points_torch(points, num_points):
    if num_points is None or num_points <= 0 or points.shape[1] == num_points:
        return points

    total_points = points.shape[1]
    if total_points > num_points:
        indices = torch.linspace(0, total_points - 1, steps=num_points, device=points.device)
        indices = indices.round().long()
        return points.index_select(1, indices)

    repeats = num_points // total_points
    remainder = num_points % total_points
    expanded = points.repeat(1, repeats, 1) if repeats > 0 else points[:, :0, :]
    if remainder > 0:
        expanded = torch.cat([expanded, points[:, :remainder, :]], dim=1)
    return expanded


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids, token_mask):
        embedded = self.embedding(token_ids)
        lengths = token_mask.long().sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.norm(self.dropout(hidden))


class PointCloudEncoder(nn.Module):
    def __init__(self, hidden_dim=256, text_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.GELU(),
        )
        self.text_condition = nn.Sequential(
            nn.Linear(text_hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, points, text_feature):
        tokens = self.point_mlp(points)
        gamma, beta = self.text_condition(text_feature).chunk(2, dim=-1)
        tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        tokens = self.norm(tokens + self.residual(tokens))
        max_feature = tokens.max(dim=1).values
        mean_feature = tokens.mean(dim=1)
        return torch.cat([max_feature, mean_feature], dim=-1)


def _load_simba_builder_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    builder_path = os.path.join(repo_root, "tools", "builder.py")
    spec = importlib.util.spec_from_file_location("simba_builder", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Simba builder module from {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SimbaCompletionWrapper(nn.Module):
    def __init__(
        self,
        enabled=False,
        config_path=None,
        checkpoint_path=None,
        device="auto",
        output_points=2048,
        freeze=True,
    ):
        super().__init__()
        self.enabled = enabled
        self.output_points = output_points
        self.freeze = freeze
        self.runtime_device = resolve_device(device)
        self.completion_model = None

        if not enabled:
            return

        if not config_path:
            raise ValueError("completion.config_path is required when use_completion=True")
        if not checkpoint_path:
            raise ValueError("completion.checkpoint_path is required when use_completion=True")

        simba_builder = _load_simba_builder_module()
        completion_cfg = cfg_from_yaml_file(config_path)
        completion_model = simba_builder.model_builder(completion_cfg.model)
        simba_builder.load_model(completion_model, checkpoint_path)
        completion_model.to(self.runtime_device)

        if freeze:
            completion_model.eval()
            for parameter in completion_model.parameters():
                parameter.requires_grad_(False)

        self.completion_model = completion_model

    def forward(self, points):
        if not self.enabled or self.completion_model is None:
            return None

        completion_input = points.to(self.runtime_device)
        if self.freeze:
            self.completion_model.eval()
            with torch.no_grad():
                completed = self.completion_model(completion_input)[-1]
        else:
            completed = self.completion_model(completion_input)[-1]

        completed = completed.to(points.device)
        return sample_points_torch(completed, self.output_points)


class CompletionAugmentedVLA(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_action_classes=0,
        action_dim=0,
        hidden_dim=256,
        point_feature_dim=256,
        text_embed_dim=128,
        text_hidden_dim=256,
        dropout=0.1,
        use_completion=False,
        completion_cfg=None,
    ):
        super().__init__()

        if num_action_classes <= 0 and action_dim <= 0:
            raise ValueError("At least one action head must be enabled")

        completion_cfg = completion_cfg or {}
        self.num_action_classes = num_action_classes
        self.action_dim = action_dim
        self.use_completion = use_completion

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
        )
        self.partial_encoder = PointCloudEncoder(
            hidden_dim=point_feature_dim,
            text_hidden_dim=text_hidden_dim,
            dropout=dropout,
        )
        self.completed_encoder = PointCloudEncoder(
            hidden_dim=point_feature_dim,
            text_hidden_dim=text_hidden_dim,
            dropout=dropout,
        ) if use_completion else None
        self.completion = SimbaCompletionWrapper(
            enabled=use_completion,
            config_path=completion_cfg.get("config_path"),
            checkpoint_path=completion_cfg.get("checkpoint_path"),
            device=completion_cfg.get("device", "auto"),
            output_points=completion_cfg.get("output_points", 2048),
            freeze=completion_cfg.get("freeze", True),
        )

        point_embedding_dim = point_feature_dim * 2
        fusion_dim = point_embedding_dim + text_hidden_dim
        if use_completion:
            fusion_dim += point_embedding_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_action_classes) if num_action_classes > 0 else None
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        ) if action_dim > 0 else None

    def forward(self, point_cloud, token_ids, token_mask):
        text_feature = self.text_encoder(token_ids, token_mask)
        partial_feature = self.partial_encoder(point_cloud, text_feature)

        features = [partial_feature, text_feature]
        completed_points = None
        if self.use_completion:
            completed_points = self.completion(point_cloud)
            completed_feature = self.completed_encoder(completed_points, text_feature)
            features = [partial_feature, completed_feature, text_feature]

        fused_feature = self.fusion(torch.cat(features, dim=-1))
        outputs = {
            "embedding": fused_feature,
        }

        if completed_points is not None:
            outputs["completed_points"] = completed_points
        if self.classifier is not None:
            outputs["action_logits"] = self.classifier(fused_feature)
        if self.regressor is not None:
            outputs["action_vector"] = self.regressor(fused_feature)
        return outputs

    def compute_loss(self, outputs, batch, cls_loss_weight=1.0, reg_loss_weight=1.0):
        total_loss = torch.zeros((), device=outputs["embedding"].device)
        losses = {}

        if self.classifier is not None and "action_label" in batch:
            classification_loss = F.cross_entropy(outputs["action_logits"], batch["action_label"])
            losses["classification_loss"] = classification_loss
            total_loss = total_loss + cls_loss_weight * classification_loss

        if self.regressor is not None and "action_vector" in batch:
            regression_loss = F.smooth_l1_loss(outputs["action_vector"], batch["action_vector"])
            losses["regression_loss"] = regression_loss
            total_loss = total_loss + reg_loss_weight * regression_loss

        losses["loss"] = total_loss
        return losses


class CompletionAugmentedDiffusionVLA(nn.Module):
    """Advanced VLA model with PointNet++ encoder and diffusion action head.

    Combines Simba geometric completion with a DP3-style diffusion policy
    for language-conditioned action generation. This is the full-featured
    variant designed for robotic manipulation and autonomous driving.

    Architecture:
        1. SimbaCompletionWrapper: frozen Simba for geometric state recovery
        2. ConditionedPointEncoder (PointNet++): hierarchical 3D feature extraction
           with FiLM conditioning from text embeddings
        3. TextEncoder: bidirectional GRU language encoder
        4. ObservationFusion: multi-modal + temporal fusion
        5. DiffusionActionHead: DDPM conditional action denoiser
        6. Optional DeterministicActionHead + classifier for hybrid training

    Key design choices:
        - Dual-branch encoding (partial + completed) isolates the contribution
          of geometric completion in ablation studies
        - Diffusion action head naturally handles multi-modal action distributions
        - FiLM + cross-attention provides two complementary text conditioning paths
        - Temporal aggregation prepares for multi-step observation windows

    Args:
        vocab_size: tokenizer vocabulary size
        num_action_classes: number of discrete action classes (0 to disable)
        action_dim: continuous action dimension
        hidden_dim: main hidden dimension for fusion
        point_feature_dim: 3D encoder output dimension
        text_embed_dim: text embedding dimension
        text_hidden_dim: text encoder hidden dimension
        dropout: global dropout rate
        use_completion: whether to use Simba for point cloud completion
        completion_cfg: Simba completion configuration dict
        action_head_type: "diffusion" or "deterministic"
        diffusion_cfg: diffusion action head configuration dict
        action_horizon: number of future action steps to predict
        temporal_horizon: number of observation history steps
        proprio_dim: proprioceptive state dimension (0 to disable)
        use_pointnetpp: use PointNet++ encoder instead of simple MLP
    """

    def __init__(
        self,
        vocab_size,
        num_action_classes=0,
        action_dim=7,
        hidden_dim=512,
        point_feature_dim=512,
        text_embed_dim=128,
        text_hidden_dim=256,
        dropout=0.1,
        use_completion=False,
        completion_cfg=None,
        action_head_type="diffusion",
        diffusion_cfg=None,
        action_horizon=1,
        temporal_horizon=1,
        proprio_dim=0,
        use_pointnetpp=True,
    ):
        super().__init__()

        from .encoder import ConditionedPointEncoder, DualBranchEncoder
        from .diffusion_policy import DiffusionActionHead, DeterministicActionHead
        from .temporal import ObservationFusion

        completion_cfg = completion_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        self.num_action_classes = num_action_classes
        self.action_dim = action_dim
        self.use_completion = use_completion
        self.action_head_type = action_head_type
        self.action_horizon = action_horizon

        # --- Text Encoder ---
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
        )

        # --- 3D Point Cloud Encoder ---
        if use_pointnetpp:
            if use_completion:
                self.point_encoder = DualBranchEncoder(
                    feature_dim=point_feature_dim,
                    text_dim=text_hidden_dim,
                    share_backbone=False,
                )
            else:
                self.point_encoder = ConditionedPointEncoder(
                    output_dim=point_feature_dim,
                    text_dim=text_hidden_dim,
                )
        else:
            # Fall back to simple encoder for lightweight mode
            self.point_encoder = PointCloudEncoder(
                hidden_dim=point_feature_dim // 2,
                text_hidden_dim=text_hidden_dim,
                dropout=dropout,
            )
            point_feature_dim = point_feature_dim  # output is 2x hidden from max+mean pool

        # --- Simba Completion ---
        self.completion = SimbaCompletionWrapper(
            enabled=use_completion,
            config_path=completion_cfg.get("config_path"),
            checkpoint_path=completion_cfg.get("checkpoint_path"),
            device=completion_cfg.get("device", "auto"),
            output_points=completion_cfg.get("output_points", 2048),
            freeze=completion_cfg.get("freeze", True),
        )

        # --- Multi-modal Fusion ---
        self.obs_fusion = ObservationFusion(
            visual_dim=point_feature_dim,
            text_dim=text_hidden_dim,
            proprio_dim=proprio_dim,
            output_dim=hidden_dim,
            temporal_horizon=temporal_horizon,
        )

        # --- Action Heads ---
        if action_head_type == "diffusion":
            self.action_head = DiffusionActionHead(
                action_dim=action_dim,
                horizon=action_horizon,
                cond_dim=hidden_dim,
                num_timesteps=diffusion_cfg.get("num_timesteps", 100),
                schedule=diffusion_cfg.get("schedule", "cosine"),
                base_dim=diffusion_cfg.get("base_dim", 128),
                clip_sample=diffusion_cfg.get("clip_sample", True),
            )
        else:
            self.action_head = DeterministicActionHead(
                cond_dim=hidden_dim,
                action_dim=action_dim * action_horizon,
                hidden_dim=hidden_dim,
            )

        # Optional classification head (for action type prediction)
        self.classifier = (
            nn.Linear(hidden_dim, num_action_classes)
            if num_action_classes > 0
            else None
        )

    def encode_observation(self, point_cloud, token_ids, token_mask, proprio_state=None):
        """Encode multi-modal observation into a single conditioning vector.

        Args:
            point_cloud: (B, N, 3) partial point cloud
            token_ids: (B, L) tokenized instruction
            token_mask: (B, L) attention mask
            proprio_state: (B, proprio_dim) optional robot state
        Returns:
            obs_feature: (B, hidden_dim) fused observation
            completed_points: (B, M, 3) or None
        """
        text_feature = self.text_encoder(token_ids, token_mask)

        completed_points = None
        if self.use_completion:
            completed_points = self.completion(point_cloud)
            from .encoder import DualBranchEncoder
            if isinstance(self.point_encoder, DualBranchEncoder):
                visual_feature = self.point_encoder(point_cloud, completed_points, text_feature)
            else:
                visual_feature = self.point_encoder(completed_points, text_feature)
        else:
            if hasattr(self.point_encoder, 'backbone'):
                visual_feature = self.point_encoder(point_cloud, text_feature)
            else:
                visual_feature = self.point_encoder(point_cloud, text_feature)

        obs_feature = self.obs_fusion(visual_feature, text_feature, proprio_state)
        return obs_feature, completed_points

    def forward(self, point_cloud, token_ids, token_mask, proprio_state=None):
        """Forward pass for training and inference.

        Args:
            point_cloud: (B, N, 3)
            token_ids: (B, L)
            token_mask: (B, L)
            proprio_state: (B, proprio_dim) optional
        Returns:
            outputs: dict with keys depending on configuration
        """
        obs_feature, completed_points = self.encode_observation(
            point_cloud, token_ids, token_mask, proprio_state,
        )

        outputs = {"embedding": obs_feature}

        if completed_points is not None:
            outputs["completed_points"] = completed_points

        if self.classifier is not None:
            outputs["action_logits"] = self.classifier(obs_feature)

        # During inference, use sampling; during training, return feature
        if not self.training:
            if self.action_head_type == "diffusion":
                outputs["action_vector"] = self.action_head.sample(obs_feature)
            else:
                outputs["action_vector"] = self.action_head(obs_feature)

        return outputs

    def compute_loss(self, outputs, batch, cls_loss_weight=1.0, reg_loss_weight=1.0):
        """Compute combined training loss.

        For diffusion head: uses denoising score matching loss.
        For deterministic head: uses Smooth L1 regression loss.
        Optional classification loss for discrete action prediction.
        """
        total_loss = torch.zeros((), device=outputs["embedding"].device)
        losses = {}

        # Classification loss
        if self.classifier is not None and "action_label" in batch:
            cls_loss = F.cross_entropy(outputs["action_logits"], batch["action_label"])
            losses["classification_loss"] = cls_loss
            total_loss = total_loss + cls_loss_weight * cls_loss

        # Action generation loss
        if "action_vector" in batch:
            obs_feature = outputs["embedding"]
            actions = batch["action_vector"]

            if self.action_head_type == "diffusion":
                action_loss = self.action_head.compute_loss(actions, obs_feature)
            else:
                action_loss = self.action_head.compute_loss(actions, obs_feature)

            losses["action_loss"] = action_loss
            total_loss = total_loss + reg_loss_weight * action_loss

        losses["loss"] = total_loss
        return losses


def build_vla_model(model_cfg, vocab_size, num_action_classes, action_dim):
    """Build VLA model from config.

    Automatically selects between the simple CompletionAugmentedVLA and
    the advanced CompletionAugmentedDiffusionVLA based on config fields.
    """
    # Use advanced model if diffusion or pointnetpp is requested
    if model_cfg.get("action_head_type") or model_cfg.get("use_pointnetpp"):
        return CompletionAugmentedDiffusionVLA(
            vocab_size=vocab_size,
            num_action_classes=num_action_classes,
            action_dim=action_dim,
            hidden_dim=model_cfg.get("hidden_dim", 512),
            point_feature_dim=model_cfg.get("point_feature_dim", 512),
            text_embed_dim=model_cfg.get("text_embed_dim", 128),
            text_hidden_dim=model_cfg.get("text_hidden_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
            use_completion=model_cfg.get("use_completion", False),
            completion_cfg=model_cfg.get("completion", {}),
            action_head_type=model_cfg.get("action_head_type", "diffusion"),
            diffusion_cfg=model_cfg.get("diffusion", {}),
            action_horizon=model_cfg.get("action_horizon", 1),
            temporal_horizon=model_cfg.get("temporal_horizon", 1),
            proprio_dim=model_cfg.get("proprio_dim", 0),
            use_pointnetpp=model_cfg.get("use_pointnetpp", True),
        )

    # Default: simple model for backward compatibility
    return CompletionAugmentedVLA(
        vocab_size=vocab_size,
        num_action_classes=num_action_classes,
        action_dim=action_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        point_feature_dim=model_cfg.get("point_feature_dim", 256),
        text_embed_dim=model_cfg.get("text_embed_dim", 128),
        text_hidden_dim=model_cfg.get("text_hidden_dim", 256),
        dropout=model_cfg.get("dropout", 0.1),
        use_completion=model_cfg.get("use_completion", False),
        completion_cfg=model_cfg.get("completion", {}),
    )
