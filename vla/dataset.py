import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.io import IO


def load_annotation_file(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    else:
        raise ValueError(f"Unsupported annotation format in {annotation_path}")

    if not samples:
        raise ValueError(f"No samples found in {annotation_path}")

    return samples


def sample_point_cloud(points, num_points, random_sample=True):
    if num_points is None or num_points <= 0:
        return points.astype(np.float32)

    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")

    points = points.astype(np.float32)
    total_points = points.shape[0]

    if total_points >= num_points:
        if random_sample:
            indices = np.random.choice(total_points, size=num_points, replace=False)
        else:
            indices = np.linspace(0, total_points - 1, num_points, dtype=np.int64)
        return points[indices]

    repeats = num_points // total_points
    remainder = num_points % total_points
    expanded = np.tile(points, (repeats, 1)) if repeats > 0 else np.empty((0, points.shape[1]), dtype=np.float32)

    if remainder > 0:
        if random_sample:
            extra_indices = np.random.choice(total_points, size=remainder, replace=total_points < remainder)
        else:
            extra_indices = np.linspace(0, total_points - 1, remainder, dtype=np.int64)
        expanded = np.concatenate([expanded, points[extra_indices]], axis=0)

    return expanded.astype(np.float32)


def normalize_point_cloud(points):
    centroid = points.mean(axis=0, keepdims=True)
    normalized = points - centroid
    scale = np.linalg.norm(normalized, axis=1).max()
    scale = float(scale) if scale > 1e-6 else 1.0
    return normalized / scale, centroid.squeeze(0).astype(np.float32), scale


class PointCloudVLADataset(Dataset):
    def __init__(
        self,
        annotation_path,
        tokenizer,
        num_points=2048,
        max_text_length=32,
        point_cloud_root=None,
        normalize_points=True,
        label_to_index=None,
        random_sample=True,
        occlusion_simulator=None,
    ):
        self.annotation_path = annotation_path
        self.samples = load_annotation_file(annotation_path)
        self.tokenizer = tokenizer
        self.num_points = num_points
        self.max_text_length = max_text_length
        self.point_cloud_root = point_cloud_root or os.path.dirname(os.path.abspath(annotation_path))
        self.normalize_points = normalize_points
        self.random_sample = random_sample
        self.occlusion_simulator = occlusion_simulator

        self.label_to_index = self._build_label_mapping(self.samples, label_to_index)
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}
        self.action_dim = self._infer_action_dim(self.samples)
        self.has_action_label = bool(self.label_to_index)
        self.has_action_vector = self.action_dim > 0
        self.has_waypoints = self._check_waypoints(self.samples)
        self.has_proprio = self._check_proprio(self.samples)
        self._validate_samples()

    @staticmethod
    def _build_label_mapping(samples, label_to_index=None):
        if label_to_index is not None:
            return dict(label_to_index)

        labels = sorted({sample["action_label"] for sample in samples if "action_label" in sample})
        return {label: index for index, label in enumerate(labels)}

    @staticmethod
    def _infer_action_dim(samples):
        action_dims = {len(sample["action_vector"]) for sample in samples if "action_vector" in sample}
        if not action_dims:
            return 0
        if len(action_dims) != 1:
            raise ValueError("All action vectors must share the same dimensionality")
        return action_dims.pop()

    @staticmethod
    def _check_waypoints(samples):
        return any("waypoints" in s for s in samples)

    @staticmethod
    def _check_proprio(samples):
        return any("proprio_state" in s for s in samples)

    def _validate_samples(self):
        for sample in self.samples:
            if "point_cloud" not in sample:
                raise KeyError("Each sample must contain a point_cloud field")
            if "instruction" not in sample:
                raise KeyError("Each sample must contain an instruction field")
            if self.has_action_label and "action_label" not in sample:
                raise KeyError("All samples must contain action_label when classification is enabled")
            if self.has_action_vector and "action_vector" not in sample:
                raise KeyError("All samples must contain action_vector when regression is enabled")

    def __len__(self):
        return len(self.samples)

    def _resolve_point_cloud_path(self, point_cloud_path):
        if os.path.isabs(point_cloud_path):
            return point_cloud_path
        return os.path.join(self.point_cloud_root, point_cloud_path)

    def __getitem__(self, index):
        sample = self.samples[index]
        point_cloud_path = self._resolve_point_cloud_path(sample["point_cloud"])

        points = IO.get(point_cloud_path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Point cloud must have shape [N, 3+] but got {points.shape} for {point_cloud_path}")

        points = points[:, :3]

        # Apply occlusion augmentation if configured
        occlusion_info = {}
        if self.occlusion_simulator is not None:
            points, occlusion_info = self.occlusion_simulator(points)

        points = sample_point_cloud(points, self.num_points, random_sample=self.random_sample)

        if self.normalize_points:
            points, center, scale = normalize_point_cloud(points)
        else:
            center = np.zeros(3, dtype=np.float32)
            scale = 1.0

        token_ids, token_mask = self.tokenizer.encode(sample["instruction"], max_length=self.max_text_length)

        item = {
            "id": sample.get("id", str(index)),
            "instruction": sample["instruction"],
            "point_cloud_path": point_cloud_path,
            "point_cloud": torch.from_numpy(points).float(),
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "token_mask": torch.tensor(token_mask, dtype=torch.bool),
            "normalization_center": torch.from_numpy(center).float(),
            "normalization_scale": torch.tensor(scale, dtype=torch.float32),
            "metadata": sample.get("metadata", {}),
        }

        if self.has_action_label:
            item["action_label"] = torch.tensor(self.label_to_index[sample["action_label"]], dtype=torch.long)

        if self.has_action_vector:
            item["action_vector"] = torch.tensor(sample["action_vector"], dtype=torch.float32)

        # Waypoint trajectories for driving scenarios
        if "waypoints" in sample:
            item["waypoints"] = torch.tensor(sample["waypoints"], dtype=torch.float32)

        # Proprioceptive state for manipulation scenarios
        if "proprio_state" in sample:
            item["proprio_state"] = torch.tensor(sample["proprio_state"], dtype=torch.float32)

        # Scenario tag for multi-domain evaluation
        if "scenario" in sample:
            item["scenario"] = sample["scenario"]

        if occlusion_info:
            item["occlusion_info"] = occlusion_info

        return item

    @staticmethod
    def collate_fn(batch):
        collated = {
            "ids": [item["id"] for item in batch],
            "instructions": [item["instruction"] for item in batch],
            "point_cloud_paths": [item["point_cloud_path"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
            "point_cloud": torch.stack([item["point_cloud"] for item in batch], dim=0),
            "token_ids": torch.stack([item["token_ids"] for item in batch], dim=0),
            "token_mask": torch.stack([item["token_mask"] for item in batch], dim=0),
            "normalization_center": torch.stack([item["normalization_center"] for item in batch], dim=0),
            "normalization_scale": torch.stack([item["normalization_scale"] for item in batch], dim=0),
        }

        if "action_label" in batch[0]:
            collated["action_label"] = torch.stack([item["action_label"] for item in batch], dim=0)

        if "action_vector" in batch[0]:
            collated["action_vector"] = torch.stack([item["action_vector"] for item in batch], dim=0)

        if "waypoints" in batch[0]:
            collated["waypoints"] = torch.stack([item["waypoints"] for item in batch], dim=0)

        if "proprio_state" in batch[0]:
            collated["proprio_state"] = torch.stack([item["proprio_state"] for item in batch], dim=0)

        if "scenario" in batch[0]:
            collated["scenarios"] = [item["scenario"] for item in batch]

        return collated
