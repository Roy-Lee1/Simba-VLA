import argparse
import json
import os
import sys

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from datasets.io import IO
from utils.config import cfg_from_yaml_file
from vla.dataset import normalize_point_cloud, sample_point_cloud
from vla.model import build_vla_model, resolve_device
from vla.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a completion-augmented point-cloud VLA model")
    parser.add_argument("--config", type=str, required=True, help="Path to the VLA config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained VLA checkpoint")
    parser.add_argument("--point_cloud", type=str, required=True, help="Path to the input point cloud")
    parser.add_argument("--instruction", type=str, required=True, help="Language instruction paired with the point cloud")
    parser.add_argument("--device", type=str, default="auto", help="Runtime device")
    parser.add_argument("--output_json", type=str, default="", help="Optional path to save inference output as JSON")
    parser.add_argument("--save_completed_points", type=str, default="", help="Optional path to save completed point cloud as NPY")
    return parser.parse_args()


def prepare_inputs(point_cloud_path, instruction, tokenizer, dataset_cfg):
    points = IO.get(point_cloud_path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must have shape [N, 3+] but got {points.shape}")

    points = points[:, :3]
    points = sample_point_cloud(points, dataset_cfg.get("num_points", 2048), random_sample=False)

    if dataset_cfg.get("normalize_points", True):
        normalized_points, center, scale = normalize_point_cloud(points)
    else:
        normalized_points = points
        center = np.zeros(3, dtype=np.float32)
        scale = 1.0

    token_ids, token_mask = tokenizer.encode(instruction, max_length=dataset_cfg.get("max_text_length", 32))
    batch = {
        "point_cloud": torch.tensor(normalized_points, dtype=torch.float32).unsqueeze(0),
        "token_ids": torch.tensor(token_ids, dtype=torch.long).unsqueeze(0),
        "token_mask": torch.tensor(token_mask, dtype=torch.bool).unsqueeze(0),
        "normalization_center": torch.tensor(center, dtype=torch.float32).unsqueeze(0),
        "normalization_scale": torch.tensor([scale], dtype=torch.float32),
    }
    return batch


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def denormalize_completed_points(completed_points, batch):
    center = batch["normalization_center"][:, None, :]
    scale = batch["normalization_scale"][:, None, None]
    return completed_points * scale + center


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    tokenizer = SimpleTokenizer.from_dict(checkpoint["tokenizer"])
    label_to_index = checkpoint.get("label_to_index", {})
    index_to_label = {index: label for label, index in label_to_index.items()}

    model = build_vla_model(
        config.model,
        vocab_size=tokenizer.vocab_size,
        num_action_classes=checkpoint.get("num_action_classes", len(label_to_index)),
        action_dim=checkpoint.get("action_dim", 0),
    )
    model.load_state_dict(checkpoint["model_state"])

    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    batch = prepare_inputs(args.point_cloud, args.instruction, tokenizer, config.dataset)
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        outputs = model(batch["point_cloud"], batch["token_ids"], batch["token_mask"])

    result = {
        "point_cloud": args.point_cloud,
        "instruction": args.instruction,
    }

    if "action_logits" in outputs:
        probabilities = torch.softmax(outputs["action_logits"], dim=-1)[0].cpu().tolist()
        predicted_index = int(np.argmax(probabilities))
        result["predicted_action_label"] = index_to_label.get(predicted_index, str(predicted_index))
        result["action_distribution"] = {
            index_to_label.get(index, str(index)): float(score)
            for index, score in enumerate(probabilities)
        }

    if "action_vector" in outputs:
        result["predicted_action_vector"] = outputs["action_vector"][0].cpu().tolist()

    if "completed_points" in outputs and args.save_completed_points:
        completed_points = denormalize_completed_points(outputs["completed_points"], batch)
        np.save(args.save_completed_points, completed_points[0].cpu().numpy())
        result["completed_points"] = args.save_completed_points

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
