"""
Evaluation script for Completion-Augmented VLA.

Supports:
    - Standard accuracy / MAE evaluation
    - Occlusion sweep: evaluate under increasing occlusion severity
    - Ablation modes: partial-only / completed-only / partial+completed
    - Per-scenario breakdown (manipulation / driving)
    - Waypoint evaluation metrics (ADE, FDE) for driving
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.config import cfg_from_yaml_file
from vla.dataset import PointCloudVLADataset, load_annotation_file
from vla.model import build_vla_model, resolve_device
from vla.occlusion import OcclusionSimulator, create_occlusion_sweep
from vla.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a completion-augmented VLA model")
    parser.add_argument("--config", type=str, required=True, help="Path to VLA config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--device", type=str, default="auto")

    # Evaluation modes
    parser.add_argument("--occlusion_sweep", action="store_true",
                        help="Run evaluation under varying occlusion levels")
    parser.add_argument("--occlusion_method", type=str, default="random",
                        choices=["viewpoint", "planar", "random", "distance", "sector"])
    parser.add_argument("--severities", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # Ablation
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "partial_only", "completion_only"],
                        help="Which input branches to use")

    parser.add_argument("--output_json", type=str, default="", help="Save results to JSON")
    parser.add_argument("--per_scenario", action="store_true", help="Report per-scenario metrics")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def compute_waypoint_metrics(predicted, ground_truth):
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE).

    These are standard metrics for trajectory/waypoint prediction in
    autonomous driving evaluation.

    Args:
        predicted: (B, T, 3) predicted waypoints
        ground_truth: (B, T, 3) ground truth waypoints
    Returns:
        ade: average L2 displacement across all timesteps
        fde: L2 displacement at the final timestep
    """
    displacement = torch.norm(predicted - ground_truth, dim=-1)  # (B, T)
    ade = displacement.mean().item()
    fde = displacement[:, -1].mean().item()
    return ade, fde


@torch.no_grad()
def evaluate(model, dataloader, device, scenario_filter=None):
    """Run evaluation and collect metrics.

    Args:
        model: VLA model in eval mode
        dataloader: evaluation dataloader
        device: compute device
        scenario_filter: optional string to only evaluate specific scenario
    Returns:
        metrics: dict of evaluation metrics
    """
    model.eval()

    total_samples = 0
    correct_cls = 0
    total_cls = 0
    total_reg_mae = 0.0
    total_reg = 0
    total_ade = 0.0
    total_fde = 0.0
    total_waypoints = 0

    scenario_metrics = {}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        batch = move_batch_to_device(batch, device)
        B = batch["point_cloud"].shape[0]

        # Filter by scenario if requested
        if scenario_filter and "scenarios" in batch:
            mask = [s == scenario_filter for s in batch["scenarios"]]
            if not any(mask):
                continue

        outputs = model(batch["point_cloud"], batch["token_ids"], batch["token_mask"])
        total_samples += B

        # Classification accuracy
        if "action_logits" in outputs and "action_label" in batch:
            preds = outputs["action_logits"].argmax(dim=-1)
            correct_cls += (preds == batch["action_label"]).sum().item()
            total_cls += B

        # Regression MAE
        if "action_vector" in outputs and "action_vector" in batch:
            mae = torch.abs(outputs["action_vector"] - batch["action_vector"]).mean(dim=-1)
            total_reg_mae += mae.sum().item()
            total_reg += B

        # Waypoint metrics (for driving)
        if "waypoints" in batch and "action_vector" in outputs:
            pred_wp = outputs["action_vector"]
            gt_wp = batch["waypoints"]
            if pred_wp.shape == gt_wp.shape:
                ade, fde = compute_waypoint_metrics(pred_wp, gt_wp)
                total_ade += ade * B
                total_fde += fde * B
                total_waypoints += B

        # Per-scenario tracking
        if "scenarios" in batch:
            for i, scenario in enumerate(batch["scenarios"]):
                if scenario not in scenario_metrics:
                    scenario_metrics[scenario] = {"correct": 0, "total": 0, "mae_sum": 0.0}
                scenario_metrics[scenario]["total"] += 1
                if "action_logits" in outputs and "action_label" in batch:
                    if outputs["action_logits"][i].argmax() == batch["action_label"][i]:
                        scenario_metrics[scenario]["correct"] += 1
                if "action_vector" in outputs and "action_vector" in batch:
                    scenario_metrics[scenario]["mae_sum"] += (
                        torch.abs(outputs["action_vector"][i] - batch["action_vector"][i]).mean().item()
                    )

    metrics = {"total_samples": total_samples}
    if total_cls > 0:
        metrics["accuracy"] = correct_cls / total_cls
    if total_reg > 0:
        metrics["action_mae"] = total_reg_mae / total_reg
    if total_waypoints > 0:
        metrics["waypoint_ade"] = total_ade / total_waypoints
        metrics["waypoint_fde"] = total_fde / total_waypoints

    if scenario_metrics:
        metrics["per_scenario"] = {}
        for scenario, sm in scenario_metrics.items():
            metrics["per_scenario"][scenario] = {
                "accuracy": sm["correct"] / max(sm["total"], 1),
                "mae": sm["mae_sum"] / max(sm["total"], 1),
                "count": sm["total"],
            }

    return metrics


def run_occlusion_sweep(model, config, tokenizer, device, method, severities, label_to_index):
    """Evaluate model under increasing occlusion levels.

    This is the key experiment demonstrating that Simba completion
    improves action prediction robustness as occlusion increases.
    """
    results = []

    for severity in severities:
        print(f"\n  Occlusion severity: {severity:.1f}")
        occ_sim = OcclusionSimulator(
            occlusion_types=[method],
            severity=severity,
            random_severity=False,
        ) if severity > 0 else None

        dataset = PointCloudVLADataset(
            annotation_path=config.dataset.val_annotations,
            tokenizer=tokenizer,
            num_points=config.dataset.get("num_points", 2048),
            max_text_length=config.dataset.get("max_text_length", 32),
            normalize_points=config.dataset.get("normalize_points", True),
            label_to_index=label_to_index,
            random_sample=False,
            occlusion_simulator=occ_sim,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.train.get("eval_batch_size", 8),
            shuffle=False,
            collate_fn=PointCloudVLADataset.collate_fn,
        )

        metrics = evaluate(model, loader, device)
        metrics["severity"] = severity
        metrics["method"] = method
        results.append(metrics)

        acc = metrics.get("accuracy", float("nan"))
        mae = metrics.get("action_mae", float("nan"))
        print(f"    accuracy={acc:.4f}  mae={mae:.4f}")

    return results


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    tokenizer = SimpleTokenizer.from_dict(checkpoint["tokenizer"])
    label_to_index = checkpoint.get("label_to_index", {})
    index_to_label = {v: k for k, v in label_to_index.items()}

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

    all_results = {}

    if args.occlusion_sweep:
        print("=" * 60)
        print("Occlusion Robustness Sweep")
        print(f"Method: {args.occlusion_method}")
        print("=" * 60)
        sweep_results = run_occlusion_sweep(
            model, config, tokenizer, device,
            method=args.occlusion_method,
            severities=args.severities,
            label_to_index=label_to_index,
        )
        all_results["occlusion_sweep"] = sweep_results
    else:
        print("=" * 60)
        print("Standard Evaluation")
        print("=" * 60)

        dataset = PointCloudVLADataset(
            annotation_path=config.dataset.val_annotations,
            tokenizer=tokenizer,
            num_points=config.dataset.get("num_points", 2048),
            max_text_length=config.dataset.get("max_text_length", 32),
            normalize_points=config.dataset.get("normalize_points", True),
            label_to_index=label_to_index,
            random_sample=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.train.get("eval_batch_size", 8),
            shuffle=False,
            collate_fn=PointCloudVLADataset.collate_fn,
        )
        metrics = evaluate(model, loader, device)
        all_results["standard"] = metrics

        print(f"\nResults:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif k != "per_scenario":
                print(f"  {k}: {v}")

        if args.per_scenario and "per_scenario" in metrics:
            print("\nPer-scenario breakdown:")
            for scenario, sm in metrics["per_scenario"].items():
                print(f"  {scenario}: acc={sm['accuracy']:.4f}  mae={sm['mae']:.4f}  n={sm['count']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
