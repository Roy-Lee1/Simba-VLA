import argparse
import os
import random
import shutil
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
from vla.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a completion-augmented point-cloud VLA model")
    parser.add_argument("--config", type=str, required=True, help="Path to the VLA config file")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def build_tokenizer(dataset_cfg, output_dir):
    vocab_path = dataset_cfg.get("vocab_path")
    if vocab_path and os.path.exists(vocab_path):
        return SimpleTokenizer.load(vocab_path), vocab_path

    train_samples = load_annotation_file(dataset_cfg.train_annotations)
    tokenizer = SimpleTokenizer.build(
        texts=[sample["instruction"] for sample in train_samples],
        min_freq=dataset_cfg.get("min_token_frequency", 1),
        max_vocab_size=dataset_cfg.get("max_vocab_size"),
    )

    vocab_path = vocab_path or os.path.join(output_dir, "vocab.json")
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    tokenizer.save(vocab_path)
    return tokenizer, vocab_path


def build_datasets(config, tokenizer):
    dataset_cfg = config.dataset
    train_dataset = PointCloudVLADataset(
        annotation_path=dataset_cfg.train_annotations,
        tokenizer=tokenizer,
        num_points=dataset_cfg.get("num_points", 2048),
        max_text_length=dataset_cfg.get("max_text_length", 32),
        point_cloud_root=dataset_cfg.get("point_cloud_root"),
        normalize_points=dataset_cfg.get("normalize_points", True),
        random_sample=True,
    )
    val_dataset = PointCloudVLADataset(
        annotation_path=dataset_cfg.val_annotations,
        tokenizer=tokenizer,
        num_points=dataset_cfg.get("num_points", 2048),
        max_text_length=dataset_cfg.get("max_text_length", 32),
        point_cloud_root=dataset_cfg.get("point_cloud_root"),
        normalize_points=dataset_cfg.get("normalize_points", True),
        label_to_index=train_dataset.label_to_index,
        random_sample=False,
    )
    return train_dataset, val_dataset


def build_dataloaders(config, train_dataset, val_dataset):
    train_cfg = config.train
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=PointCloudVLADataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=PointCloudVLADataset.collate_fn,
    )
    return train_loader, val_loader


def run_epoch(model, dataloader, device, optimizer=None, cls_loss_weight=1.0, reg_loss_weight=1.0, grad_clip=0.0):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    correct = 0
    total_labels = 0
    total_vector_mae = 0.0
    total_vectors = 0

    progress = tqdm(dataloader, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        batch_size = batch["point_cloud"].shape[0]

        with torch.set_grad_enabled(training):
            outputs = model(batch["point_cloud"], batch["token_ids"], batch["token_mask"])
            loss_dict = model.compute_loss(
                outputs,
                batch,
                cls_loss_weight=cls_loss_weight,
                reg_loss_weight=reg_loss_weight,
            )
            loss = loss_dict["loss"]

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if "action_logits" in outputs and "action_label" in batch:
            predictions = outputs["action_logits"].argmax(dim=-1)
            correct += (predictions == batch["action_label"]).sum().item()
            total_labels += batch["action_label"].numel()

        if "action_vector" in outputs and "action_vector" in batch:
            mae = torch.abs(outputs["action_vector"] - batch["action_vector"]).mean(dim=-1)
            total_vector_mae += mae.sum().item()
            total_vectors += mae.numel()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    metrics = {
        "loss": total_loss / max(total_samples, 1),
    }
    if total_labels > 0:
        metrics["accuracy"] = correct / total_labels
    if total_vectors > 0:
        metrics["vector_mae"] = total_vector_mae / total_vectors
    return metrics


def score_metrics(metrics):
    score = 0.0
    if "accuracy" in metrics:
        score += metrics["accuracy"]
    if "vector_mae" in metrics:
        score -= metrics["vector_mae"]
    return score


def save_checkpoint(path, model, optimizer, epoch, metrics, tokenizer, label_to_index):
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "label_to_index": label_to_index,
            "num_action_classes": model.num_action_classes,
            "action_dim": model.action_dim,
        },
        path,
    )


def main():
    args = parse_args()
    config = cfg_from_yaml_file(args.config)

    train_cfg = config.train
    output_dir = train_cfg.get("output_dir", os.path.join("experiments", "simba_vla"))
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(output_dir, "config.yaml"))

    set_random_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device", "auto"))

    tokenizer, vocab_path = build_tokenizer(config.dataset, output_dir)
    train_dataset, val_dataset = build_datasets(config, tokenizer)
    train_loader, val_loader = build_dataloaders(config, train_dataset, val_dataset)

    model = build_vla_model(
        config.model,
        vocab_size=tokenizer.vocab_size,
        num_action_classes=len(train_dataset.label_to_index),
        action_dim=train_dataset.action_dim,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("epochs", 50),
        eta_min=train_cfg.get("min_lr", 1e-5),
    )

    best_score = float("-inf")
    epochs = train_cfg.get("epochs", 50)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            cls_loss_weight=train_cfg.get("cls_loss_weight", 1.0),
            reg_loss_weight=train_cfg.get("reg_loss_weight", 1.0),
            grad_clip=train_cfg.get("grad_clip", 0.0),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            cls_loss_weight=train_cfg.get("cls_loss_weight", 1.0),
            reg_loss_weight=train_cfg.get("reg_loss_weight", 1.0),
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics.get('accuracy', float('nan')):.4f} | "
            f"val_mae={val_metrics.get('vector_mae', float('nan')):.4f}"
        )

        checkpoint_metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "vocab_path": vocab_path,
        }
        save_checkpoint(
            os.path.join(output_dir, "last_model.pth"),
            model,
            optimizer,
            epoch,
            checkpoint_metrics,
            tokenizer,
            train_dataset.label_to_index,
        )

        current_score = score_metrics(val_metrics)
        if current_score > best_score:
            best_score = current_score
            save_checkpoint(
                os.path.join(output_dir, "best_model.pth"),
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                tokenizer,
                train_dataset.label_to_index,
            )


if __name__ == "__main__":
    main()
