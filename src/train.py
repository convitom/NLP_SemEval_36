"""
src/train.py
Training script for SemEval 2018 Task 1 Multi-Label Emotion Classification.

Features:
  - Fine-tunes any of 4 encoder-only backbones: bert | roberta | deberta | electra
  - Multi-label: bce | bce_weighted | focal_bce | asymmetric loss
  - WeightedRandomSampler for label imbalance
  - AMP (Automatic Mixed Precision) on CUDA
  - Cosine LR schedule with linear warmup
  - Early stopping on val_loss
  - Saves best.pth checkpoint + training_log.csv

CLI:
  python src/train.py
  python src/train.py --config config/config.yaml

Notebook:
  from src.train import train
  result = train(config_path="config/config.yaml")
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import AutoModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders, NUM_EMOTIONS, EMOTION_NAMES, BACKBONE_REGISTRY
from src.utils import (
    AverageMeter, get_optimizer, get_result_dir,
    get_scheduler, load_config, set_seed, apply_threshold,
)
from models.loss import get_loss_fn


###############################################################################
#  Model
###############################################################################

class EncoderForMultiLabelClassification(nn.Module):
    """
    Generic encoder-only backbone với 11 binary classifier heads độc lập.

    Architecture:
        backbone  ->  [CLS] hidden state  ->  Dropout
                  ->  [head_0(hidden,1), head_1(hidden,1), ..., head_10(hidden,1)]
                  ->  concat  ->  (B, 11) logits

    Mỗi head là một binary classifier riêng biệt cho một emotion.
    Dùng sigmoid + per-class threshold khi inference, không dùng chung 1 threshold.

    Args:
        pretrained_name: HuggingFace model ID hoặc local path.
        num_labels:      Số emotion classes (11).
        dropout:         Dropout trước mỗi classifier head.
    """

    def __init__(
        self,
        pretrained_name: str,
        num_labels: int = NUM_EMOTIONS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone     = AutoModel.from_pretrained(pretrained_name)
        self.dropout      = nn.Dropout(dropout)
        hidden_size       = self.backbone.config.hidden_size
        # 11 independent binary heads — each outputs 1 logit
        self.classifiers  = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_labels)
        ])
        self.num_labels = num_labels

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)

        Returns:
            logits: (B, num_labels)  — raw scores, one per emotion head.
        """
        outputs    = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]          # (B, hidden)
        cls_output = self.dropout(cls_output)

        # Each head: (B, hidden) -> (B, 1); concat along dim=1 -> (B, 11)
        logits = torch.cat(
            [head(cls_output) for head in self.classifiers], dim=1
        )
        return logits


def build_model(cfg: dict) -> EncoderForMultiLabelClassification:
    """
    Instantiate model từ config.
    Mỗi emotion class có 1 binary classifier head riêng (ModuleList).
    Supported: bert | roberta | deberta | electra
    """
    name = cfg["model"]["name"].lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. "
            f"Choose from: {' | '.join(BACKBONE_REGISTRY.keys())}"
        )
    pretrained_name = BACKBONE_REGISTRY[name]["pretrained"]
    return EncoderForMultiLabelClassification(
        pretrained_name=pretrained_name,
        num_labels=int(cfg["data"]["num_emotions"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )


###############################################################################
#  One-epoch helpers
###############################################################################

def _run_epoch(
    model:         nn.Module,
    loader,
    criterion:     nn.Module,
    optimizer:     Optional[torch.optim.Optimizer],
    scheduler,
    scaler:        Optional[GradScaler],
    device:        torch.device,
    phase:         str,
    epoch:         int,
    total_epochs:  int,
    threshold:     float = 0.5,
    use_amp:       bool = False,
    amp_dtype:     torch.dtype = torch.float16,
) -> Tuple[float, float, float, float]:
    """
    Run one forward (+ optional backward) pass.

    Returns:
        (avg_loss, micro_f1, macro_f1, weighted_f1)
        F1 scores are 0.0 for train phase (not computed to save time).
    """
    # Resolve amp_dtype from device if not explicitly specified by caller
    if use_amp and amp_dtype is None:
        amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter("loss")
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    desc = f"Epoch {epoch}/{total_epochs} [{phase:>5}]"
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, unit="batch")

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            input_ids      = batch["input_ids"].to(device,      non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device,         non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            _dtype = amp_dtype if (use_amp and amp_dtype is not None) else torch.float16
            with torch.autocast(device_type=device.type, dtype=_dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)

            if not is_train:
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = apply_threshold(probs, threshold)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    pbar.close()

    # Compute F1 on val
    micro_f1 = macro_f1 = weighted_f1 = 0.0
    if not is_train and all_preds:
        y_true = np.vstack(all_labels)
        y_pred = np.vstack(all_preds)
        micro_f1    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return loss_meter.avg, micro_f1, macro_f1, weighted_f1


###############################################################################
#  Main training function
###############################################################################

def train(config_path: str = "config/config.yaml") -> Dict:
    """
    Full training pipeline.

    Returns:
        dict with keys:
            best_val_loss, best_val_micro_f1, best_epoch,
            log_path, checkpoint_path.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    train_cfg = cfg["training"]
    epochs    = int(train_cfg["epochs"])
    patience  = int(train_cfg.get("early_stopping_patience", 3))
    threshold = float(train_cfg.get("threshold", 0.5))

    # ── Device & AMP ────────────────────────────────────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_key  = cfg["model"]["name"].lower()
    amp_dtype_str = BACKBONE_REGISTRY.get(model_key, {}).get("amp_dtype", "float16")

    # DeBERTa-v3 has fp16 internal params → GradScaler's unscale raises ValueError.
    # Fix: use bfloat16 autocast (wider range, no overflow scaling needed → no GradScaler).
    # All other models use standard float16 + GradScaler.
    if device.type == "cuda" and amp_dtype_str == "bfloat16":
        use_amp    = True
        amp_dtype  = torch.bfloat16
        use_scaler = False          # bfloat16 doesn't need loss scaling
    elif device.type == "cuda":
        use_amp    = True
        amp_dtype  = torch.float16
        use_scaler = True
    else:
        use_amp    = False
        amp_dtype  = torch.float32
        use_scaler = False

    print(f"[train] Device: {device} | AMP: {use_amp} ({amp_dtype_str})")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, info = get_dataloaders(cfg)
    pos_weight  = info["pos_weight"]
    num_labels  = len(info["emotion_names"])
    label_names = info["emotion_names"]

    # ── Model ───────────────────────────────────────────────────────────────
    # Override num_labels in cfg so build_model picks up the right head size
    cfg["data"]["num_emotions"] = num_labels
    model      = build_model(cfg).to(device)
    model_name = cfg["model"]["name"]
    pretrained_name = BACKBONE_REGISTRY[model_name.lower()]["pretrained"]
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model     : {model_name}  ({pretrained_name})")
    print(f"[train] Num labels: {num_labels}  {label_names}")
    print(f"[train] Params    : {n_params:,}")

    # ── Loss / Optimizer / Scheduler ────────────────────────────────────────
    criterion   = get_loss_fn(cfg, device, pos_weight=pos_weight)
    optimizer   = get_optimizer(model, cfg)
    total_steps = len(train_loader) * epochs
    scheduler   = get_scheduler(optimizer, cfg, num_training_steps=total_steps)
    scaler      = GradScaler() if use_scaler else None

    # ── Output dirs ─────────────────────────────────────────────────────────
    result_dir = get_result_dir(cfg)
    ckpt_dir   = os.path.join("checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path  = os.path.join(result_dir, "training_log.csv")
    ckpt_path = os.path.join(ckpt_dir,   "best.pth")

    # ── CSV header ───────────────────────────────────────────────────────────
    csv_fields = [
        "epoch", "train_loss",
        "val_loss", "val_micro_f1", "val_macro_f1", "val_weighted_f1",
        "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss    = float("inf")
    best_val_micro_f1 = 0.0
    best_epoch       = 0
    no_improve       = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, _, _, _ = _run_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, "train", epoch, epochs, threshold,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )

        val_loss, micro_f1, macro_f1, weighted_f1 = _run_epoch(
            model, val_loader, criterion, None, None,
            None, device, "val", epoch, epochs, threshold,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}  "
            f"weighted_f1={weighted_f1:.4f}  "
            f"lr={current_lr:.2e}  [{elapsed:.0f}s]"
        )

        # ── CSV append ───────────────────────────────────────────────────────
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow({
                "epoch":           epoch,
                "train_loss":      round(train_loss,   6),
                "val_loss":        round(val_loss,     6),
                "val_micro_f1":    round(micro_f1,     4),
                "val_macro_f1":    round(macro_f1,     4),
                "val_weighted_f1": round(weighted_f1,  4),
                "lr":              current_lr,
            })

        # ── Checkpoint ───────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_val_micro_f1 = micro_f1
            best_epoch        = epoch
            no_improve        = 0
            torch.save({
                "epoch":           epoch,
                "model_name":      model_name,
                "pretrained_name": pretrained_name,
                "num_labels":      num_labels,
                "label_names":     label_names,
                "model_state":     model.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "scheduler":       scheduler.state_dict() if scheduler else None,
                "val_loss":        best_val_loss,
                "val_micro_f1":    best_val_micro_f1,
                "threshold":       threshold,
                "cfg":             cfg,
            }, ckpt_path)
            print(f"  ✓ Checkpoint saved (epoch {epoch})")
        else:
            no_improve += 1

        # ── Early stopping ───────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"[train] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    print(f"\n[train] Best  val_loss={best_val_loss:.4f}  "
          f"micro_f1={best_val_micro_f1:.4f}  @ epoch {best_epoch}")
    print(f"[train] Checkpoint -> {ckpt_path}")
    print(f"[train] Log        -> {log_path}")

    return {
        "best_val_loss":     best_val_loss,
        "best_val_micro_f1": best_val_micro_f1,
        "best_epoch":        best_epoch,
        "log_path":          log_path,
        "checkpoint_path":   ckpt_path,
    }


###############################################################################
#  CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GoEmotions emotion classifier.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    train(config_path=args.config)
