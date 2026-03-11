"""
src/test.py
Evaluation script for SemEval 2018 Task 1 Multi-Label Emotion Classification.

Loads the best checkpoint, runs inference on the held-out test set,
and writes a full report to results/<model_name>/.

Outputs:
  - test_report.txt          per-class precision / recall / F1
  - test_metrics.csv         micro/macro/weighted F1, hamming loss, subset accuracy
  - per_class_metrics.csv    per-emotion P/R/F1/support
  - per_class_f1.png         bar chart of per-class F1
  - prediction_heatmap.png   probability heatmap over test subset

CLI:
  python src/test.py
  python src/test.py --config config/config.yaml

Notebook:
  from src.test import evaluate
  metrics = evaluate(config_path="config/config.yaml")
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataloader import get_dataloaders, EMOTION_NAMES, NUM_EMOTIONS
from src.utils import get_result_dir, load_config, set_seed, apply_threshold, find_best_thresholds
from src.train import build_model


###############################################################################
#  Plots
###############################################################################

def _plot_per_class_f1(
    f1_scores: np.ndarray,
    class_names: list,
    save_path: str,
) -> None:
    """Horizontal bar chart of per-class F1 scores."""
    fig, ax = plt.subplots(figsize=(8, 10))
    y_pos   = np.arange(len(class_names))
    bars    = ax.barh(y_pos, f1_scores, align="center", color="steelblue", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Score (test set)")
    ax.set_xlim(0, 1.0)
    ax.axvline(f1_scores.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean={f1_scores.mean():.3f}")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, f1_scores):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_thresholds(
    thresholds:  np.ndarray,
    class_names: list,
    save_path:   str,
) -> None:
    """Bar chart showing the best threshold found per class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos   = np.arange(len(class_names))
    colors  = ["#e74c3c" if t < 0.4 else "#2ecc71" if t > 0.6 else "#3498db"
               for t in thresholds]
    ax.barh(y_pos, thresholds, align="center", color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="default=0.5")
    ax.set_xlabel("Best Threshold (val F1)")
    ax.set_title("Per-Class Optimal Threshold")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=8)
    for y, val in zip(y_pos, thresholds):
        ax.text(val + 0.01, y, f"{val:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
    n_samples: int = 60,
) -> None:
    """
    Heatmap showing predicted probabilities for a random subset of test samples.
    Rows = samples, Columns = emotions. True positive cells are bordered.
    """
    idx = np.random.default_rng(0).choice(len(probs), size=min(n_samples, len(probs)), replace=False)
    p   = probs[idx]
    l   = labels[idx]

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(p, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Probability")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Sample index (subset)")
    ax.set_title("Predicted Emotion Probabilities (test subset)")

    # Mark true positives with a dot
    rows, cols = np.where(l == 1)
    ax.scatter(cols, rows, marker="x", color="blue", s=15, linewidths=0.8, label="True label")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


###############################################################################
#  Main evaluation function
###############################################################################

def evaluate(config_path: str = "config/config.yaml") -> Dict:
    """
    Load best checkpoint, tìm per-class threshold tốt nhất trên val set,
    sau đó evaluate trên test set với các threshold đó.

    Returns dict with keys:
        micro_f1, macro_f1, weighted_f1, hamming, subset_accuracy,
        best_thresholds (np.ndarray shape (C,)),
        report_path, metrics_csv_path, per_class_csv_path,
        f1_chart_path, heatmap_path, threshold_chart_path.
    """
    cfg = load_config(config_path)
    set_seed(cfg["data"]["seed"])

    model_name = cfg["model"]["name"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] Device: {device}")

    # ── Data — cần cả val lẫn test ──────────────────────────────────────────
    _, val_loader, test_loader, info = get_dataloaders(cfg)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = os.path.join("checkpoints", model_name, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. Run train.py first."
        )

    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] Loaded epoch {ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # ── Helper: run inference on any loader ──────────────────────────────────
    def _infer(loader, desc: str):
        probs_list, labels_list = [], []
        pbar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                logits = model(
                    batch["input_ids"].to(device,      non_blocking=True),
                    batch["attention_mask"].to(device, non_blocking=True),
                )
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
                labels_list.append(batch["labels"].numpy())
        pbar.close()
        return np.vstack(probs_list), np.vstack(labels_list)

    # ── Step 1: Per-class threshold search on VAL set ────────────────────────
    print("\n[test] Searching best per-class threshold on validation set...")
    val_probs, val_labels = _infer(val_loader, "Val inference")

    best_thresholds = find_best_thresholds(
        val_probs, val_labels,
        candidates=np.arange(0.05, 0.95, 0.05),
        metric="f1",
    )

    print(f"[test] Per-class best thresholds:")
    for i, (name, t) in enumerate(zip(EMOTION_NAMES, best_thresholds)):
        val_f1 = f1_score(val_labels[:, i], (val_probs[:, i] >= t).astype(int), zero_division=0)
        print(f"  {name:<15}: threshold={t:.2f}  val_F1={val_f1:.4f}")

    # ── Step 2: Inference on TEST set ────────────────────────────────────────
    print("\n[test] Running inference on test set...")
    all_probs, all_labels = _infer(test_loader, "Test  inference")

    # Apply per-class thresholds — shape broadcast (N,C) >= (C,)
    all_preds = apply_threshold(all_probs, best_thresholds)   # (N, 11)

    # ── Aggregate metrics ────────────────────────────────────────────────────
    micro_f1    = f1_score(all_labels, all_preds, average="micro",    zero_division=0)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    hamming     = hamming_loss(all_labels, all_preds)
    subset_acc  = float((all_preds == all_labels).all(axis=1).mean())

    # Per-class metrics
    per_class_f1  = f1_score(all_labels,  all_preds, average=None, zero_division=0)
    per_class_p   = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_r   = recall_score(all_labels,    all_preds, average=None, zero_division=0)
    per_class_sup = all_labels.sum(axis=0).astype(int)

    report = classification_report(
        all_labels, all_preds, target_names=EMOTION_NAMES, zero_division=0
    )

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  Model            : {model_name}")
    print(f"  Threshold        : {threshold}")
    print(f"  Micro  F1        : {micro_f1:.4f}")
    print(f"  Macro  F1        : {macro_f1:.4f}")
    print(f"  Weighted F1      : {weighted_f1:.4f}")
    print(f"  Hamming Loss     : {hamming:.4f}")
    print(f"  Subset Accuracy  : {subset_acc:.4f}")
    print("=" * 62)
    print("\nPer-class Classification Report:\n")
    print(report)

    # ── Save outputs ─────────────────────────────────────────────────────────
    result_dir = get_result_dir(cfg)

    # 1. Text report
    report_path = os.path.join(result_dir, "test_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model          : {model_name}\n")
        f.write(f"Checkpoint epoch: {ckpt.get('epoch','?')}\n")
        f.write(f"Per-class thresholds (optimised on val set):\n")
        for name, t in zip(EMOTION_NAMES, best_thresholds):
            f.write(f"  {name:<15}: {t:.2f}\n")
        f.write(f"\nMicro  F1      : {micro_f1:.4f}\n")
        f.write(f"Macro  F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1    : {weighted_f1:.4f}\n")
        f.write(f"Hamming Loss   : {hamming:.4f}\n")
        f.write(f"Subset Accuracy: {subset_acc:.4f}\n\n")
        f.write("Per-class Classification Report:\n")
        f.write(report)

    # 2. Aggregate metrics CSV
    metrics_csv = os.path.join(result_dir, "test_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "micro_f1", "macro_f1", "weighted_f1",
            "hamming_loss", "subset_accuracy", "best_epoch",
        ])
        writer.writeheader()
        writer.writerow({
            "model":           model_name,
            "micro_f1":        round(micro_f1,    4),
            "macro_f1":        round(macro_f1,    4),
            "weighted_f1":     round(weighted_f1, 4),
            "hamming_loss":    round(hamming,     4),
            "subset_accuracy": round(subset_acc,  4),
            "best_epoch":      ckpt.get("epoch", "?"),
        })

    # 3. Per-class metrics CSV (includes best threshold per class)
    per_class_csv = os.path.join(result_dir, "per_class_metrics.csv")
    with open(per_class_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "emotion", "threshold", "precision", "recall", "f1", "support"
        ])
        writer.writeheader()
        for i, name in enumerate(EMOTION_NAMES):
            writer.writerow({
                "emotion":   name,
                "threshold": round(float(best_thresholds[i]), 2),
                "precision": round(float(per_class_p[i]),     4),
                "recall":    round(float(per_class_r[i]),     4),
                "f1":        round(float(per_class_f1[i]),    4),
                "support":   int(per_class_sup[i]),
            })

    # 4. Per-class F1 bar chart
    f1_chart_path = os.path.join(result_dir, "per_class_f1.png")
    _plot_per_class_f1(per_class_f1, EMOTION_NAMES, f1_chart_path)

    # 5. Per-class threshold chart
    threshold_chart_path = os.path.join(result_dir, "per_class_thresholds.png")
    _plot_thresholds(best_thresholds, EMOTION_NAMES, threshold_chart_path)

    # 6. Prediction heatmap
    heatmap_path = os.path.join(result_dir, "prediction_heatmap.png")
    _plot_prediction_heatmap(all_probs, all_labels, EMOTION_NAMES, heatmap_path)

    print(f"\n[test] Report            -> {report_path}")
    print(f"[test] Metrics CSV       -> {metrics_csv}")
    print(f"[test] Per-class CSV     -> {per_class_csv}")
    print(f"[test] F1 bar chart      -> {f1_chart_path}")
    print(f"[test] Threshold chart   -> {threshold_chart_path}")
    print(f"[test] Heatmap           -> {heatmap_path}")

    return {
        "micro_f1":              micro_f1,
        "macro_f1":              macro_f1,
        "weighted_f1":           weighted_f1,
        "hamming":               hamming,
        "subset_accuracy":       subset_acc,
        "best_thresholds":       best_thresholds,
        "report_path":           report_path,
        "metrics_csv_path":      metrics_csv,
        "per_class_csv_path":    per_class_csv,
        "f1_chart_path":         f1_chart_path,
        "threshold_chart_path":  threshold_chart_path,
        "heatmap_path":          heatmap_path,
    }


###############################################################################
#  CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GoEmotions emotion classifier.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    evaluate(config_path=args.config)
