"""
src/dataloader.py
SemEval 2018 Task 1 — Multi-Label Emotion Classification DataLoader.

Dataset format: Tab-separated .txt files with columns:
    ID  |  Tweet  |  anger  |  anticipation  |  disgust  |  fear  |  joy  |
    love  |  optimism  |  pessimism  |  sadness  |  surprise  |  trust

11 emotion labels, values are 0 or 1 (already binary, no Neutral class).
Samples with all-zero labels (no emotion) are kept as-is — the model
learns to output all-low probabilities for them.

Files:
    data/2018-E-c-En-train.txt   (6838 samples)
    data/2018-E-c-En-dev.txt     (886  samples)
    data/2018-E-c-En-test-gold.txt
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer


#==============================================================================
#  Backbone registry
#==============================================================================

BACKBONE_REGISTRY: Dict[str, Dict[str, str]] = {
    "bert":    {"pretrained": "google-bert/bert-base-uncased"},
    "roberta": {"pretrained": "FacebookAI/roberta-base"},
    "deberta": {"pretrained": "microsoft/deberta-v3-base"},
    "electra": {"pretrained": "google/electra-base-discriminator"},
}


#==============================================================================
#  Label metadata
#==============================================================================

EMOTION_NAMES: List[str] = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "love", "optimism", "pessimism", "sadness", "surprise", "trust",
]
NUM_EMOTIONS = len(EMOTION_NAMES)   # 11


#==============================================================================
#  Dataset
#==============================================================================

class SemEvalDataset(Dataset):
    """
    PyTorch Dataset for SemEval 2018 Task 1 TSV files.

    Reads the TSV, tokenises the Tweet column, returns multi-hot label vectors.

    Args:
        filepath:   Path to the .txt (TSV) file.
        tokenizer:  HuggingFace tokenizer.
        max_length: Maximum token length for padding/truncation.
    """

    def __init__(
        self,
        filepath: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length

        df = pd.read_csv(filepath, sep="\t")

        # Validate expected columns exist
        missing = [c for c in EMOTION_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in {filepath}: {missing}")

        self.texts  = df["Tweet"].astype(str).tolist()
        self.labels = df[EMOTION_NAMES].values.astype(np.float32)  # (N, 11)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),        # (max_length,)
            "attention_mask": enc["attention_mask"].squeeze(0),   # (max_length,)
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),  # (11,)
        }


#==============================================================================
#  Weighted sampler
#==============================================================================

def build_weighted_sampler(dataset: SemEvalDataset) -> WeightedRandomSampler:
    """
    WeightedRandomSampler so that samples containing rare-emotion labels
    are drawn more frequently each epoch.

    Weight of sample i = mean of (1 / label_frequency) over its active labels.
    All-zero samples (no emotion) get the minimum weight.

    Args:
        dataset: A SemEvalDataset instance (train split).

    Returns:
        WeightedRandomSampler producing len(dataset) samples per epoch.
    """
    labels_mat   = dataset.labels                          # (N, 11)
    label_counts = labels_mat.sum(axis=0).clip(min=1)     # (11,) avoid div/0
    inv_freq     = 1.0 / label_counts                     # (11,)

    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, row in enumerate(labels_mat):
        pos = row > 0
        sample_weights[i] = inv_freq[pos].mean() if pos.any() else inv_freq.min()

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


#==============================================================================
#  pos_weight for BCE losses
#==============================================================================

def compute_pos_weight(dataset: SemEvalDataset, device: torch.device) -> torch.Tensor:
    """
    Per-class positive weight for BCEWithLogitsLoss:
        pos_weight[c] = (N - n_pos[c]) / n_pos[c]

    Returns:
        Tensor of shape (11,) on device.
    """
    n            = len(dataset)
    label_counts = dataset.labels.sum(axis=0).clip(min=1)  # (11,)
    pos_weight   = (n - label_counts) / label_counts
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


#==============================================================================
#  Main factory
#==============================================================================

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train / val / test DataLoaders from SemEval 2018 TSV files.

    Args:
        cfg: Parsed config dict.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

        info_dict keys:
            'emotion_names' : list[str]  — 11 label names in column order
            'pos_weight'    : Tensor(11) — for weighted BCE
            'label_counts'  : dict {emotion: int}  — train set counts
    """
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    data_dir   = data_cfg["data_dir"]
    max_length = int(data_cfg.get("max_length", 128))
    batch_size = int(train_cfg.get("batch_size", 32))

    train_file = data_cfg.get("train_file", "2018-E-c-En-train.txt")
    val_file   = data_cfg.get("val_file",   "2018-E-c-En-dev.txt")
    test_file  = data_cfg.get("test_file",  "2018-E-c-En-test-gold.txt")

    train_path = os.path.join(data_dir, train_file)
    val_path   = os.path.join(data_dir, val_file)
    test_path  = os.path.join(data_dir, test_file)

    # ── Validate files exist ─────────────────────────────────────────────────
    for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[DataLoader] {split} file not found: '{path}'\n"
                f"  Set data.{split}_file in config or place file in data_dir='{data_dir}'"
            )

    # ── Tokenizer ────────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {' | '.join(BACKBONE_REGISTRY.keys())}"
        )
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = SemEvalDataset(train_path, tokenizer, max_length)
    val_ds   = SemEvalDataset(val_path,   tokenizer, max_length)
    test_ds  = SemEvalDataset(test_path,  tokenizer, max_length)

    # ── Sampler ──────────────────────────────────────────────────────────────
    train_sampler = build_weighted_sampler(train_ds)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Info ─────────────────────────────────────────────────────────────────
    label_counts_dict = {
        EMOTION_NAMES[i]: int(train_ds.labels[:, i].sum())
        for i in range(NUM_EMOTIONS)
    }

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_pw     = train_cfg.get("pos_weight", None)
    pos_weight = (
        torch.tensor(raw_pw, dtype=torch.float32, device=device)
        if raw_pw is not None
        else compute_pos_weight(train_ds, device)
    )

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DataLoader] Backbone  : {model_name} ({pretrained})")
    print(f"[DataLoader] Max length: {max_length}")
    print(f"[DataLoader] Label counts (train):")
    for k, v in sorted(label_counts_dict.items(), key=lambda x: -x[1]):
        print(f"    {k:<15}: {v}")

    return train_loader, val_loader, test_loader, {
        "emotion_names": EMOTION_NAMES,
        "pos_weight":    pos_weight,
        "label_counts":  label_counts_dict,
    }
