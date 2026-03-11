"""
src/dataloader.py
SemEval 2018 Task 1 — Multi-Label Emotion Classification DataLoader.

Dataset format: Tab-separated .txt files with columns:
    ID  |  Tweet  |  anger  |  anticipation  |  disgust  |  fear  |  joy  |
    love  |  optimism  |  pessimism  |  sadness  |  surprise  |  trust

11 emotion labels, values are 0 or 1 (already binary, no Neutral class).

Rare-class strategy (3 layers):
  1. SynonymAugDataset  — offline synonym-replacement augmentation applied
                          only to samples containing rare-class labels.
  2. build_weighted_sampler — exponential inverse-frequency weighting so
                              rare-class samples are drawn much more often.
  3. compute_pos_weight   — per-class BCE pos_weight (used by loss fn).
"""

from __future__ import annotations

import os
import random
import re
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

# Classes with train count < 1000 — treated as "rare"
RARE_CLASSES: List[str] = ["trust", "surprise", "anticipation", "pessimism", "love"]
RARE_INDICES: List[int]  = [EMOTION_NAMES.index(c) for c in RARE_CLASSES]


#==============================================================================
#  Simple synonym-replacement augmentation (no external library needed)
#==============================================================================

# Hand-crafted synonym map for common Twitter words. Extend as needed.
_SYNONYM_MAP: Dict[str, List[str]] = {
    "happy":       ["glad", "pleased", "delighted", "joyful"],
    "good":        ["great", "wonderful", "fantastic", "excellent"],
    "bad":         ["terrible", "awful", "horrible", "dreadful"],
    "sad":         ["unhappy", "miserable", "sorrowful", "depressed"],
    "angry":       ["furious", "enraged", "mad", "irritated"],
    "scared":      ["afraid", "frightened", "terrified", "anxious"],
    "love":        ["adore", "cherish", "treasure", "care about"],
    "hate":        ["despise", "loathe", "detest", "dislike"],
    "think":       ["believe", "feel", "consider", "reckon"],
    "amazing":     ["incredible", "awesome", "remarkable", "stunning"],
    "hard":        ["difficult", "tough", "challenging", "demanding"],
    "hope":        ["wish", "expect", "trust", "anticipate"],
    "worry":       ["fear", "dread", "fret", "stress"],
    "thankful":    ["grateful", "appreciative", "blessed"],
    "proud":       ["honored", "pleased", "satisfied", "glad"],
    "awful":       ["terrible", "dreadful", "horrible", "atrocious"],
    "excited":     ["thrilled", "enthusiastic", "eager", "pumped"],
    "surprised":   ["shocked", "astonished", "stunned", "amazed"],
    "trust":       ["rely on", "depend on", "believe in", "have faith in"],
    "pessimistic": ["cynical", "negative", "doubtful", "skeptical"],
}


def _synonym_replace(text: str, n: int = 2, seed: int = None) -> str:
    """
    Replace up to n words in text with synonyms from _SYNONYM_MAP.

    Args:
        text:  Input string.
        n:     Max number of replacements.
        seed:  Optional random seed for reproducibility.

    Returns:
        Augmented string (may be identical if no matches found).
    """
    rng    = random.Random(seed)
    words  = text.split()
    idxs   = list(range(len(words)))
    rng.shuffle(idxs)

    replaced = 0
    for i in idxs:
        w = re.sub(r"[^a-zA-Z]", "", words[i]).lower()
        if w in _SYNONYM_MAP and replaced < n:
            words[i] = rng.choice(_SYNONYM_MAP[w])
            replaced += 1

    return " ".join(words)


#==============================================================================
#  Dataset  (with optional rare-class augmentation)
#==============================================================================

class SemEvalDataset(Dataset):
    """
    PyTorch Dataset for SemEval 2018 Task 1 TSV files.

    For training splits, samples that contain at least one rare-class label
    are augmented with synonym replacement (``augment_rare=True``).
    This effectively duplicates rare-class samples with slightly perturbed text,
    giving the model more diverse signal without changing label distribution.

    Args:
        filepath:     Path to the .txt (TSV) file.
        tokenizer:    HuggingFace tokenizer.
        max_length:   Maximum token length.
        augment_rare: If True, append synonym-augmented copies of rare-class
                      samples to the dataset (train only).
        aug_copies:   Number of augmented copies per rare-class sample.
    """

    def __init__(
        self,
        filepath:     str,
        tokenizer:    AutoTokenizer,
        max_length:   int  = 128,
        augment_rare: bool = False,
        aug_copies:   int  = 2,
    ) -> None:
        self.tokenizer    = tokenizer
        self.max_length   = max_length

        df = pd.read_csv(filepath, sep="\t")
        missing = [c for c in EMOTION_NAMES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in {filepath}: {missing}")

        texts  = df["Tweet"].astype(str).tolist()
        labels = df[EMOTION_NAMES].values.astype(np.float32)  # (N, 11)

        # ── Rare-class augmentation (train only) ─────────────────────────────
        if augment_rare and aug_copies > 0:
            extra_texts:  List[str]       = []
            extra_labels: List[np.ndarray] = []

            rare_mask = labels[:, RARE_INDICES].sum(axis=1) > 0  # (N,) bool

            for i, (t, l) in enumerate(zip(texts, labels)):
                if rare_mask[i]:
                    for copy_idx in range(aug_copies):
                        aug = _synonym_replace(t, n=2, seed=i * 100 + copy_idx)
                        extra_texts.append(aug)
                        extra_labels.append(l.copy())

            texts  = texts  + extra_texts
            labels = np.vstack([labels, np.array(extra_labels)]) if extra_labels else labels

            n_orig = len(df)
            n_aug  = len(extra_texts)
            print(f"[DataLoader] Augmented {sum(rare_mask)} rare-class samples "
                  f"× {aug_copies} copies → +{n_aug} samples  "
                  f"(total: {n_orig} → {n_orig + n_aug})")

        self.texts  = texts
        self.labels = labels

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float32),
        }


#==============================================================================
#  Weighted sampler — exponential oversampling for rare classes
#==============================================================================

def build_weighted_sampler(
    dataset:      SemEvalDataset,
    power:        float = 2.0,
    rare_boost:   float = 3.0,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler with exponential inverse-frequency weighting.

    Improvements over the previous mean-inv-freq version:
      1. ``power`` raises inv_freq to a power before averaging, making the
         contrast between rare and common classes more extreme.
      2. ``rare_boost`` additionally multiplies the weight of any sample
         that contains at least one RARE_CLASS label.

    Args:
        dataset:    SemEvalDataset (train split).
        power:      Exponent applied to per-class inverse frequency.
                    1.0 = linear (old behaviour), 2.0 = quadratic (default).
        rare_boost: Extra multiplier for samples with rare-class labels.

    Returns:
        WeightedRandomSampler producing len(dataset) samples per epoch.
    """
    labels_mat   = dataset.labels                              # (N, 11)
    label_counts = labels_mat.sum(axis=0).clip(min=1)         # (11,)
    inv_freq     = (1.0 / label_counts) ** power              # exponential

    sample_weights = np.zeros(len(dataset), dtype=np.float64)
    for i, row in enumerate(labels_mat):
        pos = row > 0
        sample_weights[i] = inv_freq[pos].mean() if pos.any() else inv_freq.min()

    # Extra boost for rare-class samples
    rare_mask = labels_mat[:, RARE_INDICES].sum(axis=1) > 0   # (N,)
    sample_weights[rare_mask] *= rare_boost

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


#==============================================================================
#  pos_weight for BCE losses
#==============================================================================

def compute_pos_weight(
    dataset: SemEvalDataset,
    device:  torch.device,
    scale:   float = 1.5,
) -> torch.Tensor:
    """
    Per-class positive weight for BCEWithLogitsLoss:
        pos_weight[c] = ((N - n_pos[c]) / n_pos[c]) ^ scale

    ``scale > 1`` amplifies the weight difference between rare and common classes.
    Default scale=1.5 gives a moderate boost without overfitting to rare classes.

    Returns:
        Tensor of shape (11,) on device.
    """
    n            = len(dataset)
    label_counts = dataset.labels.sum(axis=0).clip(min=1)
    pos_weight   = ((n - label_counts) / label_counts) ** scale
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


#==============================================================================
#  Main factory
#==============================================================================

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train / val / test DataLoaders from SemEval 2018 TSV files.

    Config keys used (under ``data:``):
        augment_rare  (bool)  Enable synonym augmentation for rare classes.
                              Default: true
        aug_copies    (int)   Augmented copies per rare-class sample.
                              Default: 2
        sampler_power (float) Exponent for inverse-freq weighting.
                              Default: 2.0
        rare_boost    (float) Extra weight multiplier for rare-class samples.
                              Default: 3.0
        pw_scale      (float) pos_weight scaling exponent.
                              Default: 1.5

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
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

    # Rare-class augmentation config
    augment_rare  = bool(data_cfg.get("augment_rare",  True))
    aug_copies    = int(data_cfg.get("aug_copies",     2))
    sampler_power = float(data_cfg.get("sampler_power", 2.0))
    rare_boost    = float(data_cfg.get("rare_boost",    3.0))
    pw_scale      = float(data_cfg.get("pw_scale",      1.5))

    for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[DataLoader] {split} file not found: '{path}'"
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
    train_ds = SemEvalDataset(train_path, tokenizer, max_length,
                              augment_rare=augment_rare, aug_copies=aug_copies)
    val_ds   = SemEvalDataset(val_path,   tokenizer, max_length,
                              augment_rare=False)
    test_ds  = SemEvalDataset(test_path,  tokenizer, max_length,
                              augment_rare=False)

    # ── Sampler ──────────────────────────────────────────────────────────────
    train_sampler = build_weighted_sampler(
        train_ds, power=sampler_power, rare_boost=rare_boost
    )

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
        else compute_pos_weight(train_ds, device, scale=pw_scale)
    )

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DataLoader] Backbone     : {model_name} ({pretrained})")
    print(f"[DataLoader] augment_rare : {augment_rare}  ×{aug_copies} copies")
    print(f"[DataLoader] sampler_power: {sampler_power}  rare_boost: {rare_boost}")
    print(f"[DataLoader] pw_scale     : {pw_scale}")
    print(f"[DataLoader] pos_weight   :")
    for i, name in enumerate(EMOTION_NAMES):
        print(f"    {name:<15}: {pos_weight[i].item():.2f}")

    return train_loader, val_loader, test_loader, {
        "emotion_names": EMOTION_NAMES,
        "pos_weight":    pos_weight,
        "label_counts":  label_counts_dict,
    }
