"""
models/loss.py
Loss functions for multi-label emotion classification.

All losses accept:
    logits : (B, C)  raw un-activated scores
    targets: (B, C)  multi-hot float labels  {0.0, 1.0}

Supported loss types (config key  training.loss):
  + "bce"             - Standard BCEWithLogitsLoss
  + "bce_weighted"    - BCEWithLogitsLoss with auto pos_weight
  + "focal_bce"       - Per-sample focal weighting on BCE
  + "asymmetric"      - Asymmetric Loss (ASL) — best for multi-label imbalance
  + "per_class_asl"   - ASL with DIFFERENT gamma_neg per class (rare vs common)
                        Best for handling rare classes specifically
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "BCELoss",
    "FocalBCELoss",
    "AsymmetricLoss",
    "PerClassASL",
    "get_loss_fn",
]


#==============================================================================
#  1. Standard / Weighted BCE
#==============================================================================

class BCELoss(nn.Module):
    """Binary cross-entropy with logits (multi-label)."""

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


#==============================================================================
#  2. Focal BCE
#==============================================================================

class FocalBCELoss(nn.Module):
    """Focal loss for multi-label (binary) classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs        = torch.sigmoid(logits)
        p_t          = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma
        alpha_t      = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss         = alpha_t * focal_weight * bce_loss

        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss


#==============================================================================
#  3. Asymmetric Loss  (standard — same gamma for all classes)
#==============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (Ridnik et al., 2021).
    Uses different gamma for positives vs negatives, same for all classes.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip:      float = 0.05,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs     = torch.sigmoid(logits)
        probs_neg = (probs - self.clip).clamp(min=0.0) if self.clip > 0 else probs

        loss_pos = targets       * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1.0 - probs_neg).clamp(min=1e-8))

        if self.gamma_pos > 0:
            loss_pos = ((1.0 - probs) ** self.gamma_pos) * loss_pos
        if self.gamma_neg > 0:
            loss_neg = (probs_neg ** self.gamma_neg) * loss_neg

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss


#==============================================================================
#  4. Per-Class ASL  (different gamma per class group)
#==============================================================================

class PerClassASL(nn.Module):
    """
    Asymmetric Loss with class-specific gamma values.

    Rare classes (few positive samples) use a lower gamma_neg so the model
    is less aggressive at suppressing their negative predictions, keeping it
    more sensitive to weak positive signals.

    Common classes use higher gamma_neg to down-weight easy negatives
    and focus on hard examples.

    Args:
        rare_indices:   List of class indices treated as "rare".
        gamma_pos_common, gamma_neg_common: ASL gammas for common classes.
        gamma_pos_rare,   gamma_neg_rare:   ASL gammas for rare classes.
        clip_common:    Probability shift for common classes.
        clip_rare:      Probability shift for rare classes (usually 0).
        reduction:      'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        rare_indices:       List[int],
        gamma_pos_common:   float = 1.0,
        gamma_neg_common:   float = 2.0,
        gamma_pos_rare:     float = 0.0,
        gamma_neg_rare:     float = 1.0,
        clip_common:        float = 0.0,
        clip_rare:          float = 0.0,
        reduction:          str   = "mean",
    ) -> None:
        super().__init__()
        self.rare_indices      = rare_indices
        self.gamma_pos_common  = gamma_pos_common
        self.gamma_neg_common  = gamma_neg_common
        self.gamma_pos_rare    = gamma_pos_rare
        self.gamma_neg_rare    = gamma_neg_rare
        self.clip_common       = clip_common
        self.clip_rare         = clip_rare
        self.reduction         = reduction

    def _asl_loss(
        self,
        probs:     torch.Tensor,   # (B,)
        targets:   torch.Tensor,   # (B,)
        gamma_pos: float,
        gamma_neg: float,
        clip:      float,
    ) -> torch.Tensor:
        """Compute ASL loss for a single class column."""
        probs_neg = (probs - clip).clamp(min=0.0) if clip > 0 else probs

        loss_pos = targets       * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1.0 - probs_neg).clamp(min=1e-8))

        if gamma_pos > 0:
            loss_pos = ((1.0 - probs) ** gamma_pos) * loss_pos
        if gamma_neg > 0:
            loss_neg = (probs_neg ** gamma_neg) * loss_neg

        return -(loss_pos + loss_neg)   # (B,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C)
            targets: (B, C) multi-hot float
        """
        probs       = torch.sigmoid(logits)            # (B, C)
        num_classes = logits.size(1)
        loss_cols   = []

        for c in range(num_classes):
            if c in self.rare_indices:
                col_loss = self._asl_loss(
                    probs[:, c], targets[:, c],
                    self.gamma_pos_rare, self.gamma_neg_rare, self.clip_rare,
                )
            else:
                col_loss = self._asl_loss(
                    probs[:, c], targets[:, c],
                    self.gamma_pos_common, self.gamma_neg_common, self.clip_common,
                )
            loss_cols.append(col_loss.unsqueeze(1))

        loss = torch.cat(loss_cols, dim=1)             # (B, C)

        if self.reduction == "mean":  return loss.mean()
        if self.reduction == "sum":   return loss.sum()
        return loss


#==============================================================================
#  5. Factory
#==============================================================================

def get_loss_fn(
    cfg:        dict,
    device:     torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build and return the loss function from config.

    Config keys (under training:):
        loss                  Default: "per_class_asl"
        focal_gamma           Default: 2.0
        asl_gamma_pos         Default: 0.0
        asl_gamma_neg         Default: 4.0
        asl_clip              Default: 0.05
        asl_gamma_pos_common  Default: 1.0
        asl_gamma_neg_common  Default: 2.0
        asl_gamma_pos_rare    Default: 0.0
        asl_gamma_neg_rare    Default: 1.0
        asl_clip_rare         Default: 0.0
    """
    from src.dataloader import RARE_INDICES   # avoid circular at module level

    train_cfg = cfg.get("training", {})
    loss_name = train_cfg.get("loss", "per_class_asl").lower().strip()

    focal_g  = float(train_cfg.get("focal_gamma",   2.0))
    asl_gp   = float(train_cfg.get("asl_gamma_pos", 0.0))
    asl_gn   = float(train_cfg.get("asl_gamma_neg", 4.0))
    asl_clip = float(train_cfg.get("asl_clip",      0.05))

    gp_common = float(train_cfg.get("asl_gamma_pos_common", 1.0))
    gn_common = float(train_cfg.get("asl_gamma_neg_common", 2.0))
    gp_rare   = float(train_cfg.get("asl_gamma_pos_rare",   0.0))
    gn_rare   = float(train_cfg.get("asl_gamma_neg_rare",   1.0))
    clip_rare = float(train_cfg.get("asl_clip_rare",        0.0))

    if loss_name == "bce":
        return BCELoss(pos_weight=None).to(device)

    elif loss_name == "bce_weighted":
        if pos_weight is None:
            raise ValueError("loss='bce_weighted' requires pos_weight.")
        return BCELoss(pos_weight=pos_weight.to(device)).to(device)

    elif loss_name == "focal_bce":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return FocalBCELoss(gamma=focal_g, pos_weight=pw).to(device)

    elif loss_name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=asl_gp, gamma_neg=asl_gn, clip=asl_clip,
        ).to(device)

    elif loss_name == "per_class_asl":
        return PerClassASL(
            rare_indices=RARE_INDICES,
            gamma_pos_common=gp_common,
            gamma_neg_common=gn_common,
            gamma_pos_rare=gp_rare,
            gamma_neg_rare=gn_rare,
            clip_common=0.0,
            clip_rare=clip_rare,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            "Choose from: bce | bce_weighted | focal_bce | asymmetric | per_class_asl"
        )


from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "BCELoss",
    "FocalBCELoss",
    "AsymmetricLoss",
    "get_loss_fn",
]


#==============================================================================
#  1. Standard / Weighted BCE
#==============================================================================

class BCELoss(nn.Module):
    """
    Binary cross-entropy with logits (multi-label).

    Args:
        pos_weight: Tensor of shape (num_classes,) for class imbalance.
                    pos_weight[c] = neg_samples[c] / pos_samples[c]
        reduction:  'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C) raw scores.
            targets: (B, C) multi-hot float labels.
        """
        return self.loss_fn(logits, targets)


#==============================================================================
#  2. Focal BCE
#==============================================================================

class FocalBCELoss(nn.Module):
    """
    Focal loss adapted for multi-label (binary) classification.

    Applies focal modulation independently to each class:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:      Focusing exponent γ >= 0.
        alpha:      Scalar balance factor (applied to positive terms).
        pos_weight: Optional per-class weight tensor.
        reduction:  'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma      = gamma
        self.alpha      = alpha
        self.reduction  = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )   # (B, C)

        probs         = torch.sigmoid(logits)
        # p_t: probability of the true label
        p_t           = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight  = (1.0 - p_t) ** self.gamma

        # alpha_t
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * focal_weight * bce_loss    # (B, C)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


#==============================================================================
#  3. Asymmetric Loss  (ASL)  — Ridnik et al., 2021
#==============================================================================

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Key idea: use *different* gamma values for positives vs negatives.
    Negatives are far more numerous in multi-label tasks, so down-weight
    easy negatives more aggressively.

    Additionally, a probability *shift* (clip) m shifts the negative
    probability down by m before computing the loss, effectively zeroing
    out very confident negative predictions.

        For positive  (y=1): loss = -(1-p)^{γ+}  * log(p)
        For negative  (y=0): loss = -(max(p-m,0))^{γ-} * log(1 - max(p-m,0))

    Recommended defaults: γ+ = 0, γ- = 4, m = 0.05

    Args:
        gamma_pos:  γ for positive samples (usually 0 = standard BCE for positives).
        gamma_neg:  γ for negative samples (usually 2–4).
        clip:       Probability margin m for negative shifting.
        reduction:  'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip      = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C)
            targets: (B, C) multi-hot float
        """
        probs = torch.sigmoid(logits)

        # Probability shifting for negatives
        if self.clip > 0:
            probs_neg = (probs - self.clip).clamp(min=0.0)
        else:
            probs_neg = probs

        # Positive and negative BCE separately
        loss_pos = targets       * (torch.log(probs.clamp(min=1e-8)))
        loss_neg = (1 - targets) * (torch.log((1.0 - probs_neg).clamp(min=1e-8)))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            focal_pos = (1.0 - probs) ** self.gamma_pos
            loss_pos  = focal_pos * loss_pos

        if self.gamma_neg > 0:
            focal_neg = probs_neg ** self.gamma_neg
            loss_neg  = focal_neg * loss_neg

        loss = -(loss_pos + loss_neg)               # (B, C)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


#==============================================================================
#  4. Factory
#==============================================================================

def get_loss_fn(
    cfg: dict,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Build and return the loss function specified in ``cfg['training']['loss']``.

    Config keys consulted (all under ``training:``):
        loss          (str)   Name of the loss.       Default: "asymmetric"
        focal_gamma   (float) Focal BCE γ.             Default: 2.0
        asl_gamma_pos (float) ASL γ for positives.    Default: 0.0
        asl_gamma_neg (float) ASL γ for negatives.    Default: 4.0
        asl_clip      (float) ASL probability shift.  Default: 0.05

    Args:
        cfg:        Parsed config dict.
        device:     Target device.
        pos_weight: Pre-computed per-class positive weight tensor (27,).
                    Used by "bce_weighted". Pass None to skip weighting.

    Returns:
        An ``nn.Module`` with signature ``forward(logits, targets) -> Tensor``.
    """
    train_cfg = cfg.get("training", {})

    loss_name  = train_cfg.get("loss",          "asymmetric").lower().strip()
    focal_g    = float(train_cfg.get("focal_gamma",   2.0))
    asl_gp     = float(train_cfg.get("asl_gamma_pos", 0.0))
    asl_gn     = float(train_cfg.get("asl_gamma_neg", 4.0))
    asl_clip   = float(train_cfg.get("asl_clip",      0.05))

    if loss_name == "bce":
        return BCELoss(pos_weight=None).to(device)

    elif loss_name == "bce_weighted":
        if pos_weight is None:
            raise ValueError(
                "loss='bce_weighted' requires pos_weight. "
                "Pass the tensor returned by compute_pos_weight()."
            )
        return BCELoss(pos_weight=pos_weight.to(device)).to(device)

    elif loss_name == "focal_bce":
        pw = pos_weight.to(device) if pos_weight is not None else None
        return FocalBCELoss(gamma=focal_g, pos_weight=pw).to(device)

    elif loss_name == "asymmetric":
        return AsymmetricLoss(
            gamma_pos=asl_gp,
            gamma_neg=asl_gn,
            clip=asl_clip,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown loss '{loss_name}'. "
            "Choose from: bce | bce_weighted | focal_bce | asymmetric"
        )
