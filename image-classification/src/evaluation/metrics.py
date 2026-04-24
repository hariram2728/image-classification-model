"""
Evaluation metrics for image classification.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)


class AccuracyMeter:
    """Compute and store accuracy metrics."""

    def __init__(self, top_k: Tuple[int, ...] = (1, 5)):
        self.top_k = top_k
        self.correct = {k: 0 for k in top_k}
        self.total = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with batch predictions.

        Args:
            outputs: Model output logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        batch_size = targets.size(0)
        self.total += batch_size

        # Get top-k predictions
        with torch.no_grad():
            max_k = max(self.top_k)
            _, pred = outputs.topk(max_k, dim=1)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for k in self.top_k:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                self.correct[k] += correct_k.item()

    def compute(self) -> Dict[str, float]:
        """Compute accuracy for each k."""
        return {f"acc@{k}": 100.0 * self.correct[k] / self.total for k in self.top_k}

    def reset(self):
        """Reset all counters."""
        self.correct = {k: 0 for k in self.top_k}
        self.total = 0


class ClassificationMetrics:
    """
    Compute comprehensive classification metrics.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []

    def update(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """
        Update metrics with batch predictions.

        Args:
            outputs: Model output logits or predictions
            targets: Ground truth labels
            probs: Predicted probabilities (optional)
        """
        if outputs.dim() > 1:
            preds = outputs.argmax(dim=1)
        else:
            preds = outputs

        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())

        if probs is not None:
            self.all_probs.append(probs.cpu().numpy())

    def compute_all(self) -> Dict[str, any]:
        """
        Compute all classification metrics.

        Returns:
            Dictionary containing all metrics
        """
        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Per-class metrics
        metrics["per_class_precision"] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics["per_class_recall"] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()
        metrics["per_class_f1"] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist()

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # AUC-ROC (if we have probabilities)
        if self.all_probs:
            y_probs = np.vstack(self.all_probs)
            try:
                metrics["auc_macro"] = roc_auc_score(
                    y_true, y_probs, average="macro", multi_class="ovr"
                )
                metrics["auc_micro"] = roc_auc_score(
                    y_true, y_probs, average="micro", multi_class="ovr"
                )
            except ValueError:
                metrics["auc_macro"] = 0.0
                metrics["auc_micro"] = 0.0

        return metrics

    def get_classification_report(self) -> str:
        """Get sklearn classification report string."""
        return classification_report(
            self.all_targets,
            self.all_preds,
            target_names=self.class_names,
        )

    def reset(self):
        """Reset all stored predictions and targets."""
        self.all_preds = []
        self.all_targets = []
        self.all_probs = []


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.

        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_metrics(
    predictions: List[int],
    targets: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """
    Compute all standard classification metrics.

    Args:
        predictions: List of predicted labels
        targets: List of ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = ClassificationMetrics(num_classes)
    metrics.all_preds = predictions
    metrics.all_targets = targets
    return metrics.compute_all()
