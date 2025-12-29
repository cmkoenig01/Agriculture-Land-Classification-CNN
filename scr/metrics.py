"""Metrics + plots for classification evaluation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_prob, class_labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)

    try:
        metrics["Loss"] = log_loss(y_true, y_prob, labels=list(range(len(class_labels))))
    except Exception:
        metrics["Loss"] = float("nan")

    try:
        if len(class_labels) == 2:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))
            metrics["ROC-AUC"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
    except Exception:
        metrics["ROC-AUC"] = float("nan")

    metrics["Classification Report"] = classification_report(y_true, y_pred, target_names=class_labels)
    metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


def print_metrics(y_true, y_pred, y_prob, class_labels, model_name: str):
    m = compute_metrics(y_true, y_pred, y_prob, class_labels)

    print(f"\nEvaluation metrics for {model_name}")
    print(f"Accuracy: {m['Accuracy']:.4f}")
    if not np.isnan(m["ROC-AUC"]):
        print(f"ROC-AUC:  {m['ROC-AUC']:.4f}")
    if not np.isnan(m["Loss"]):
        print(f"Loss:    {m['Loss']:.4f}")

    print("\nClassification report:\n")
    print(m["Classification Report"])

    disp = ConfusionMatrixDisplay(confusion_matrix=m["Confusion Matrix"], display_labels=class_labels)
    disp.plot()
    plt.show()
