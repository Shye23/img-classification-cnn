from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from utils import save_json


def evaluate(model, val_loader, device, criterion=None, output_dir=None, epoch=None, logger=None):
    model.eval()

    all_preds = []
    all_labels = []
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

            total_samples += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    validation_loss = running_loss / total_samples if total_samples else 0.0
    validation_accuracy = (
        sum(int(pred == label) for pred, label in zip(all_preds, all_labels)) / total_samples
        if total_samples
        else 0.0
    )

    class_names = getattr(getattr(val_loader, "dataset", None), "classes", None)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names if class_names is not None else None,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(all_labels, all_preds)

    metrics = {
        "epoch": epoch,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        suffix = f"epoch_{epoch:03d}" if epoch is not None else "latest"

        save_json(output_path / f"validation_metrics_{suffix}.json", metrics)
        save_json(
            output_path / f"confusion_matrix_{suffix}.json",
            {
                "epoch": epoch,
                "labels": class_names if class_names is not None else list(range(len(matrix))),
                "confusion_matrix": matrix.tolist(),
            },
        )
        (output_path / f"classification_report_{suffix}.txt").write_text(report, encoding="utf-8")

    summary = f"Validation Loss: {validation_loss:.4f} | Validation Accuracy: {validation_accuracy:.4f}"
    if logger is not None:
        logger.info(summary)
    else:
        print(summary)

    return metrics