from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loaders
from evaluate import evaluate
from model import get_model
from utils import append_csv_row, save_json, setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "dataset_final"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_LOG_PATH = OUTPUT_DIR / "training.log"
METRICS_CSV_PATH = OUTPUT_DIR / "metrics.csv"
LAST_MODEL_PATH = OUTPUT_DIR / "model.pth"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger(TRAIN_LOG_PATH)

train_loader, val_loader = get_loaders(
    str(DATA_DIR / "train"),
    str(DATA_DIR / "val"),
)

model = get_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 18
best_val_accuracy = 0.0
best_epoch = 0

logger.info("Starting training on %s", device)

for epoch in range(epochs):
    logger.info("Epoch %s/%s", epoch + 1, epochs)

    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    train_loss = running_loss / total_samples if total_samples else 0.0
    train_accuracy = running_correct / total_samples if total_samples else 0.0

    val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion=criterion,
        output_dir=OUTPUT_DIR,
        epoch=epoch + 1,
        logger=logger,
    )

    append_csv_row(
        METRICS_CSV_PATH,
        ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_metrics["validation_loss"], 6),
            "val_accuracy": round(val_metrics["validation_accuracy"], 6),
        },
    )

    torch.save(model.state_dict(), LAST_MODEL_PATH)

    if val_metrics["validation_accuracy"] >= best_val_accuracy:
        best_val_accuracy = val_metrics["validation_accuracy"]
        best_epoch = epoch + 1
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        logger.info("New best model saved at epoch %s with validation accuracy %.4f", best_epoch, best_val_accuracy)

    logger.info("Train Loss: %.4f | Train Accuracy: %.4f", train_loss, train_accuracy)

save_json(
    OUTPUT_DIR / "training_summary.json",
    {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "last_model_path": str(LAST_MODEL_PATH),
        "best_model_path": str(BEST_MODEL_PATH),
        "metrics_csv_path": str(METRICS_CSV_PATH),
        "training_log_path": str(TRAIN_LOG_PATH),
    },
)

logger.info("Training complete. Last model saved to %s", LAST_MODEL_PATH)
logger.info("Best model saved to %s", BEST_MODEL_PATH)