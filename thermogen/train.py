"""Train the ESM-2 thermostability classifier on existing data splits."""

import argparse
import math
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .model import Thermo, TEMPERATURE_THRESHOLD, load_embedding, resolve_device


class TrainingEmbeddings(Dataset):
    def __init__(self, directory):
        self.paths = sorted(path for path in directory.glob("*.pt") if path.is_file())
        if not self.paths:
            raise ValueError(f"No .pt embeddings found in {directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        payload, vector = load_embedding(path)
        try:
            temperature = float(payload["label"].rsplit("_", 1)[-1])
        except (KeyError, AttributeError, ValueError) as exc:
            raise ValueError(f"{path}: label must end with a temperature in °C") from exc
        if not math.isfinite(temperature):
            raise ValueError(f"{path}: temperature must be finite")
        return vector, int(temperature >= TEMPERATURE_THRESHOLD)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    targets, predictions, scores = [], [], []
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for vectors, labels in loader:
            logits = model(vectors.to(device))
            loss = loss_fn(logits, labels.to(device))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.numel()
            targets.extend(labels.tolist())
            predictions.extend(logits.detach().argmax(dim=1).tolist())
            scores.extend(logits.detach().softmax(dim=1)[:, 1].tolist())

    metrics = {
        "loss": total_loss / len(targets),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "mcc": matthews_corrcoef(targets, predictions),
    }
    if len(set(targets)) == 2:
        metrics["roc_auc"] = roc_auc_score(targets, scores)
        metrics["average_precision"] = average_precision_score(targets, scores)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.num_workers < 0:
        parser.error("epochs must be positive, batch size at least 2, and workers non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.train_dir.resolve() == args.val_dir.resolve():
        parser.error("training and validation require separate directories")

    train_data = TrainingEmbeddings(args.train_dir)
    val_data = TrainingEmbeddings(args.val_dir)
    if len(train_data) < args.batch_size:
        parser.error("training set is smaller than one batch. Reduce --batch-size.")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = Thermo().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    best_accuracy = -1.0
    with SummaryWriter(str(args.output_dir / "tensorboard")) as writer:
        for epoch in range(args.epochs):
            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_metrics = run_epoch(model, val_loader, loss_fn, device)
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                message = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                print(f"epoch={epoch} {split} {message}", flush=True)
                for key, value in metrics.items():
                    writer.add_scalar(f"{split}/{key}", value, epoch)

            state = model.state_dict()
            torch.save(state, args.output_dir / f"60_tm_discriminator_{epoch}.pt")
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save(state, args.output_dir / "best_model_epoch.pt")
            scheduler.step()


if __name__ == "__main__":
    main()
