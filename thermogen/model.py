"""Thermostability classifier and its input representation."""

from pathlib import Path

import torch
from torch import nn

EMBEDDING_LAYER = 33
EMBEDDING_DIM = 1280
TEMPERATURE_THRESHOLD = 60.0


class Thermo(nn.Module):
    """Binary classifier with the original notebook checkpoint layout."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(640, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Linear(160, 2),
        )

    def forward(self, embeddings):
        return self.model(embeddings)


def load_embedding(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    try:
        vector = payload["mean_representations"][EMBEDDING_LAYER]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing layer {EMBEDDING_LAYER} embedding") from exc
    if not isinstance(vector, torch.Tensor) or vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"{path}: expected a {EMBEDDING_DIM}-element vector")
    vector = vector.float()
    if not torch.isfinite(vector).all():
        raise ValueError(f"{path}: embedding contains non-finite values")
    return payload, vector


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
