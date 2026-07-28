"""Optional small temporal convolution candidate for grouped CSI windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TcnArtifact:
    classes: tuple[str, ...]
    sequence_length: int
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    state_dict: dict


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "TCN candidate requires the optional requirements-tcn.txt"
        ) from error
    return torch, nn


def build_sequences(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    sequence_length: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    labels: list[str] = []
    sequence_groups: list[str] = []
    for group in np.unique(groups):
        indexes = np.flatnonzero(groups == group)
        for offset in range(0, len(indexes) - sequence_length + 1):
            selected = indexes[offset : offset + sequence_length]
            if len(set(y[selected])) != 1:
                continue
            sequences.append(x[selected].T)
            labels.append(str(y[selected[-1]]))
            sequence_groups.append(str(group))
    if not sequences:
        raise ValueError("no contiguous grouped sequences can be constructed")
    return (
        np.stack(sequences).astype(np.float32),
        np.asarray(labels),
        np.asarray(sequence_groups),
    )


def _create_model(input_channels: int, class_count: int):
    torch, nn = _torch()

    class TemporalConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(input_channels, 64, kernel_size=3, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, class_count),
            )

        def forward(self, values):
            return self.network(values)

    return TemporalConvNet()


def fit_tcn(
    sequences: np.ndarray,
    labels: np.ndarray,
    *,
    classes: Sequence[str],
    epochs: int = 40,
    seed: int = 42,
) -> TcnArtifact:
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = np.mean(sequences, axis=(0, 2), keepdims=True)
    scale = np.std(sequences, axis=(0, 2), keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized = (sequences - mean) / scale
    class_to_index = {label: index for index, label in enumerate(classes)}
    target = np.asarray([class_to_index[str(label)] for label in labels])

    model = _create_model(sequences.shape[1], len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    tensor_x = torch.from_numpy(normalized.astype(np.float32))
    tensor_y = torch.from_numpy(target.astype(np.int64))
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(tensor_x), tensor_y)
        loss.backward()
        optimizer.step()
    return TcnArtifact(
        classes=tuple(classes),
        sequence_length=sequences.shape[2],
        feature_mean=mean.reshape(-1).astype(np.float32),
        feature_scale=scale.reshape(-1).astype(np.float32),
        state_dict=model.state_dict(),
    )


def predict_tcn(artifact: TcnArtifact, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = _torch()
    model = _create_model(sequences.shape[1], len(artifact.classes))
    model.load_state_dict(artifact.state_dict)
    mean = artifact.feature_mean.reshape(1, -1, 1)
    scale = artifact.feature_scale.reshape(1, -1, 1)
    normalized = (sequences - mean) / scale
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.from_numpy(normalized.astype(np.float32))), dim=1
        ).numpy()
    indexes = np.argmax(probabilities, axis=1)
    labels = np.asarray([artifact.classes[index] for index in indexes])
    return labels, probabilities
