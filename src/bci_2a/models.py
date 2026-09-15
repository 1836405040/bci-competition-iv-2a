"""A compact EEGNet implementation for four-class EEG decoding."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int = 4,
                 dropout: float = 0.5, F1: int = 8, D: int = 2, F2: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, 1, n_channels, n_samples)).flatten(1).shape[1]
        self.classifier = nn.Linear(flattened, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def fit_eegnet(X_train, y_train, X_valid, y_valid, *, epochs: int = 100,
               batch_size: int = 64, lr: float = 1e-3, weight_decay: float = 1e-4,
               seed: int = 42, device: str | None = None) -> tuple[EEGNet, dict[str, list[float]]]:
    """Train EEGNet on one fold and return history.

    Inputs are NumPy arrays shaped ``(trials, channels, samples)``. Validation data
    is used only for reporting and early model selection; no preprocessing is fit here.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.as_tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    X_valid = torch.as_tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
    y_valid = torch.as_tensor(y_valid, dtype=torch.long).to(device)
    model = EEGNet(X_train.shape[2], X_train.shape[3]).to(device)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "valid_accuracy": []}
    for _ in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            accuracy = float((model(X_valid).argmax(1) == y_valid).float().mean().cpu())
        history["train_loss"].append(sum(losses) / max(len(losses), 1))
        history["valid_accuracy"].append(accuracy)
    return model, history
