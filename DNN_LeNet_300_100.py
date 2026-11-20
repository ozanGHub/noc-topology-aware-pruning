# DNN_LeNet_300_100.py
# Dataset: MNIST (via torchvision)
# Columns: 784 pixel features, 10 digit classes
# Shuffle -> 80/20 split -> Train -> Evaluate: acc, f1, precision, recall,
# model size, inference time, training time
#
# Model: DNN_LeNet_300_100 MLP 784-300-100-10
from typing import Optional

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from eval_report import EvalReport

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Dataset
# -----------------------------
class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _load_mnist(data_dir: str = "./data"):
    """Download MNIST and return flattened, normalised numpy arrays."""
    train_ds = datasets.MNIST(data_dir, train=True,  download=True)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True)

    mean, std = 0.1307, 0.3081

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_train = ((X_train - mean) / std).astype(np.float32)
    y_train = train_ds.targets.numpy()

    X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_test = ((X_test - mean) / std).astype(np.float32)
    y_test = test_ds.targets.numpy()

    return X_train, X_test, y_train, y_test


# -----------------------------
# Model: 784-300-100-10
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_features: int = 784, h1: int = 300, h2: int = 100,
                 num_classes: int = 10):
        super().__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(h2, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------------
# Utilities
# -----------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def model_state_size_bytes(model: nn.Module) -> int:
    total = 0
    for _, v in model.state_dict().items():
        total += v.numel() * v.element_size()
    return total

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for Xb, _ in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def timed_inference(model: nn.Module, X_test: np.ndarray, device: torch.device, repeats: int = 500) -> dict:
    model.eval()
    Xt = torch.from_numpy(X_test.astype(np.float32)).to(device)
    x1 = Xt[0:1]

    # Warm-up
    with torch.no_grad():
        for _ in range(30):
            _ = model(x1)
            _ = model(Xt)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Batch timing
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(Xt)
    sync()
    t1 = time.perf_counter()
    batch_ms = (t1 - t0) * 1000.0 / repeats

    # Single timing
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats * 10):
            _ = model(x1)
    sync()
    t1 = time.perf_counter()
    single_ms = (t1 - t0) * 1000.0 / (repeats * 10)

    return {"batch_ms": float(batch_ms), "single_ms": float(single_ms)}


# -----------------------------
# Train or Load
# -----------------------------
def load_or_train(
    data_dir: str = "./data",
    model_path: str = "DNN_LeNet_300_100.pth",
    seed: int = 42,
    test_size: float = 0.2,
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 200,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    force_train: bool = False,
    patience: int = 15,
    min_delta: float = 1e-4,
):
    """
    Returns:
      model, scaler (None), (X_train, X_test, y_train, y_test), timings dict, device

    Early stopping monitors test accuracy every epoch and stops when accuracy
    has not improved by more than min_delta for patience consecutive epochs.
    The best model weights are restored before returning.

    Args:
        patience:   Number of epochs with no improvement before stopping.
        min_delta:  Minimum accuracy gain to count as improvement.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load + split (MNIST comes pre-split into 60k train / 10k test)
    X_train, X_test, y_train, y_test = _load_mnist(data_dir)

    train_ds = NumpyDataset(X_train, y_train)
    test_ds  = NumpyDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Model
    model = MLP(in_features=784, h1=300, h2=100, num_classes=10).to(device)

    # Load if available (unless forced to train)
    timings = {"training_time_s": 0.0, "loaded_from_disk": False}
    if (not force_train) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        timings["loaded_from_disk"] = True
        return model, None, (X_train, X_test, y_train, y_test), timings, device

    # Otherwise train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping state
    best_acc       = 0.0
    best_state     = None
    patience_count = 0
    best_epoch     = 1

    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * Xb.size(0)

        epoch_loss /= len(train_ds)

        # Evaluate every epoch for early stopping
        y_pred = predict(model, test_loader, device)
        mets   = _evaluate(y_test, y_pred)
        acc    = mets["accuracy"]

        if epoch in {1, 5, 10, 25, 50, 100, epochs} or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                f"acc {acc:.4f} f1 {mets['f1']:.4f} "
                f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}"
            )

        # Early stopping check
        if acc > best_acc + min_delta:
            best_acc       = acc
            best_epoch     = epoch
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[EarlyStopping] Stopped at epoch {epoch} — "
                      f"no improvement for {patience} epochs. "
                      f"Best test acc: {best_acc:.4f} (epoch {best_epoch})")
                break

        model.train()

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[EarlyStopping] Restored best weights from epoch {best_epoch} "
              f"(acc={best_acc:.4f})")

    t1 = time.perf_counter()
    timings["training_time_s"] = float(t1 - t0)

    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path} to disk")

    return model, None, (X_train, X_test, y_train, y_test), timings, device


# -----------------------------
# Evaluate full report
# -----------------------------
def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    timings,
    batch_size: int = 64,
    threshold: float = 0.5,
    repeats: int = 500
) -> dict:
    test_ds = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_pred  = predict(model, test_loader, device)
    metrics = _evaluate(y_test, y_pred)

    n_params = count_params(model)
    size_bytes = model_state_size_bytes(model)
    size_kb = size_bytes / 1024.0

    inf = timed_inference(model, X_test, device, repeats=repeats)

    return EvalReport(
        device=device,
        loaded_from_disk=timings["loaded_from_disk"],
        training_time_s=timings["training_time_s"],
        n_test=int(len(test_ds)),
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        params=int(n_params),
        state_size_kb=float(size_kb),
        batch_ms=inf["batch_ms"],
        single_ms=inf["single_ms"],
    )

# -----------------------------
# Fine-tune (post-pruning)
# -----------------------------

#Adam Optimizer Fine Tune
def fine_tune(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> nn.Module:
    
    #Continue training a pruned model to recover accuracy.

    #Unlike load_or_train, this does not reset weights or reload from disk —
    #it picks up exactly where the pruned model left off.

    #Args:
    #    model:        The pruned nn.Module to fine-tune (mutated in place).
    #    X_train:      Training features (flattened, normalised).
    #    y_train:      Training labels.
    #    device:       torch device.
    #    lr:           Learning rate.
    #    epochs:       Number of fine-tuning epochs.
    #    batch_size:   Mini-batch size.
    #    weight_decay: L2 regularisation.
    #    threshold:    Unused (kept for API compatibility).
    #    verbose:      Print loss/metrics each epoch.

    #Returns:
    #    The same model, fine-tuned in place.
    
    if seed is not None:
        set_seed(seed)

    train_ds = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)

        epoch_loss /= len(train_ds)

        if verbose and epoch in {1, 5, epochs}:
            # Reuse existing helpers for a quick progress snapshot
            test_ds = NumpyDataset(X_train, y_train)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            y_pred = predict(model, test_loader, device)
            mets = _evaluate(y_train, y_pred)
            print(
                f"  [fine-tune] Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                f"acc {mets['accuracy']:.4f} f1 {mets['f1']:.4f} "
                f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}"
            )

    model.eval()
    return model

"""

#CosineAnnealingLR Scheduler Fine Tune
def fine_tune(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> nn.Module:

    #Continue training a pruned model to recover accuracy.
    #Unlike load_or_train, this does not reset weights or reload from disk —
    #it picks up exactly where the pruned model left off.
    #Args:
    #    model:        The pruned nn.Module to fine-tune (mutated in place).
    #    X_train:      Training features (flattened, normalised).
    #    y_train:      Training labels.
    #    device:       torch device.
    #    lr:           Learning rate.
    #    epochs:       Number of fine-tuning epochs.
    #    batch_size:   Mini-batch size.
    #    weight_decay: L2 regularisation.
    #    threshold:    Unused (kept for API compatibility).
    #    verbose:      Print loss/metrics each epoch.
    #    seed:         Optional random seed for reproducibility.
    #Returns:
    #    The same model, fine-tuned in place.

    if seed is not None:
        set_seed(seed)

    train_ds = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()

            # Zero gradients for pruned weights before optimizer step
            # so Adam's momentum stats stay clean for dead weights
            for mod in model.modules():
                if isinstance(mod, nn.Linear) and hasattr(mod, "weight_mask"):
                    # When a pruning hook is active, weight is a non-leaf tensor
                    # (weight_orig * weight_mask), so the gradient lives on weight_orig
                    grad_tensor = mod.weight_orig if hasattr(mod, "weight_orig") else mod.weight
                    if grad_tensor.grad is not None:
                        grad_tensor.grad.mul_(mod.weight_mask)

            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)

        scheduler.step()
        epoch_loss /= len(train_ds)

        if verbose and epoch in {1, 5, epochs}:
            test_ds = NumpyDataset(X_train, y_train)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            y_pred = predict(model, test_loader, device)
            mets = _evaluate(y_train, y_pred)
            print(
                f"  [fine-tune] Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                f"acc {mets['accuracy']:.4f} f1 {mets['f1']:.4f} "
                f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}"
            )

    model.eval()
    return model
"""