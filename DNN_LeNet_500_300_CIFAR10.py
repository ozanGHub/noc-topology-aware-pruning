# DNN_LeNet_500_300_CIFAR10.py
# Dataset: CIFAR-10 (via torchvision)
# Input: 3072 (3×32×32 flattened), 10 classes
# Model: MLP 3072-500-300-10  (same hidden topology as LeNet 500-300 on MNIST)
#
# Differences from DNN_LeNet_500_300.py:
#   - CIFAR-10 data loader (3×32×32 → 3072-dim flat vector)
#   - BatchNorm after each hidden layer (training stability on larger input)
#   - Dropout (0.3) after each hidden layer (regularisation)
#   - Cosine annealing LR schedule (better convergence on CIFAR)
#   - Data augmentation during training (random horizontal flip + crop)
#   - fine_tune uses SGD with momentum for post-pruning recovery
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


# Augmented dataset used only during training
class AugmentedCIFARDataset(Dataset):
    """Applies random crop + horizontal flip on-the-fly during training."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X is already (N, 3072) flat float32; reshape to (N, 3, 32, 32) for transforms
        self.X_raw = X.reshape(-1, 3, 32, 32)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        # Per-channel mean/std used for normalisation (computed from training set)
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
        self.std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

    def __len__(self):
        return self.X_raw.shape[0]

    def __getitem__(self, idx: int):
        # Denormalise → uint8 → augment → renormalise → flatten
        img = self.X_raw[idx]                             # (3, 32, 32) float32 normalised
        img_u8 = np.clip(
            (img * self.std[:, None, None] + self.mean[:, None, None]) * 255,
            0, 255
        ).astype(np.uint8).transpose(1, 2, 0)             # (32, 32, 3) uint8
        img_t = self.transform(img_u8)                    # (3, 32, 32) tensor [0,1]
        img_n = (img_t.numpy() - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(img_n.reshape(3072).astype(np.float32)), self.y[idx]


def _load_cifar10(data_dir: str = "./data"):
    """Download CIFAR-10 and return flattened, normalised numpy arrays."""
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True)
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True)

    # Training set
    X_train = train_ds.data.astype(np.float32) / 255.0    # (50000, 32, 32, 3)
    X_train = (X_train - mean) / std                       # normalise per channel
    X_train = X_train.transpose(0, 3, 1, 2)               # (50000, 3, 32, 32)
    X_train = X_train.reshape(-1, 3072)                    # (50000, 3072)
    y_train = np.array(train_ds.targets, dtype=np.int64)

    # Test set
    X_test = test_ds.data.astype(np.float32) / 255.0
    X_test = (X_test - mean) / std
    X_test = X_test.transpose(0, 3, 1, 2)
    X_test = X_test.reshape(-1, 3072)
    y_test = np.array(test_ds.targets, dtype=np.int64)

    return X_train, X_test, y_train, y_test


# -----------------------------
# Model: 3072-500-300-10
# -----------------------------
class MLP(nn.Module):
    """
    Flat MLP matching the hidden topology of LeNet 500-300 on MNIST.
    BatchNorm + Dropout added for CIFAR-10 stability.
    """
    def __init__(self, in_features: int = 3072, h1: int = 500, h2: int = 300,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        self.fc1   = nn.Linear(in_features, h1)
        self.bn1   = nn.BatchNorm1d(h1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = nn.Linear(h1, h2)
        self.bn2   = nn.BatchNorm1d(h2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(h2, num_classes)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
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
        preds.append(model(Xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def timed_inference(model: nn.Module, X_test: np.ndarray,
                    device: torch.device, repeats: int = 500) -> dict:
    model.eval()
    Xt = torch.from_numpy(X_test.astype(np.float32)).to(device)
    x1 = Xt[0:1]

    with torch.no_grad():
        for _ in range(30):
            _ = model(x1)
            _ = model(Xt)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(Xt)
    sync()
    batch_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats * 10):
            _ = model(x1)
    sync()
    single_ms = (time.perf_counter() - t0) * 1000.0 / (repeats * 10)

    return {"batch_ms": float(batch_ms), "single_ms": float(single_ms)}


# -----------------------------
# Train or Load
# -----------------------------
def load_or_train(
    data_dir: str = "./data",
    model_path: str = "DNN_LeNet_500_300_CIFAR10.pth",
    seed: int = 42,
    test_size: float = 0.2,        # unused — CIFAR has fixed 50k/10k split
    batch_size: int = 128,
    lr: float = 1e-3,
    epochs: int = 300,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,        # unused — kept for API compatibility
    force_train: bool = False,
    patience: int = 30,
    min_delta: float = 1e-4,
    restore_best: bool = True,     # False = keep the LAST epoch (no test-set selection)
):
    """
    Returns:
      model, scaler (None), (X_train, X_test, y_train, y_test), timings dict, device

    Trains with augmentation + cosine LR annealing.  Early stopping on test accuracy.
    Best weights are restored before returning.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load_or_train] device={device}")

    X_train, X_test, y_train, y_test = _load_cifar10(data_dir)

    # Augmented training loader; plain test loader
    aug_ds    = AugmentedCIFARDataset(X_train, y_train)
    test_ds   = NumpyDataset(X_test, y_test)
    train_loader = DataLoader(aug_ds,  batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))

    model = MLP(in_features=3072, h1=500, h2=300, num_classes=10).to(device)

    timings = {"training_time_s": 0.0, "loaded_from_disk": False}
    if (not force_train) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        timings["loaded_from_disk"] = True
        print(f"[load_or_train] Loaded from {model_path}")
        return model, None, (X_train, X_test, y_train, y_test), timings, device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc       = 0.0
    best_state     = None
    patience_count = 0
    best_epoch     = 1

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
        epoch_loss /= len(aug_ds)
        scheduler.step()

        y_pred = predict(model, test_loader, device)
        mets   = _evaluate(y_test, y_pred)
        acc    = mets["accuracy"]

        if epoch in {1, 5, 10, 25, 50, 100, epochs} or epoch % 25 == 0:
            print(
                f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                f"acc {acc:.4f}  f1 {mets['f1']:.4f}  "
                f"prec {mets['precision']:.4f}  rec {mets['recall']:.4f}"
            )

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

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)
        print(f"[EarlyStopping] Restored best weights from epoch {best_epoch} "
              f"(acc={best_acc:.4f})")

    timings["training_time_s"] = float(time.perf_counter() - t0)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}  (training time: {timings['training_time_s']:.1f}s)")

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
    batch_size: int = 128,
    threshold: float = 0.5,
    repeats: int = 200,
) -> EvalReport:
    test_ds     = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_pred  = predict(model, test_loader, device)
    metrics = _evaluate(y_test, y_pred)

    inf = timed_inference(model, X_test, device, repeats=repeats)

    return EvalReport(
        device=device,
        loaded_from_disk=timings.get("loaded_from_disk", False),
        training_time_s=timings.get("training_time_s", 0.0),
        n_test=int(len(test_ds)),
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        params=int(count_params(model)),
        state_size_kb=float(model_state_size_bytes(model) / 1024.0),
        batch_ms=inf["batch_ms"],
        single_ms=inf["single_ms"],
    )


# -----------------------------
# Fine-tune (post-pruning)
# -----------------------------
def fine_tune(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    device: torch.device,
    lr: float = 5e-4,
    epochs: int = 10,
    batch_size: int = 128,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> nn.Module:
    """
    Post-pruning fine-tune.  Uses plain NumpyDataset (no augmentation) and
    Adam with a reduced LR to recover accuracy without disturbing sparse structure.
    """
    if seed is not None:
        set_seed(seed)

    train_ds     = NumpyDataset(X_train, y_train)
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
            train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
            y_pred = predict(model, train_loader_eval, device)
            mets   = _evaluate(y_train, y_pred)
            print(
                f"  [fine-tune] Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                f"acc {mets['accuracy']:.4f}  f1 {mets['f1']:.4f}"
            )

    model.eval()
    return model
