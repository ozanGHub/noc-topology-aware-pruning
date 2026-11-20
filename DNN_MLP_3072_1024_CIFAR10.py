# DNN_MLP_3072_1024_CIFAR10.py
# Dataset: CIFAR-10 (flattened, 32×32×3 = 3072 features)
# Model: MLP 3072-1024-1024-10
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
# Dataset helpers
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

class AugmentedCIFARDataset(Dataset):
    """Training set with random crop + horizontal flip, then flatten."""
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data    = data       # (N, 3, 32, 32) float32 normalised
        self.targets = targets
        self.aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        img = self.aug(self.data[idx])     # (3, 32, 32)
        return img.reshape(-1), self.targets[idx]


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _load_cifar10(data_dir: str = "./data"):
    """Download CIFAR-10 and return flattened, normalised numpy arrays."""
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,
                                 transform=transforms.ToTensor())
    test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,
                                 transform=transforms.ToTensor())

    def _to_arrays(ds):
        loader = DataLoader(ds, batch_size=512, shuffle=False)
        imgs, labels = [], []
        for x, y in loader:
            imgs.append(norm(x))
            labels.append(y)
        imgs   = torch.cat(imgs)    # (N, 3, 32, 32)
        labels = torch.cat(labels)
        return imgs, labels

    train_imgs, train_lbls = _to_arrays(train_ds)
    test_imgs,  test_lbls  = _to_arrays(test_ds)

    X_train = train_imgs.view(len(train_imgs), -1).numpy().astype(np.float32)
    y_train = train_lbls.numpy()
    X_test  = test_imgs.view(len(test_imgs), -1).numpy().astype(np.float32)
    y_test  = test_lbls.numpy()

    return X_train, X_test, y_train, y_test, train_imgs, train_lbls


# -----------------------------
# Model: 3072-1024-1024-10
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_features: int = 3072, h1: int = 1024, h2: int = 1024,
                 num_classes: int = 10):
        super().__init__()
        self.fc1   = nn.Linear(in_features, h1)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(h2, num_classes)

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
        preds.append(model(Xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def timed_inference(model: nn.Module, X_test: np.ndarray, device: torch.device,
                    repeats: int = 500) -> dict:
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
    model_path: str = "DNN_MLP_3072_1024_CIFAR10.pth",
    seed: int = 42,
    test_size: float = 0.2,   # unused — CIFAR-10 has fixed split
    batch_size: int = 128,
    lr: float = 5e-4,
    epochs: int = 300,
    weight_decay: float = 1e-4,
    threshold: float = 0.5,
    force_train: bool = False,
    patience: int = 30,
    min_delta: float = 1e-4,
    restore_best: bool = True,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test, train_imgs, train_lbls = _load_cifar10(data_dir)

    test_ds      = NumpyDataset(X_test, y_test)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MLP(in_features=3072, h1=1024, h2=1024, num_classes=10).to(device)

    timings = {"training_time_s": 0.0, "loaded_from_disk": False}
    if (not force_train) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        timings["loaded_from_disk"] = True
        return model, None, (X_train, X_test, y_train, y_test), timings, device

    train_ds_aug = AugmentedCIFARDataset(train_imgs.to(device), train_lbls)
    train_loader = DataLoader(train_ds_aug, batch_size=batch_size, shuffle=True,
                              num_workers=0)

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
        epoch_loss /= len(train_ds_aug)
        scheduler.step()

        y_pred = predict(model, test_loader, device)
        mets   = _evaluate(y_test, y_pred)
        acc    = mets["accuracy"]

        if epoch in {1, 5, 10, 25, 50, 100, epochs} or epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"acc {acc:.4f} f1 {mets['f1']:.4f} "
                  f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}")

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
    batch_size: int = 128,
    threshold: float = 0.5,
    repeats: int = 500,
) -> EvalReport:
    test_ds     = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_pred  = predict(model, test_loader, device)
    metrics = _evaluate(y_test, y_pred)
    inf     = timed_inference(model, X_test, device, repeats=repeats)

    return EvalReport(
        device=device,
        loaded_from_disk=timings["loaded_from_disk"],
        training_time_s=timings["training_time_s"],
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
    verbose: bool = True,
    seed: Optional[int] = None,
) -> nn.Module:
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
            test_ds     = NumpyDataset(X_train, y_train)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            y_pred = predict(model, test_loader, device)
            mets = _evaluate(y_train, y_pred)
            print(f"  [fine-tune] Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"acc {mets['accuracy']:.4f} f1 {mets['f1']:.4f} "
                  f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}")

    model.eval()
    return model
