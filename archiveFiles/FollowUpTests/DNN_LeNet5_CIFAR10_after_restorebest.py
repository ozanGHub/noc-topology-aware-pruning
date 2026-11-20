# DNN_LeNet5_CIFAR10.py
# Dataset: CIFAR-10 (32×32×3)
# Architecture: modern LeNet-5
#   conv1(3→6, 5×5) → ReLU → MaxPool(2) → 14×14×6
#   c3  (6→16, 5×5) → ReLU → MaxPool(2) → 5×5×16
#   flatten → 400
#   c5  (400→120)   → ReLU
#   f6  (120→84)    → ReLU
#   output (84→10)
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _load_cifar10_raw(data_dir: str = "./data"):
    """Return (X_train, X_test, y_train, y_test) as 4D tensors [N,3,32,32] float32."""
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
        imgs   = torch.cat(imgs).numpy().astype(np.float32)   # [N,3,32,32]
        labels = torch.cat(labels).numpy()
        return imgs, labels

    X_train, y_train = _to_arrays(train_ds)
    X_test,  y_test  = _to_arrays(test_ds)
    return X_train, X_test, y_train, y_test


class LeNet5(nn.Module):
    """Modern LeNet-5 for CIFAR-10.
    Layer names preserved from the original paper: conv1, c3, c5, f6, output.
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3, 6, kernel_size=5)    # 32→28, 6 filters
        self.pool   = nn.MaxPool2d(2, 2)                 # shared pooling
        self.c3     = nn.Conv2d(6, 16, kernel_size=5)   # 14→10, 16 filters
        self.c5     = nn.Linear(400, 120)                # 5×5×16=400 → 120
        self.f6     = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))   # [N,6,14,14]
        x = self.pool(self.relu(self.c3(x)))      # [N,16,5,5]
        x = x.view(x.size(0), -1)                # [N,400]
        x = self.relu(self.c5(x))                # [N,120]
        x = self.relu(self.f6(x))                # [N,84]
        return self.output(x)                     # [N,10]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def model_state_size_bytes(model: nn.Module) -> int:
    return sum(v.numel() * v.element_size() for v in model.state_dict().values())

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for Xb, _ in loader:
        preds.append(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
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


def load_or_train(
    data_dir:     str   = "./data",
    model_path:   str   = "DNN_LeNet5_CIFAR10.pth",
    seed:         int   = 42,
    test_size:    float = 0.2,
    batch_size:   int   = 128,
    lr:           float = 1e-3,
    epochs:       int   = 200,
    weight_decay: float = 1e-4,
    threshold:    float = 0.5,
    force_train:  bool  = False,
    patience:     int   = 20,
    min_delta:    float = 1e-4,
    restore_best: bool  = True,    # False = keep the LAST epoch (no test-set selection)
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = _load_cifar10_raw(data_dir)

    train_ds     = NumpyDataset(X_train, y_train)
    test_ds      = NumpyDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = LeNet5().to(device)

    timings = {"training_time_s": 0.0, "loaded_from_disk": False}
    if (not force_train) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        timings["loaded_from_disk"] = True
        return model, None, (X_train, X_test, y_train, y_test), timings, device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, best_epoch = 0.0, None, 1
    patience_count = 0

    t0 = time.perf_counter()
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
        scheduler.step()

        y_pred = predict(model, test_loader, device)
        mets   = _evaluate(y_test, y_pred)
        acc    = mets["accuracy"]

        if epoch in {1, 5, 10, 25, 50, 100, epochs} or epoch % 25 == 0:
            print(f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"acc {acc:.4f} f1 {mets['f1']:.4f} "
                  f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}")

        if acc > best_acc + min_delta:
            best_acc, best_epoch, patience_count = acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[EarlyStopping] Stopped at epoch {epoch}. "
                      f"Best acc={best_acc:.4f} (epoch {best_epoch})")
                break
        model.train()

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)
        print(f"[EarlyStopping] Restored epoch {best_epoch} (acc={best_acc:.4f})")

    timings["training_time_s"] = float(time.perf_counter() - t0)
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")
    return model, None, (X_train, X_test, y_train, y_test), timings, device


def evaluate_model(
    model:      nn.Module,
    X_test:     np.ndarray,
    y_test:     np.ndarray,
    device:     torch.device,
    timings,
    batch_size: int   = 128,
    threshold:  float = 0.5,
    repeats:    int   = 500,
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


def fine_tune(
    model:        nn.Module,
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    *,
    device:       torch.device,
    lr:           float = 1e-3,
    epochs:       int   = 10,
    batch_size:   int   = 128,
    weight_decay: float = 1e-4,
    threshold:    float = 0.5,
    verbose:      bool  = True,
    seed: Optional[int] = None,
) -> nn.Module:
    if seed is not None:
        set_seed(seed)

    train_ds     = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
            mets   = _evaluate(y_train, y_pred)
            print(f"  [fine-tune] Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"acc {mets['accuracy']:.4f} f1 {mets['f1']:.4f}")
    model.eval()
    return model
