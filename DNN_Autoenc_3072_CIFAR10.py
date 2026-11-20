# DNN_Autoenc_3072_CIFAR10.py
# Dataset: CIFAR-10 (flattened, 32×32×3 = 3072 features)
#
# Model: Fully-connected Autoencoder + Classifier
#   Encoder  : fc_enc1 (3072→512, ReLU) → fc_enc2 (512→128, ReLU)  [bottleneck]
#   Decoder  : fc_dec1 (128→512,  ReLU) → fc_dec2 (512→3072, Sigmoid)
#   Classifier: fc_cls (128→10)  [branches from bottleneck]
#
# Loss: α × MSE(reconstruction, input) + β × CrossEntropy(prediction, label)
#       default α=1.0, β=1.0
#
# Pruning note:
#   PrunableNetwork sees only the EncoderClassifier wrapper:
#     fc_enc1 (3072→512) → fc_enc2 (512→128) → fc_cls (128→10)
#   The decoder (fc_dec1, fc_dec2) is excluded from pruning.
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


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _load_cifar10(data_dir: str = "./data"):
    """Download CIFAR-10 and return flattened, normalised numpy arrays (N, 3072)."""
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
        imgs   = torch.cat(imgs)
        labels = torch.cat(labels)
        X = imgs.view(len(imgs), -1).numpy().astype(np.float32)
        y = labels.numpy()
        return X, y

    X_train, y_train = _to_arrays(train_ds)
    X_test,  y_test  = _to_arrays(test_ds)
    return X_train, X_test, y_train, y_test


# -----------------------------
# Model: Autoencoder + Classifier
# -----------------------------
class AutoencoderClassifier(nn.Module):
    """
    Fully-connected autoencoder with a classification branch at the bottleneck.

    Architecture:
        Encoder    : fc_enc1 3072→512 (ReLU) → fc_enc2 512→128 (ReLU)  ← bottleneck
        Classifier : fc_cls  128→10   (from bottleneck, declared before decoder)
        Decoder    : fc_dec1 128→512  (ReLU) → fc_dec2 512→3072 (Sigmoid)

    fc_cls is declared before the decoder so PrunableNetwork sees:
      fc_enc1 → fc_enc2 → fc_cls → fc_dec1 → fc_dec2
    """

    def __init__(self):
        super().__init__()

        # Encoder (prunable)
        self.fc_enc1 = nn.Linear(3072, 512)
        self.relu_e1 = nn.ReLU()
        self.fc_enc2 = nn.Linear(512, 128)
        self.relu_e2 = nn.ReLU()

        # Classifier — declared BEFORE decoder
        self.fc_cls  = nn.Linear(128, 10)

        # Decoder (NOT pruned)
        self.fc_dec1 = nn.Linear(128, 512)
        self.relu_d1 = nn.ReLU()
        self.fc_dec2 = nn.Linear(512, 3072)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_e2(self.fc_enc2(self.relu_e1(self.fc_enc1(x))))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc_dec2(self.relu_d1(self.fc_dec1(z))))

    def forward(self, x: torch.Tensor):
        z      = self.encode(x)
        recon  = self.decode(z)
        logits = self.fc_cls(z)
        return recon, logits


# -----------------------------
# Combined loss
# -----------------------------
def autoenc_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    logits: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    beta: float  = 1.0,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(recon, x)
    ce  = nn.functional.cross_entropy(logits, y)
    return alpha * mse + beta * ce


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
        _, logits = model(Xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
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
    data_dir: str    = "./data",
    model_path: str  = "DNN_Autoenc_3072_CIFAR10.pth",
    seed: int        = 42,
    test_size: float = 0.2,   # unused — CIFAR-10 has fixed split
    batch_size: int  = 128,
    lr: float        = 1e-3,
    epochs: int      = 200,
    weight_decay: float = 1e-4,
    alpha: float     = 1.0,
    beta: float      = 1.0,
    threshold: float = 0.5,
    force_train: bool = False,
    patience: int    = 20,
    min_delta: float = 1e-4,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = _load_cifar10(data_dir)

    train_ds     = NumpyDataset(X_train, y_train)
    test_ds      = NumpyDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model = AutoencoderClassifier().to(device)

    timings = {"training_time_s": 0.0, "loaded_from_disk": False}
    if (not force_train) and os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        timings["loaded_from_disk"] = True
        return model, None, (X_train, X_test, y_train, y_test), timings, device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            recon, logits = model(Xb)
            loss = autoenc_loss(recon, Xb, logits, yb, alpha=alpha, beta=beta)
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

    if best_state is not None:
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
    batch_size: int  = 128,
    threshold: float = 0.5,
    repeats: int     = 500,
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
    lr: float        = 1e-3,
    epochs: int      = 10,
    batch_size: int  = 128,
    weight_decay: float = 1e-4,
    alpha: float     = 1.0,
    beta: float      = 1.0,
    threshold: float = 0.5,
    verbose: bool    = True,
    seed: Optional[int] = None,
) -> nn.Module:
    if seed is not None:
        set_seed(seed)

    train_ds     = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, logits = model(Xb)
            loss = autoenc_loss(recon, Xb, logits, yb, alpha=alpha, beta=beta)
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
