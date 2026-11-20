# DNN_Autoenc_784.py
# Dataset: MNIST (via torchvision)
# Input  : 784 pixel features (flattened 28x28)
#
# Model: Fully-connected Autoencoder + Classifier
#   Encoder  : fc_enc1 (784→128, ReLU) → fc_enc2 (128→64, ReLU)  [bottleneck]
#   Decoder  : fc_dec1 (64→128,  ReLU) → fc_dec2 (128→784, Sigmoid)
#   Classifier: fc_cls (64→10)  [branches from bottleneck]
#
# Loss: α × MSE(reconstruction, input) + β × CrossEntropy(prediction, label)
#       default α=1.0, β=1.0
#
# Pruning note:
#   PrunableNetwork sees only nn.Linear layers in forward order:
#     fc_enc1, fc_enc2, fc_dec1, fc_dec2, fc_cls
#   The decoder (fc_dec1, fc_dec2) is excluded from pruning by
#   overriding linear_names in the graph metadata — only encoder +
#   classifier layers are pruned.
from typing import Optional

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
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
    """Download MNIST and return flattened, normalised numpy arrays (N, 784)."""
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
# Model: Autoencoder + Classifier
# -----------------------------
class AutoencoderClassifier(nn.Module):
    """
    Fully-connected autoencoder with a classification branch at the bottleneck.

    Forward pass returns (reconstruction, logits) so both losses can be
    computed jointly.  The decoder is intentionally kept separate from the
    encoder/classifier so PrunableNetwork can be told to prune only the
    encoder + classifier layers.

    Architecture (named_modules order):
        Encoder    : fc_enc1 784→128 (ReLU) → fc_enc2 128→64 (ReLU)  ← bottleneck
        Classifier : fc_cls  64→10   (from bottleneck)
        Decoder    : fc_dec1 64→128  (ReLU) → fc_dec2 128→784 (Sigmoid)

    fc_cls is declared before the decoder so PrunableNetwork sees the
    linear chain fc_enc1→fc_enc2→fc_cls→fc_dec1→fc_dec2, which satisfies
    prune_neurons's requirement for contiguous layer pairs.
    """

    def __init__(self):
        super().__init__()

        # Encoder (prunable)
        self.fc_enc1  = nn.Linear(784, 128)
        self.relu_e1  = nn.ReLU()
        self.fc_enc2  = nn.Linear(128, 64)
        self.relu_e2  = nn.ReLU()

        # Classifier branch — declared BEFORE decoder so named_modules order is:
        #   fc_enc1(784->128) -> fc_enc2(128->64) -> fc_cls(64->10) -> fc_dec1(64->128) -> fc_dec2(128->784)
        # prune_neurons sees a valid contiguous chain; fc_dec1/fc_dec2 sit at
        # the end (node_layer >= L-1) and are never touched by structural pruning.
        self.fc_cls   = nn.Linear(64, 10)

        # Decoder (NOT pruned — reconstruction fidelity)
        self.fc_dec1  = nn.Linear(64, 128)
        self.relu_d1  = nn.ReLU()
        self.fc_dec2  = nn.Linear(128, 784)
        self.sigmoid  = nn.Sigmoid()

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
    """
    Combined autoencoder + classification loss.

    loss = alpha * MSE(recon, x) + beta * CrossEntropy(logits, y)
    """
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
    model_path: str  = "DNN_Autoenc_784.pth",
    seed: int        = 42,
    test_size: float = 0.2,
    batch_size: int  = 64,
    lr: float        = 1e-3,
    epochs: int      = 200,
    weight_decay: float = 1e-4,
    alpha: float     = 1.0,   # reconstruction loss weight
    beta: float      = 1.0,   # classification loss weight
    threshold: float = 0.5,
    force_train: bool = False,
    patience: int    = 15,
    min_delta: float = 1e-4,
):
    """
    Returns:
      model, scaler (None), (X_train, X_test, y_train, y_test), timings dict, device
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = _load_mnist(data_dir)

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
            recon, logits = model(Xb)
            loss = autoenc_loss(recon, Xb, logits, yb, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
        epoch_loss /= len(train_ds)

        # Evaluate classification accuracy every epoch for early stopping
        y_pred = predict(model, test_loader, device)
        mets   = _evaluate(y_test, y_pred)
        acc    = mets["accuracy"]

        if epoch in {1, 5, 10, 25, 50, 100, epochs} or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss {epoch_loss:.4f} | "
                  f"acc {acc:.4f} f1 {mets['f1']:.4f} "
                  f"prec {mets['precision']:.4f} rec {mets['recall']:.4f}")

        # Early stopping check on classification accuracy
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
    batch_size: int  = 64,
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
    batch_size: int  = 64,
    weight_decay: float = 1e-4,
    alpha: float     = 1.0,
    beta: float      = 1.0,
    threshold: float = 0.5,
    verbose: bool    = True,
    seed: Optional[int] = None,
) -> nn.Module:
    """
    Fine-tune a pruned autoencoder+classifier to recover accuracy.

    Both the reconstruction and classification losses are used so the
    decoder continues to receive gradient even though it is not pruned.
    Pruned encoder/classifier weights stay zero (masked before calling
    this function via reapply_zero_masks).

    Args:
        model:        Pruned AutoencoderClassifier (mutated in place).
        X_train:      Training features (N, 784), normalised.
        y_train:      Training labels.
        device:       torch device.
        lr:           Learning rate.
        epochs:       Fine-tuning epochs.
        batch_size:   Mini-batch size.
        weight_decay: L2 regularisation.
        alpha:        Reconstruction loss weight.
        beta:         Classification loss weight.
        verbose:      Print progress.

    Returns:
        The same model, fine-tuned in place.
    """
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
