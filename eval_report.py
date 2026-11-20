import torch

class EvalReport:
    def __init__(
        self,
        device: torch.device,
        loaded_from_disk: bool,
        training_time_s: float,
        n_test: int,
        accuracy: float,
        f1: float,
        precision: float,
        recall: float,
        params: int,
        state_size_kb: float,
        batch_ms: float,
        single_ms: float,
    ):
        self.device = device
        self.loaded_from_disk = loaded_from_disk
        self.training_time_s = training_time_s
        self.n_test = n_test
        self.accuracy = accuracy
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.params = params
        self.state_size_kb = state_size_kb
        self.batch_ms = batch_ms
        self.single_ms = single_ms

    def __str__(self) -> str:
        lines = []
        lines.append(f"Device   : {self.device}")
        lines.append(f"Loaded?  : {self.loaded_from_disk}")
        if not self.loaded_from_disk:
            lines.append(f"Train time: {self.training_time_s:.3f} s")

        lines.append(f"Test size: {self.n_test} samples")
        lines.append(f"Accuracy : {self.accuracy:.4f}")
        lines.append(f"F1       : {self.f1:.4f}")
        lines.append(f"Precision: {self.precision:.4f}")
        lines.append(f"Recall   : {self.recall:.4f}")
        lines.append(f"Params   : {self.params} trainable")
        lines.append(f"State size (approx): {self.state_size_kb:.2f} KB")
        lines.append(
            f"Inference time (avg): batch {self.batch_ms:.4f} ms | "
            f"single {self.single_ms:.6f} ms"
        )

        return "\n".join(lines)