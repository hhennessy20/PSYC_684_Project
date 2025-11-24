# Harry Hennessy
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# Random seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ADReSS loading class
class ADRESSSpectrogramDataset(Dataset):
    """
    Loads ADReSS Normalised_audio-chunks and turns them into log-mel spectrograms.

    Expects structure:
        root_dir/
            cc/
                *.wav
            cd/
                *.wav

    Labels:
        cc -> 0 (healthy/control)
        cd -> 1 (AD)

    Speaker IDs:
        Each speaker is defined by the first four characters of the filename stem,
        e.g. 'S001_xxx.wav' -> 'S001'
    """

    def __init__(
        self,
        root_dir: Path,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        duration_sec: float = 8.0,
    ):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.num_samples = int(sample_rate * duration_sec)

        self.label_map = {
            "cc": 0,  # healthy
            "cd": 1,  # alzheimer's
        }

        self.samples: List[Tuple[Path, int]] = []
        self.subject_ids: List[str] = []
        self.labels: List[int] = []

        for label_name, label_id in self.label_map.items():
            class_dir = self.root_dir / label_name
            if not class_dir.is_dir():
                continue
            for wav_path in sorted(class_dir.rglob("*.wav")):
                subj_id = self._get_subject_id_from_path(wav_path)
                self.samples.append((wav_path, label_id))
                self.subject_ids.append(subj_id)
                self.labels.append(label_id)

        if len(self.samples) == 0:
            raise RuntimeError(f"No .wav files found under {self.root_dir}")

        # Convert to Mel Spectrogram
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

    # Split samples by speaker to prevent overlap
    @staticmethod
    def _get_subject_id_from_path(path: Path) -> str:
        stem = path.stem
        if len(stem) < 4:
            raise ValueError(f"Filename too short to contain speaker ID: {path.name}")
        return stem[:4]

    def __len__(self) -> int:
        return len(self.samples)

    # Loads and preprocesses audio
    def _load_audio(self, path: Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(path)
        # Convert to mono audio
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Pad/trim to fixed length
        num_current = waveform.size(1)
        if num_current < self.num_samples:
            pad_amount = self.num_samples - num_current
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_current > self.num_samples:
            start = (num_current - self.num_samples) // 2
            waveform = waveform[:, start:start + self.num_samples]

        return waveform

    # Converts to mel spectrogram
    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]
        waveform = self._load_audio(wav_path)

        # Mel spectrogram
        spec = self.mel_spec(waveform)  # (1, n_mels, T)
        spec = torch.log(spec + 1e-9)

        # Normalize per example
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / (std + 1e-8)

        label_t = torch.tensor(label, dtype=torch.float32)
        return spec, label_t
    
# CNN model definition
class AudioCNN(nn.Module):
    """
    2D CNN for spectrogram classification.
    Input: (batch, 1, n_mels, time)
    Output: logits for binary classification (batch,)
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 16 -> 8
        )

        self.classifier = None
        self._n_mels = n_mels

    def _build_classifier_if_needed(self, x: torch.Tensor):
        if self.classifier is None:
            with torch.no_grad():
                feats = self.features(x)
                feats_flat_dim = feats.shape[1] * feats.shape[2] * feats.shape[3]

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feats_flat_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 1),
            )
            self.classifier.to(x.device)

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._build_classifier_if_needed(x)
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits.view(-1)

# Training
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    spec_augment: nn.Module | None = None,
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        # Augment on CPU before moving to GPU
        if spec_augment is not None:
            specs = spec_augment(specs)

        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * specs.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for specs, labels in loader:
            specs = specs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(specs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * specs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Main
def main():
    set_seed(42)

    # Paths
    train_chunks_root = Path("train") / "Normalised_audio-chunks"

    # Hyperparameters
    sample_rate = 16000
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    duration_sec = 8.0

    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-3
    val_split = 0.2

    # Early stopping
    es_patience = 20
    min_delta = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base dataset
    full_dataset = ADRESSSpectrogramDataset(
        root_dir=train_chunks_root,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        duration_sec=duration_sec,
    )
    print(f"Total chunks in dataset: {len(full_dataset)}")

    # Split by subject
    subject_ids = np.array(full_dataset.subject_ids)
    unique_subjects = np.unique(subject_ids)

    rng = np.random.default_rng(42)
    rng.shuffle(unique_subjects)

    val_subject_count = max(1, int(len(unique_subjects) * val_split))
    val_subjects = set(unique_subjects[:val_subject_count])

    train_indices = [i for i, sid in enumerate(subject_ids) if sid not in val_subjects]
    val_indices = [i for i, sid in enumerate(subject_ids) if sid in val_subjects]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Ensure no subject overlap
    train_subjects = set(subject_ids[train_indices])
    val_subjects_actual = set(subject_ids[val_indices])
    overlap = train_subjects & val_subjects_actual

    print(f"Unique subjects total: {len(unique_subjects)}")
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects_actual)}")
    if overlap:
        raise RuntimeError(f"Subject overlap between train and val: {overlap}")

    print(f"Train chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Augment train set
    spec_augment = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
        torchaudio.transforms.TimeMasking(time_mask_param=20),
    )

    # Model, loss, optimizer, scheduler
    model = AudioCNN(n_mels=n_mels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_path = "best_adress_cnn.pt"

    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, spec_augment=spec_augment
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)

        # Check for improvement
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Track best val acc and best loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved to {best_model_path} "
                f"(val_loss={val_loss:.4f}, val_acc={val_acc:.4f})"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve > 15:
                print(f"  -> No significant improvement for {epochs_no_improve} epoch(s). Stopping soon.")

        # Early stopping
        if epochs_no_improve >= es_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement in val loss for {es_patience} epochs)."
            )
            break

    print(f"Training complete. Best val acc at lowest val loss: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()



