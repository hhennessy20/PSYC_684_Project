# Harry Hennessy
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

# Add src directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import ADRESS_DIARIZED_DIR, RANDOM_SEED, MODEL_CKPT_PATH, MODELS_DIR


# Seed
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Shared constants
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
DURATION_SEC = 8.0


# Base dataset: one random window per file
class ADRESSSpectrogramDataset(Dataset):
    # expects:
    #   root/cc/*.wav
    #   root/cd/*.wav
    def __init__(
        self,
        root_dir: Path,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        duration_sec: float = DURATION_SEC,
    ):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.num_samples = int(sample_rate * duration_sec)

        self.label_map = {"cc": 0, "cd": 1}

        self.samples: List[Tuple[Path, int]] = []
        self.subject_ids: List[str] = []
        self.labels: List[int] = []

        for label_name, label_id in self.label_map.items():
            d = self.root_dir / label_name
            if not d.is_dir():
                continue
            for wav in sorted(d.rglob("*.wav")):
                sid = wav.stem[:4]
                self.samples.append((wav, label_id))
                self.subject_ids.append(sid)
                self.labels.append(label_id)

        if len(self.samples) == 0:
            raise RuntimeError(f"No .wav files found under {self.root_dir}")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

    def __len__(self) -> int:
        return len(self.samples)

    # random crop loader
    def _load_audio_random(self, path: Path) -> torch.Tensor:
        audio_np, sr = sf.read(str(path), dtype='float32')
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.T)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        n = waveform.size(1)
        if n < self.num_samples:
            pad = self.num_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif n > self.num_samples:
            start = random.randint(0, n - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]

        return waveform

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        w = self._load_audio_random(path)
        spec = self.mel(w)
        spec = torch.log(spec + 1e-9)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec, torch.tensor(label, dtype=torch.float32)


# repeats each file K times per epoch (train only)
class RepeatedDataset(Dataset):
    def __init__(self, base_ds: Dataset, repeats_per_file: int = 4):
        self.base = base_ds
        self.repeats = repeats_per_file

    def __len__(self) -> int:
        return len(self.base) * self.repeats

    def __getitem__(self, idx: int):
        return self.base[idx // self.repeats]


# deterministic val chunks per file
class ValChunksDataset(Dataset):
    # creates fixed windows for each file (same every epoch)
    def __init__(
        self,
        full_dataset: ADRESSSpectrogramDataset,
        indices: List[int],
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        duration_sec: float = DURATION_SEC,
        chunks_per_file: int = 4,
    ):
        self.full_dataset = full_dataset
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration_sec)
        self.chunks_per_file = chunks_per_file

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )

        self.entries: List[Tuple[int, int]] = []

        for ds_idx in indices:
            path, _ = self.full_dataset.samples[ds_idx]
            audio_np, sr = sf.read(str(path), dtype='float32')
            if audio_np.ndim == 1:
                waveform = torch.from_numpy(audio_np).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_np.T)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

            n = waveform.size(1)
            if n <= self.num_samples:
                self.entries.append((ds_idx, 0))
                continue

            max_start = n - self.num_samples
            if chunks_per_file == 1:
                start = max_start // 2
                self.entries.append((ds_idx, start))
            else:
                for j in range(chunks_per_file):
                    frac = j / (chunks_per_file - 1)
                    start = int(round(frac * max_start))
                    self.entries.append((ds_idx, start))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        ds_idx, start = self.entries[idx]
        path, label = self.full_dataset.samples[ds_idx]

        audio_np, sr = sf.read(str(path), dtype='float32')
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.T)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        n = waveform.size(1)
        if n < self.num_samples:
            pad = self.num_samples - n
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            start = min(start, n - self.num_samples)
            waveform = waveform[:, start:start + self.num_samples]

        spec = self.mel(waveform)
        spec = torch.log(spec + 1e-9)
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

        return spec, torch.tensor(label, dtype=torch.float32)


# model
class AudioCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = None

    def _init_fc(self, x: torch.Tensor):
        with torch.no_grad():
            f = self.features(x)
            d = f.shape[1] * f.shape[2] * f.shape[3]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        ).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.classifier is None:
            self._init_fc(x)
        f = self.features(x)
        out = self.classifier(f)
        return out.view(-1)


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


def main():
    set_seed(RANDOM_SEED)

    train_root = ADRESS_DIARIZED_DIR

    batch_size = 32
    num_epochs = 200
    learning_rate = 1e-3
    val_split = 0.2

    train_repeats_per_file = 4
    val_chunks_per_file = 4

    es_patience = 20
    min_delta = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = ADRESSSpectrogramDataset(train_root)
    print(f"Total recordings in dataset: {len(full_dataset)}")

    subject_ids = np.array(full_dataset.subject_ids)
    unique_subjects = np.unique(subject_ids)

    rng = np.random.default_rng(42)
    rng.shuffle(unique_subjects)

    val_subject_count = max(1, int(len(unique_subjects) * val_split))
    val_subjects = set(unique_subjects[:val_subject_count])

    train_indices = [i for i, sid in enumerate(subject_ids) if sid not in val_subjects]
    val_indices = [i for i, sid in enumerate(subject_ids) if sid in val_subjects]

    train_subset = Subset(full_dataset, train_indices)

    train_subjects = set(subject_ids[train_indices])
    val_subjects_actual = set(subject_ids[val_indices])
    overlap = train_subjects & val_subjects_actual

    print(f"Unique subjects total: {len(unique_subjects)}")
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects_actual)}")
    if overlap:
        raise RuntimeError(f"Subject overlap between train and val: {overlap}")

    train_dataset = RepeatedDataset(
        train_subset,
        repeats_per_file=train_repeats_per_file,
    )

    val_dataset = ValChunksDataset(
        full_dataset=full_dataset,
        indices=val_indices,
        chunks_per_file=val_chunks_per_file,
    )

    print(
        f"Train windows per epoch: {len(train_dataset)}, "
        f"Val windows: {len(val_dataset)}"
    )

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

    spec_augment = nn.Sequential(
        torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
        torchaudio.transforms.TimeMasking(time_mask_param=20),
    )

    model = AudioCNN(n_mels=N_MELS).to(device)
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODEL_CKPT_PATH
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

        loss_improved = val_loss + min_delta < best_val_loss
        acc_improved = val_acc > best_val_acc + 1e-6

        if loss_improved:
            best_val_loss = val_loss
        if acc_improved:
            best_val_acc = val_acc

        save_ckpt = False
        if acc_improved:
            save_ckpt = True
        elif not acc_improved and abs(val_acc - best_val_acc) <= 1e-6 and loss_improved:
            save_ckpt = True

        if save_ckpt:
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved to {best_model_path} "
                f"(val_loss={val_loss:.4f}, val_acc={val_acc:.4f})"
            )

        if loss_improved or acc_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve > es_patience:
                print(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement in val loss or acc for {es_patience} epochs)."
                )
                break

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()



