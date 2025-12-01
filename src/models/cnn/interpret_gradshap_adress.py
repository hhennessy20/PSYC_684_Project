# Harry Hennessy
import os
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import GradientShap
import torchaudio

from train_adress_cnn import (
    AudioCNN,
    set_seed,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    DURATION_SEC,
)

# Crop zero-padded/silent time frames for visualization.
def crop_to_speech(spec_np, attr_np, eps=1e-4):
    pad_val = spec_np.min()
    mask = (spec_np > pad_val + eps).any(axis=0)
    idxs = np.nonzero(mask)[0]
    if len(idxs) == 0:
        return spec_np, attr_np

    last = idxs[-1]
    return spec_np[:, : last + 1], attr_np[:, : last + 1]

#  Wraps AudioCNN for Captum (shape (B, 1)).
class WrappedModel(torch.nn.Module):

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        return out.unsqueeze(1)

# Compute GradSHAP attributions for a batch of specs.
def compute_gradshap(model: AudioCNN, batch_specs: torch.Tensor, device: torch.device, n_baseline: int = 2, n_samples: int = 8,
    stdevs: float = 0.09,) -> torch.Tensor:

    wrapped = WrappedModel(model).to(device)
    wrapped.eval()

    gradshap = GradientShap(wrapped)

    # Account for noise
    baseline = torch.zeros_like(batch_specs[0:1]).repeat(n_baseline, 1, 1, 1)
    baseline += 0.01 * torch.randn_like(baseline)

    attributions = gradshap.attribute(
        batch_specs.to(device),
        baselines=baseline.to(device),
        target=0,
        n_samples=n_samples,
        stdevs=stdevs,
    )
    return attributions.detach().cpu()


def make_mel_transform(sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS,):

    return torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        center=True,power=2.0,)

# Full-recording log-mel spectrogram
def full_logmel_from_wav(wav_path: Path, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS,) -> torch.Tensor:

    waveform, sr = torchaudio.load(str(wav_path))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    mel_transform = make_mel_transform(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    with torch.no_grad():
        spec = mel_transform(waveform)
        spec = torch.log(spec + 1e-9)

    return spec

# Number of spectrogram frames for one training window.
def get_window_num_frames(sample_rate: int = SAMPLE_RATE, duration_sec: float = DURATION_SEC, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,) -> int:

    num_samples = int(sample_rate * duration_sec)
    dummy_wave = torch.zeros(1, num_samples)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )
    with torch.no_grad():
        spec = mel_transform(dummy_wave)
        spec = torch.log(spec + 1e-9)

    return spec.shape[-1]

# Full-file GradSHAP saliency using sliding windows.
def compute_full_file_saliency(model: AudioCNN, wav_path: Path, device: torch.device, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH, n_mels: int = N_MELS, duration_sec: float = DURATION_SEC, hop_sec: float = 2.0, gradshap_n_baseline: int = 2,
    gradshap_n_samples: int = 8, gradshap_stdevs: float = 0.09,) -> Tuple[np.ndarray, np.ndarray, float]:

    spec_full = full_logmel_from_wav(wav_path=wav_path, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,)

    # Normalize full spec for PDSM input
    full_mean = spec_full.mean()
    full_std = spec_full.std()
    spec_full_norm = (spec_full - full_mean) / (full_std + 1e-8)

    _, _, T_full = spec_full.shape

    # Window/hop in spectrogram frames
    window_frames = get_window_num_frames(
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    hop_frames = max(1, int(window_frames * (hop_sec / duration_sec)))

    window_specs = []
    starts = []

    t = 0
    while t < T_full:
        t_start = t
        t_end = t_start + window_frames

        if t_end <= T_full:
            spec_win = spec_full[:, :, t_start:t_end].clone()
            num_real_frames = window_frames
        else:
            num_real_frames = T_full - t_start
            spec_win = torch.full(
                (1, n_mels, window_frames),
                fill_value=spec_full.min().item(),
                dtype=spec_full.dtype,
            )
            spec_win[:, :, :num_real_frames] = spec_full[:, :, t_start:T_full]

        w_mean = spec_win.mean()
        w_std = spec_win.std()
        spec_win_norm = (spec_win - w_mean) / (w_std + 1e-8)

        window_specs.append(spec_win_norm)
        starts.append((t_start, num_real_frames))

        if t_end >= T_full:
            break
        t += hop_frames

    if len(window_specs) == 0:
        spec_np = spec_full_norm.squeeze(0).cpu().numpy()
        M_zero = np.zeros_like(spec_np)
        return spec_np, M_zero, 0.5

    window_specs_tensor = torch.stack(window_specs, dim=0)

    # Window-level probabilities
    model.eval()
    with torch.no_grad():
        logits = model(window_specs_tensor.to(device))
        probs = torch.sigmoid(logits)
        p_ad_recording = float(probs.mean().item())

    # Window-level attributions
    max_windows_per_batch = 4
    attrs_list = []
    num_windows = window_specs_tensor.size(0)

    for i in range(0, num_windows, max_windows_per_batch):
        batch = window_specs_tensor[i : i + max_windows_per_batch]
        attrs_chunk = compute_gradshap(
            model=model,
            batch_specs=batch,
            device=device,
            n_baseline=gradshap_n_baseline,
            n_samples=gradshap_n_samples,
            stdevs=gradshap_stdevs,
        )
        attrs_list.append(attrs_chunk)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attrs_tensor = torch.cat(attrs_list, dim=0)

    # Overlap-add
    M_full = torch.zeros(n_mels, T_full, dtype=torch.float32)
    counts = torch.zeros(T_full, dtype=torch.float32)

    for win_idx, (t_start, num_real_frames) in enumerate(starts):
        t_end_real = t_start + num_real_frames
        attr_win = attrs_tensor[win_idx, 0, :, :num_real_frames]

        M_full[:, t_start:t_end_real] += attr_win
        counts[t_start:t_end_real] += 1.0

    counts = torch.clamp(counts, min=1.0)
    M_full = M_full / counts.unsqueeze(0)

    spec_full_norm_np = spec_full_norm.squeeze(0).cpu().numpy()
    M_full_np = M_full.cpu().numpy()

    return spec_full_norm_np, M_full_np, p_ad_recording

# Plot spectrogram and saliency map
def plot_pair(spec: np.ndarray, attr: np.ndarray, label: int, prob: float, outpath: Path,):
    spec_np = np.asarray(spec)
    attr_np = np.asarray(attr)

    spec_np, attr_np = crop_to_speech(spec_np, attr_np)

    # time axis in seconds
    hop_sec = HOP_LENGTH / SAMPLE_RATE
    t_spec = np.arange(spec_np.shape[1]) * hop_sec
    t_attr = np.arange(attr_np.shape[1]) * hop_sec

    vmax = np.percentile(np.abs(attr_np), 99)
    if vmax <= 0:
        vmax = 1e-6
    vmin = -vmax

    label_name = "CN" if label == 0 else "AD"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im0 = axes[0].imshow(
        spec_np,
        origin="lower",
        aspect="auto",
        extent=[t_spec[0], t_spec[-1], 0, spec_np.shape[0]],
    )
    axes[0].set_title("Log-mel spectrogram")
    axes[0].set_xlabel("Seconds")
    axes[0].set_ylabel("Mel bins")
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Log-mel power")

    im1 = axes[1].imshow(
        attr_np,
        origin="lower",
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        extent=[t_attr[0], t_attr[-1], 0, attr_np.shape[0]],
    )
    axes[1].set_title(f"GradSHAP (label={label_name}, p(AD)={prob:.2f})")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Mel bins")
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Attribution (red=↑AD, blue=↓AD)")

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


# List all diarized recordings and labels.
def _enumerate_full_recordings(root: Path,) -> Tuple[List[Path], List[int], List[str]]:
    label_map = {"cc": 0, "cd": 1}

    paths: List[Path] = []
    labels: List[int] = []
    sids: List[str] = []

    for label_name, lbl in label_map.items():
        class_dir = root / label_name
        if not class_dir.is_dir():
            continue
        for wav_path in sorted(class_dir.rglob("*.wav")):
            stem = wav_path.stem
            if len(stem) < 4:
                raise ValueError(f"Filename too short to contain speaker ID: {wav_path.name}")
            sid = stem[:4]

            paths.append(wav_path)
            labels.append(lbl)
            sids.append(sid)

    if not paths:
        raise RuntimeError(f"No .wav files found under {root}")

    return paths, labels, sids

#Returns a list of dicts for the top-K least/most AD-confident validation recordings.
def get_saliency_maps_for_pdsm(
    train_root: str | Path = "train/Diarized_full_wave_enhanced_audio",
    model_ckpt: str | Path = "best_adress_cnn.pt",
    val_split: float = 0.2,
    k_most_confident: int = 3,
    k_least_confident: int = 3,
    hop_sec: float = 2.0,
    device: torch.device | None = None,
) -> List[Dict]:
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = Path(train_root)

    paths, labels, sids = _enumerate_full_recordings(train_root)
    sids = np.array(sids)
    labels = np.array(labels)

    unique_sids = np.unique(sids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_sids)

    val_n = max(1, int(len(unique_sids) * val_split))
    val_sid_set = set(unique_sids[:val_n])

    val_indices = [i for i, sid in enumerate(sids) if sid in val_sid_set]

    print(
        f"Total recordings: {len(paths)} | "
        f"Val subjects: {len(val_sid_set)} | "
        f"Val recordings: {len(val_indices)}"
    )

    if not val_indices:
        return []

    # Construct classifier
    model = AudioCNN(n_mels=N_MELS).to(device)

    # Dummy forward to build classifier head
    dummy_T_win = get_window_num_frames()
    dummy_spec = torch.zeros(1, 1, N_MELS, dummy_T_win).to(device)
    with torch.no_grad():
        _ = model(dummy_spec)

    # Load weights
    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {model_ckpt}")

    # Full-file saliency for validation recordings
    full_results = []
    for idx in val_indices:
        wav_path = paths[idx]
        label = int(labels[idx])
        sid = str(sids[idx])

        print(f"Processing {wav_path.name} (subject={sid})...")
        spec_full_np, M_full_np, p_ad = compute_full_file_saliency(
            model=model,
            wav_path=wav_path,
            device=device,
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            duration_sec=DURATION_SEC,
            hop_sec=hop_sec,
        )

        full_results.append(
            {
                "M": M_full_np,
                "spec": spec_full_np,
                "label": label,
                "p_ad": p_ad,
                "path": str(wav_path),
                "subject": sid,
            }
        )

    if not full_results:
        return []

    # Sort by p(AD)
    full_results_sorted = sorted(full_results, key=lambda r: r["p_ad"])
    least_confident = full_results_sorted[:k_least_confident]
    most_confident = (
        full_results_sorted[-k_most_confident:] if k_most_confident > 0 else []
    )

    ranked = []
    for r in least_confident:
        r_copy = dict(r)
        r_copy["rank_type"] = "least_confident"
        ranked.append(r_copy)

    for r in reversed(most_confident):
        r_copy = dict(r)
        r_copy["rank_type"] = "most_confident"
        ranked.append(r_copy)

    return ranked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_root",
        default="train/Diarized_full_wave_enhanced_audio",
        help="Root directory for diarized full-wave audio.",
    )
    parser.add_argument(
        "--model_ckpt",
        default="best_adress_cnn.pt",
        help="Path to trained AudioCNN checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        default="gradshap_val_plots",
        help="Directory to save GradSHAP plots.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation subject fraction.",
    )
    parser.add_argument(
        "--k_most_confident",
        type=int,
        default=3,
        help="Number of most AD-confident recordings.",
    )
    parser.add_argument(
        "--k_least_confident",
        type=int,
        default=3,
        help="Number of least AD-confident recordings.",
    )
    parser.add_argument(
        "--hop_sec",
        type=float,
        default=2.0,
        help="Sliding-window hop in seconds.",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = get_saliency_maps_for_pdsm(
        train_root=args.train_root,
        model_ckpt=args.model_ckpt,
        val_split=args.val_split,
        k_most_confident=args.k_most_confident,
        k_least_confident=args.k_least_confident,
        hop_sec=args.hop_sec,
        device=device,
    )

    if not results:
        print("No saliency maps produced.")
        return

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save plots
    for i, r in enumerate(results):
        spec_np = r["spec"]
        M_np = r["M"]
        label = r["label"]
        p_ad = r["p_ad"]
        path = Path(r["path"])
        rank_type = r["rank_type"]
        sid = r["subject"]

        fname = (
            f"{i:02d}_{rank_type}_sub{sid}_"
            f"{path.stem}_pAD{p_ad:.2f}_lbl{label}.png"
        )
        outpath = outdir / fname
        plot_pair(spec_np, M_np, label, p_ad, outpath)
        print(f"saved: {outpath}")

    print(f"Done. GradSHAP plots saved under: {outdir}")


if __name__ == "__main__":
    main()
