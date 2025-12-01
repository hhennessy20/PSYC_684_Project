# Harry Hennessy
import os
from pathlib import Path
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from captum.attr import GradientShap

from train_adress_cnn import ADRESSSpectrogramDataset, AudioCNN, set_seed


def crop_to_speech(spec_np, attr_np, eps=1e-4):
    """
    Crop zero-padded/silent time frames from visualization based on padding floor.

    spec_np: (n_mels, T)
    attr_np: (n_mels, T)
    """
    pad_val = spec_np.min()

    mask = (spec_np > pad_val + eps).any(axis=0)
    idxs = np.nonzero(mask)[0]
    if len(idxs) == 0:
        return spec_np, attr_np

    last = idxs[-1]
    return spec_np[:, : last + 1], attr_np[:, : last + 1]


class WrappedModel(torch.nn.Module):
    """
    Wraps AudioCNN to shape (B, 1)
    for Captum's expected interface.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)  # (B,)
        return out.unsqueeze(1)  # (B, 1)


def compute_gradshap(model, batch_specs, device,
                     n_baseline=8, n_samples=32, stdevs=0.09):
    """
    Computes GradSHAP attributions.

    batch_specs: (B, 1, n_mels, T)
    Returns attributions: (B, 1, n_mels, T) on CPU.
    """
    wrapped = WrappedModel(model).to(device)
    wrapped.eval()

    gradshap = GradientShap(wrapped)

    # Account for noise in baselines
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


def plot_pair(spec, attr, label, prob, outpath):
    """
    Plots with color map for AD vs. CN.
    spec:  (1, n_mels, T) tensor
    attr:  (1, n_mels, T) tensor
    label: 0 or 1 (CN or AD)
    prob:  p(AD) from sigmoid
    """
    spec_np = spec.squeeze(0).numpy()
    attr_np = attr.squeeze(0).numpy()

    # Crop padded frames & silence (for visualization only)
    spec_np, attr_np = crop_to_speech(spec_np, attr_np)

    # Normalization for colormap
    vmax = np.percentile(np.abs(attr_np), 99)
    if vmax <= 0:
        vmax = 1e-6
    vmin = -vmax

    label_name = "CN" if label == 0 else "AD"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Input spectrogram
    im0 = axes[0].imshow(spec_np, origin="lower", aspect="auto")
    axes[0].set_title("Input log-mel spectrogram")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Mel bins")
    fig.colorbar(im0, ax=axes[0])

    # GradSHAP
    im1 = axes[1].imshow(
        attr_np,
        origin="lower",
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"GradSHAP (label={label_name}, p(AD)={prob:.2f})")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Mel bins")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath, dpi=200)
    plt.close(fig)



def get_saliency_maps_for_pdsm(
    train_root: str | Path = "train/Normalised_audio-chunks",
    model_ckpt: str | Path = "best_adress_cnn.pt",
    val_split: float = 0.2,
    batch_size: int = 32,
    num_examples: int = 4,
    device: torch.device | None = None,
):
    """
    Function to obtain GradSHAP saliency maps M for
    correctly classified validation examples, suitable as input for PDSM

    returns results : list[dict]
        Each dict has:
            'M'      : np.ndarray of shape (n_mels, T)
                       GradSHAP saliency map for this example.
            'spec'   : np.ndarray of shape (n_mels, T)
                       normalized log-mel spectrogram used as input.
            'label'  : int
                       0 = CN (control), 1 = AD.
            'p_ad'   : float
                       model's predicted P(AD) for this chunk.
    """
    set_seed(42)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = Path(train_root)

    # Rebuild validation set exactly as in main()
    ds = ADRESSSpectrogramDataset(train_root)
    subject_ids = np.array(ds.subject_ids)
    unique = np.unique(subject_ids)

    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    val_n = max(1, int(len(unique) * val_split))
    val_subs = set(unique[:val_n])

    val_inds = [i for i, sid in enumerate(subject_ids) if sid in val_subs]
    val_set = Subset(ds, val_inds)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Construct classifier and load weights
    model = AudioCNN(n_mels=64).to(device)
    with torch.no_grad():
        dummy_spec, _ = ds[0]
        dummy_spec = dummy_spec.unsqueeze(0).to(device)
        _ = model(dummy_spec)

    state_dict = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Collect correctly classified examples from val set
    correct_examples = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            for i in range(specs.size(0)):
                label = float(labels[i].item())
                pred = float(preds[i].item())
                p_ad = float(probs[i].item())

                if pred == label:
                    spec_cpu = specs[i].detach().cpu()
                    label_int = int(label)
                    correct_examples.append((spec_cpu, label_int, p_ad))

    if not correct_examples:
        return []

    # Split into CN and AD and pick top-K by confidence
    cn_examples = [ex for ex in correct_examples if ex[1] == 0]
    ad_examples = [ex for ex in correct_examples if ex[1] == 1]

    if len(cn_examples) == 0 and len(ad_examples) == 0:
        return []

    cn_sorted = sorted(cn_examples, key=lambda ex: ex[2])
    ad_sorted = sorted(ad_examples, key=lambda ex: ex[2], reverse=True)

    k = num_examples
    cn_top = cn_sorted[:k] if cn_sorted else []
    ad_top = ad_sorted[:k] if ad_sorted else []

    all_selected = cn_top + ad_top
    if not all_selected:
        return []

    specs_tensor = torch.stack([ex[0] for ex in all_selected], dim=0)
    attrs_tensor = compute_gradshap(model, specs_tensor, device)

    # Build result list with raw M (no cropping, no abs, no thresholding)
    results = []
    for spec, attr, (_, label_int, p_ad) in zip(
        specs_tensor, attrs_tensor, all_selected
    ):
        spec_np = spec.squeeze(0).cpu().numpy()   # (n_mels, T)
        M_np = attr.squeeze(0).cpu().numpy()      # (n_mels, T)

        results.append(
            {
                "M": M_np,
                "spec": spec_np,
                "label": int(label_int),
                "p_ad": float(p_ad),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_root",
        default="train/Normalised_audio-chunks",
        help="Root directory for training chunks.",
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
        "--num_examples",
        type=int,
        default=4,
        help="Top-K per class.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation subject fraction (matches training split seed).",
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Rebuild val set
    ds = ADRESSSpectrogramDataset(Path(args.train_root))
    subject_ids = np.array(ds.subject_ids)
    unique = np.unique(subject_ids)

    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    val_n = max(1, int(len(unique) * args.val_split))
    val_subs = set(unique[:val_n])

    val_inds = [i for i, sid in enumerate(subject_ids) if sid in val_subs]
    val_set = Subset(ds, val_inds)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Total chunks in full train dataset: {len(ds)} | "
        f"Val subjects: {len(val_subs)} | Val chunks: {len(val_set)}"
    )

    # Construct classifier
    model = AudioCNN(n_mels=64).to(device)
    with torch.no_grad():
        dummy_spec, _ = ds[0]
        dummy_spec = dummy_spec.unsqueeze(0).to(device)
        _ = model(dummy_spec)

    # Load weights
    state_dict = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.model_ckpt}")

    # Collect correct examples from val set
    correct_examples = []

    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            for i in range(specs.size(0)):
                label = float(labels[i].item())
                pred = float(preds[i].item())
                p_ad = float(probs[i].item())

                if pred == label:
                    spec_cpu = specs[i].detach().cpu()
                    label_int = int(label)
                    correct_examples.append((spec_cpu, label_int, p_ad))

    print(f"Total correctly classified val examples: {len(correct_examples)}")

    if not correct_examples:
        print("No correctly classified examples found.")
        return

    # Split into CN and AD and pick top-K by confidence
    cn_examples = [ex for ex in correct_examples if ex[1] == 0]
    ad_examples = [ex for ex in correct_examples if ex[1] == 1]

    if len(cn_examples) == 0 or len(ad_examples) == 0:
        print("Not enough correct examples from both classes.")
        return

    cn_sorted = sorted(cn_examples, key=lambda ex: ex[2])
    ad_sorted = sorted(ad_examples, key=lambda ex: ex[2], reverse=True)

    k = args.num_examples
    cn_top = cn_sorted[:k]
    ad_top = ad_sorted[:k]

    print(f"Selected {len(cn_top)} CN and {len(ad_top)} AD examples.")

    # Compute GradSHAP attributions for all selected examples
    all_selected = cn_top + ad_top
    specs_tensor = torch.stack([ex[0] for ex in all_selected], dim=0)
    attrs_tensor = compute_gradshap(model, specs_tensor, device)

    # Save plots
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    cn_idx = 0
    ad_idx = 0

    for spec, attr, (_, label_int, p_ad) in zip(
        specs_tensor, attrs_tensor, all_selected
    ):
        if label_int == 0:
            fname = f"cn_example_{cn_idx:02d}_pAD{p_ad:.2f}.png"
            cn_idx += 1
        else:
            fname = f"ad_example_{ad_idx:02d}_pAD{p_ad:.2f}.png"
            ad_idx += 1

        outpath = outdir / fname
        plot_pair(spec.cpu(), attr.cpu(), label_int, p_ad, outpath)
        print(f"saved: {outpath}")

    print(f"Done. Plots saved under: {outdir}")


if __name__ == "__main__":
    main()
