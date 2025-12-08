import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_CKPT_PATH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "cnn"))
from train_adress_cnn import (
    AudioCNN,
    set_seed,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    DURATION_SEC,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, example_input, device=DEVICE):
    """
    Loads an AudioCNN state_dict, ignoring old classifier layers.
    Automatically reinitializes classifier from input shape.
    """

    # Initialize model (classifier=None for now)
    model = AudioCNN()
    model.to(device)

    # Load checkpoint
    state = torch.load(model_path, map_location=device)

    # Filter out classifier keys (they do not match)
    filtered = {k: v for k, v in state.items() if not k.startswith("classifier.")}

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Force classifier creation using example input
    if model.classifier is None:
        with torch.no_grad():
            model._init_fc(example_input.to(device))

    model.eval()
    return model


def infer_directory(model, spectrogram_dir, device=DEVICE, batch_size=1):
    """
    Run inference on all .pt spectrogram tensors in a directory.

    Args:
        model: Loaded PyTorch model
        spectrogram_dir: Directory containing .pt spectrograms
        device: CPU/GPU
        batch_size: Optional batching

    Returns:
        results: dict {filename: class_probabilities}
    """
    results = {}
    spec_files = sorted(Path(spectrogram_dir).glob("*_spec.pt"))

    with torch.no_grad():
        for spec_file in spec_files:
            spect = torch.load(spec_file, map_location=device)

            # Ensure model expected dims [B, C, F, T]
            if spect.dim() == 3:
                spect = spect.unsqueeze(0)
            elif spect.dim() == 2:
                spect = spect.unsqueeze(0).unsqueeze(0)

            spect = spect.to(device)

            logits = model(spect)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            results[spec_file.name] = probs

            print(f"[OK] {spec_file.name} -> {probs}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AudioCNN Spectrogram Inference")
    parser.add_argument("--model", default=None, help="Path to pretrained AudioCNN .pt file. Defaults to MODEL_CKPT_PATH from config.")
    parser.add_argument("--data", required=True, help="Directory of spectrogram .pt files")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    spec_files = sorted(Path(args.data).glob("*_spec.pt"))
    first_spec = torch.load(spec_files[0])  
    
    
    if first_spec.dim() == 3:
        first_spec = first_spec.unsqueeze(0)  # ensure batch
    elif first_spec.dim() == 2:
        first_spec = first_spec.unsqueeze(0).unsqueeze(0)
    model_path = args.model if args.model else MODEL_CKPT_PATH
    model = load_model(model_path, first_spec)
    results = infer_directory(model, args.data, batch_size=args.batch_size)

    print("\n✔ Inference Complete!")
    print(f"Processed {len(results)} spectrograms.")