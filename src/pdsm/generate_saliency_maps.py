import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_CKPT_PATH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "cnn"))
from interpret_gradshap_adress import get_saliency_maps_for_pdsm
from train_adress_cnn import (
    AudioCNN,
    set_seed,
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    DURATION_SEC,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_root",
        default="train/Diarized_full_wave_enhanced_audio",
        help="Root directory for diarized full-wave audio.",
    )
    parser.add_argument(
        "--saliency_save_dir",
        default="saliencies",
        help="Output dir",
    )
    parser.add_argument(
        "--model_ckpt",
        default=None,
        help="Path to trained AudioCNN checkpoint. Defaults to MODEL_CKPT_PATH from config.",
    )
    # parser.add_argument(
    #     "--output_dir",
    #     default="gradshap_val_plots",
    #     help="Directory to save GradSHAP plots.",
    # )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation subject fraction.",
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
        model_ckpt=args.model_ckpt if args.model_ckpt else MODEL_CKPT_PATH,
        val_split=args.val_split,
        hop_sec=args.hop_sec,
        device=device,
        returnAll=True
    )
    
    # results:
    # {
    #     "M": M_full_np, shape (64, time frames)
    #     "spec": spec_full_np, 
    #     "label": label,
    #     "p_ad": p_ad,
    #     "path": str(wav_path),
    #     "subject": sid,
    # }
    
    
    os.makedirs(args.saliency_save_dir, exist_ok=True)

    for i, item in enumerate(results):
        M_tensor = torch.tensor(item["M"])
        spec_tensor = torch.tensor(item["spec"])
        
        # Create a filename base — adjust as needed
        subject = item["subject"]
        base_name = f"{subject}"
        
        M_path = os.path.join(args.saliency_save_dir, f"{base_name}_M.pt")
        spec_path = os.path.join(args.saliency_save_dir, f"{base_name}_spec.pt")

        torch.save(M_tensor, M_path)
        torch.save(spec_tensor, spec_path)

        print(f"Saved: {M_path}")
        print(f"Saved: {spec_path}")
    
    return

if __name__=="__main__":
    main()