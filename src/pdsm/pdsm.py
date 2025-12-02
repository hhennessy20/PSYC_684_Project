import argparse
import os
from pathlib import Path
from ppgs import PHONEME_TO_INDEX_MAPPING, PHONEMES
import matplotlib.pyplot as plt

# Fixes torchcodec and ffmpeg issues
ffmpeg_dll_dir = Path(r"C:/Users/jackm/miniconda3/Library/bin")  # adjust if your conda root differs
assert ffmpeg_dll_dir.exists(), ffmpeg_dll_dir
os.add_dll_directory(str(ffmpeg_dll_dir))

import torch
import numpy as np


def preprocess(M):
    """
    User-defined preprocessing of saliency map M
    M: Tensor [F, T]
    Return: Mf: Tensor [F, T]
    """
    # TODO: implement preprocessing methods THRESHOLDING and ABS
    return M

def pool(M_segment):
    """
    User-defined pooling operator for energy calculation.
    M_segment: Tensor [F, duration]
    Return: scalar energy value
    """
    # TEST MEAN POOLING
    # sum pooling
    return torch.sum(M_segment).item()

def get_phoneme_boundaries(X_ep):
    """
    Determine phoneme boundaries from argmax phoneme sequence X_ep.
    Input:
        X_ep: [T] sequence of phoneme indices per frame
    Output:
        List of tuples (phoneme_label, start_idx, end_idx)
    """
    boundaries = []
    prev_label = X_ep[0]
    start = 0

    for t in range(1, len(X_ep)):
        if X_ep[t] != prev_label:
            boundaries.append((prev_label, start, t))
            prev_label = X_ep[t]
            start = t

    boundaries.append((prev_label, start, len(X_ep)))  # final segment
    return boundaries

def plot_pdsm(M, Mc, selected_phonemes, out_file_path):
    
    plt.imshow(Mc, aspect='auto')
    for entry in selected_phonemes:
        mid = (entry["start"] + entry["end"]) // 2
        plt.text(mid, M.shape[0] + 0.5, entry["phoneme_label"],
                ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.savefig(out_file_path)
    

def phoneme_discretization(M, X_p, k):
    """
    Core algorithm.
    Inputs:
        M: saliency map [F, T]
        X_p: PPG matrix [N, T]
        k: number of phonemes to select
    Output:
        Mc: phoneme discretized saliency map [F, T]
    """
    F, T = M.shape

    # Step 1: preprocess
    # Mf = preprocess(M)
    Mf = M

    # Step 2: time-to-phoneme alignment
    X_ep = torch.argmax(X_p, dim=0)  # [T]

    # Step 3: determine phoneme boundaries
    boundaries = get_phoneme_boundaries(X_ep)

    # Step 4: energy calculation per phoneme segment
    energies = [pool(Mf[:, s:e]) for (_, s, e) in boundaries]

    # Step 5: select top-k phonemes by energy
    top_k_indices = np.argsort(energies)[-k:]

    # Step 6: initialize output
    Mc = torch.zeros_like(M)

    # Step 7: fill mask & track labels
    selected_phonemes = []
    for idx in top_k_indices:
        p, s, e = boundaries[idx]
        Mc[:, s:e] = 1.0
        selected_phonemes.append({
            "phoneme_id": int(p),
            "phoneme_label": PHONEMES[int(p)],
            "start": int(s),
            "end": int(e)
        })

    # Sort segments by start time for clean plotting
    selected_phonemes.sort(key=lambda x: x["start"])

    return Mc, selected_phonemes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency map .pt file")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt file")
    parser.add_argument("--k", type=int, default=30, help="Number of phonemes to keep")
    parser.add_argument("--save_dir", type=str, default="src/pdsm/pdsm_out", help="Where to save output")
    args = parser.parse_args()

    # Load inputs
    M = torch.load(args.M_path)        # expected shape [F, T]
    X_p = torch.load(args.X_p_path)    # expected shape [N, T]

    # Run algorithm
    Mc, selected_phonemes = phoneme_discretization(M, X_p, args.k)

    # Save output
    os.makedirs(args.save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.M_path))[0]
    out_file = os.path.join(args.save_dir, f"{base[:3]}.pt")
    torch.save(Mc, out_file)
    print(f"Saved phoneme discretized saliency map to {out_file}")
    
    # save figure
    figure_out_file = os.path.join(args.save_dir, f"{base[:3]}_pdsm.png")
    plot_pdsm(M, Mc, selected_phonemes, figure_out_file)
    print(f"Saved figure of phoneme discretized saliency map to {out_file}")


if __name__ == "__main__":
    main()