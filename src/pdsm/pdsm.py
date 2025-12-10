import argparse
import os
import sys
from enum import Enum
from pathlib import Path

from ppgs import PHONEME_TO_INDEX_MAPPING, PHONEMES

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving only
import matplotlib.pyplot as plt

import torch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "cnn"))
from train_adress_cnn import (
    SAMPLE_RATE,
    HOP_LENGTH,
)


def plot_pdsm(M, Mc, spec, selected_phonemes, out_file_path, includeInputVisuals=False, start_frame=None,end_frame=None):
    
    """
    Overlay the phoneme-discretized saliency map (Mc) on top of
    the original spectrogram (spec). Phoneme labels shown as x-ticks.
    
    This is just an example so it will be cropped around the center for legibility.

    Inputs:
      spec: original spectrogram [F, T]
      Mc: binary mask [F, T]
      selected_phonemes: list of phoneme dicts
    Output:
      Figure saved to out_file_path
    """

    F, T = spec.shape
    
    frame_hop = HOP_LENGTH / SAMPLE_RATE
    times = np.arange(T) * frame_hop  # in seconds
    if start_frame is not None:
        times = np.arange(start_frame, end_frame) * frame_hop  # in seconds
    
    if includeInputVisuals:
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        # saliency
        axs[0].imshow(M, aspect='auto',
                    extent=[times[0], times[-1], 0, F],
                    origin='lower')
        axs[0].set_title("Original Saliency Map (M)")
        axs[0].set_ylabel("Mel Bins")

        # original spectrogram
        axs[1].imshow(spec, aspect='auto',
                    extent=[times[0], times[-1], 0, F],
                    origin='lower')
        axs[1].set_title("Original Spectrogram")
        axs[1].set_ylabel("Mel Bins")
        
        overlay_axis = 2
        
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 10), sharex=True)
        axs = [axs]
        overlay_axis = 0
        
    # Background spectrogram
    axs[overlay_axis].imshow(spec, aspect='auto',
               extent=[times[0], times[-1], 0, F],
               origin='lower')

    # Overlay top-k region mask
    axs[overlay_axis].imshow(Mc, aspect='auto',
               extent=[times[0], times[-1], 0, F],
               origin='lower', alpha=0.35)

    # Place text inside highlighted regions
    for entry in selected_phonemes:
        start = entry["start"] + start_frame if start_frame is not None else entry["start"]
        end = entry["end"] + start_frame if start_frame is not None else entry["end"]
        start_sec = start * frame_hop
        end_sec = end * frame_hop
        mid_sec = (start_sec + end_sec) / 2.0

        # vertical placement: mid-frequency of the mask
        mid_freq = F * 0.5

        # phoneme text label
        axs[overlay_axis].text(mid_sec, mid_freq,
                 entry["phoneme_label"],
                 color="white", fontsize=10, rotation=90,
                 ha="center", va="center",
                 bbox=dict(facecolor='black', alpha=0.5, lw=0))

        # optional boundaries
        axs[overlay_axis].axvline(x=start_sec, linestyle='--',
                    linewidth=0.5, color='white')

    axs[overlay_axis].set_title("Spectrogram with Phoneme-Discretized Saliency Overlay")
    axs[overlay_axis].set_xlabel("Time (seconds)")
    axs[overlay_axis].set_ylabel("Mel Bins")

    plt.tight_layout()
    plt.savefig(out_file_path, dpi=200)
    plt.close()
    
    
    

def preprocess(M, mode="threshold", threshold_val=0):
    """
    User-defined Preprocessing of saliency map M
    M: Tensor [F, T]
    Return: Mf: Tensor [F, T]
    """
    # threshold
    if mode == "threshold":
        return M.clamp(min=threshold_val)
    elif mode == "absolute":
        return torch.abs(M)
    else:
        raise ValueError("Invalid mode. Supported modes: 'threshold', 'absolute'.")
    

def pool(M_segment, mode="sum"):
    """
    Pooling operator for energy calculation. 
    Parameters: 
    - M_segment: Tensor [F, duration]
    - type: [sum, mean]
    Return: scalar energy value
    """
    
    if mode == "sum":
        return torch.sum(M_segment).item()
    elif mode == "mean":
        return torch.mean(M_segment).item()
    else:
        raise ValueError("Invalid value for [type] parameter. Must be 'sum' or 'mean'.")
    

def get_phoneme_boundaries(X_ep):
    """
    Determine phoneme boundaries from argmax phoneme sequence X_ep.
    Input:
        X_ep: [T] sequence of phoneme indices per frame
    Output:
        List of tuples (phoneme_label, start_idx, end_idx)
    """
    boundaries = []
    prev_label = int(X_ep[0])
    start = 0

    for t in range(1, len(X_ep)):
        if X_ep[t] != prev_label:
            boundaries.append((prev_label, start, t))
            prev_label = int(X_ep[t])
            start = t

    boundaries.append((prev_label, start, len(X_ep)))  # final segment
    return boundaries


def phoneme_discretization(M, X_p, k=.1, preprocess_mode="threshold", threshold_val=0, pool_mode="sum", verbose=True):
    """
    Core algorithm.
    Inputs:
        M: saliency map [F, T]
        X_p: PPG matrix [N, T]
        k: number of phonemes to select. Default is 1/4 of all phonemes. If k=0, keep all phonemes.
    Output:
        Mc: phoneme discretized saliency map [F, T]
    """

    # Step 1: preprocess
    if verbose: print("Preprocessing")
    Mf = preprocess(M, preprocess_mode, threshold_val)
    # Mf = M

    # Step 2: time-to-phoneme alignment
    if verbose: print("Aligning phonemes and collecting boundaries")
    X_ep = torch.argmax(X_p, dim=0)  # [T]

    # Step 3: determine phoneme boundaries
    boundaries = get_phoneme_boundaries(X_ep)

    # Step 4: energy calculation per phoneme segment
    maxed_out = False
    if k==0:
        k = len(boundaries)
    else:
        if k >= len(boundaries):
            maxed_out = True
            
        if isinstance(k, float):
            
            if k <= 1.0:
                k = min(int(k*len(boundaries)), len(boundaries))
            else:
                k = min(int(k), len(boundaries))
            
        else:
            k = min(k, len(boundaries))
            
    if verbose: print(f"Pooling energy per phoneme segment. Keeping {k} highest energy phonemes")
    energies = np.array([pool(Mf[:, s:e], pool_mode) for (_, s, e) in boundaries])

    # Step 5: select top-k phonemes by energy
    top_k_indices = np.argsort(energies)[-k:]
    
    rejected_phonemes = np.argsort(energies)[:-k]

    # Step 6: initialize output
    Mc = torch.zeros_like(M)

    # Step 7: fill mask & track labels
    if verbose: print("Binarizing")
    phonemes = []
    for idx in top_k_indices:
        p, s, e = boundaries[idx]
        Mc[:, s:e] = 1.0
        phonemes.append({
            "phoneme_id": int(p),
            "phoneme_label": PHONEMES[int(p)],
            "start_frame": int(s),
            "end_frame": int(e),
            "selected": True
        })
    
    for idx in rejected_phonemes:
        p, s, e = boundaries[idx]
        phonemes.append({
            "phoneme_id": int(p),
            "phoneme_label": PHONEMES[int(p)],
            "start_frame": int(s),
            "end_frame": int(e),
            "selected": False
        })

    # Sort segments by start time for clean plotting
    phonemes.sort(key=lambda x: x["start_frame"])

    return Mc, phonemes, maxed_out


def crop_tensors(spec, M, X_p, crop_fraction):
    F, T = spec.shape

    crop_T = int(T * crop_fraction)
    crop_T = max(1, min(crop_T, T))

    center = T // 2
    start = max(0, center - crop_T // 2)
    end = min(T, start + crop_T)

    # Crop before processing
    spec = spec[:, start:end]
    M = M[:, start:end]
    X_p = X_p[:, start:end]
    
    return spec, M, X_p, start, end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency map .pt file")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt file")
    parser.add_argument("--Spec_path", type=str, required=True, help="Path to spectrogram .pt file")
    parser.add_argument("--k", type=int, default=0, help="Number of phonemes to keep")
    parser.add_argument("--save_dir", type=str, default="src/pdsm/pdsm_out", help="Where to save output")
    parser.add_argument("--crop_fraction", type=float, default=1.0, help="Amount of sample to crop. useful for creating legible figures")
    parser.add_argument(
        "--include_input_visual",
        action="store_true",
        help="Include the spectrogram and saliency map in the figure"
    )
    args = parser.parse_args()

    # Load inputs
    M = torch.load(args.M_path)        # expected shape [F, T]
    spec = torch.load(args.Spec_path)    # expected shape [F, T]
    X_p = torch.load(args.X_p_path)    # expected shape [N, T]
    
    if args.crop_fraction < 1.0:
        spec, M, X_p, start, end = crop_tensors(spec, M, X_p, args.crop_fraction)

    # Run algorithm
    Mc, selected_phonemes = phoneme_discretization(M, X_p, args.k)

    # Save output
    os.makedirs(args.save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.M_path))[0]
    
    if args.crop_fraction < 1.0:
        out_file = os.path.join(args.save_dir, f"{base[:4]}_crop{str(args.crop_fraction)}.pt")
    else:
        out_file = os.path.join(args.save_dir, f"{base[:4]}.pt")
    torch.save(Mc, out_file)
    print(f"Saved phoneme discretized saliency map to {out_file}")
    
    # save figure
    figure_out_file = os.path.join(args.save_dir, f"{base[:4]}_pdsm")
    if args.crop_fraction < 1.0:
        
        figure_out_file += f"_crop{str(args.crop_fraction)}"
    if args.include_input_visual:
        figure_out_file += "_withInput"
        
    figure_out_file += ".png"
    plot_pdsm(M, Mc, spec, selected_phonemes, figure_out_file, args.include_input_visual, start, end)
    print(f"Saved figure of phoneme discretized saliency map to {figure_out_file}")


if __name__ == "__main__":
    main()