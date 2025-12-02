import argparse
import os
import math
from pathlib import Path
from ppgs import PHONEME_TO_INDEX_MAPPING, PHONEMES
import pypar

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving only
import matplotlib.pyplot as plt
from collections import Counter
# Fixes torchcodec and ffmpeg issues
ffmpeg_dll_dir = Path(r"C:/Users/jackm/miniconda3/Library/bin")  # adjust if your conda root differs
assert ffmpeg_dll_dir.exists(), ffmpeg_dll_dir
os.add_dll_directory(str(ffmpeg_dll_dir))

import torch
import numpy as np
from train_adress_cnn import (
    SAMPLE_RATE,
    HOP_LENGTH,
)


def preprocess(M):
    """
    User-defined preprocessing of saliency map M
    M: Tensor [F, T]
    Return: Mf: Tensor [F, T]
    """
    # TODO: implement preprocessing methods THRESHOLDING and ABS
    
    # threshold
    return M.clamp(min=0)
    # return M

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
    prev_label = int(X_ep[0])
    start = 0

    for t in range(1, len(X_ep)):
        if X_ep[t] != prev_label:
            boundaries.append((prev_label, start, t))
            prev_label = int(X_ep[t])
            start = t

    boundaries.append((prev_label, start, len(X_ep)))  # final segment
    return boundaries


def phoneme_discretization(M, X_p, k=0):
    """
    Core algorithm.
    Inputs:
        M: saliency map [F, T]
        X_p: PPG matrix [N, T]
        k: number of phonemes to select. Default is 1/4 of all phonemes
    Output:
        Mc: phoneme discretized saliency map [F, T]
    """
    F, T = M.shape

    # Step 1: preprocess
    print("Preprocessing")
    Mf = preprocess(M)
    # Mf = M

    # Step 2: time-to-phoneme alignment
    print("Aligning phonemes and collecting boundaries")
    X_ep = torch.argmax(X_p, dim=0)  # [T]

    # Step 3: determine phoneme boundaries
    boundaries = get_phoneme_boundaries(X_ep)

    # Step 4: energy calculation per phoneme segment
    if k==0:
        k = len(boundaries) // 4
    print(f"Pooling energy per phoneme segment. Keeping {k} highest energy phonemes")
    energies = [pool(Mf[:, s:e]) for (_, s, e) in boundaries]

    # Step 5: select top-k phonemes by energy
    top_k_indices = np.argsort(energies)[-k:]

    # Step 6: initialize output
    Mc = torch.zeros_like(M)

    # Step 7: fill mask & track labels
    print("Binarizing")
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


def plot_phoneme_hist(selected_phonemes, tot_samples, out_file, k):
    
    phoneme_counts = Counter(p['phoneme_id'] for p in selected_phonemes if PHONEMES[p['phoneme_id']] != pypar.SILENCE)

    # Sort by frequency descending
    sorted_items = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)

    # Separate into lists
    ids   = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    labels = [PHONEMES[i] for i in ids]

    # Plot histogram
    plt.figure(figsize=(14, 6))
    plt.bar(labels, counts)
    plt.xlabel("Phoneme")
    plt.ylabel("Frequency")
    plt.title(f"AD- Flagged Phoneme Histogram (k={k}, Across {tot_samples} WAV)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_file)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files")
    parser.add_argument("--k", type=int, default=0, help="Number of phonemes to keep. Defaults to 1/4 of total")
    parser.add_argument("--save_dir", type=str, default="src/pdsm/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    saliency_map_paths = sorted(Path(args.M_path).glob("*_M.pt"))
    ppg_paths = sorted(Path(args.X_p_path).glob("*.pt"))
    
    all_phonemes = []
    processed_ctr = 0
    for M_path in saliency_map_paths:
        patientID = os.path.splitext(os.path.basename(M_path))[0][:4] # first 4 chars are patient id
        ppg_path = ""
        
        for i,p in enumerate(ppg_paths):
            if patientID in str(p):
                ppg_path = p
                ppg_paths.pop(i)
                break
        
        # Load inputs
        M = torch.load(M_path)        # expected shape [F, T]
        X_p = torch.load(ppg_path)    # expected shape [N, T]
        
        # Run algorithm
        Mc, selected_phonemes = phoneme_discretization(M, X_p, args.k)
        
        all_phonemes.extend(selected_phonemes)
        
        processed_ctr += 1
        
        # Save output
        
        base = os.path.splitext(os.path.basename(M_path))[0]
        print(f"{base} Done")
        
        if args.save_pt:
            
            out_file = os.path.join(args.save_dir, f"{base[:4]}.pt")
            torch.save(Mc, out_file)
            print(f"Saved phoneme discretized saliency map to {out_file}")
    
    plot_phoneme_hist(all_phonemes,processed_ctr, os.path.join(args.save_dir, f"cc_phoneme_hist_nosilence_k{args.k}.png"), args.k)
    

if __name__ == "__main__":
    main()