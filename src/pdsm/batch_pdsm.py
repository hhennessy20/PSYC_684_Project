import argparse
import os
import sys
import math
from pathlib import Path

from ppgs import PHONEME_TO_INDEX_MAPPING, PHONEMES
import pypar

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving only
import matplotlib.pyplot as plt
from collections import Counter

import torch

from pdsm import phoneme_discretization


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


def run_batch(saliency_map_paths, ppg_paths, top_k, output_dir, preprocess_mode, pool_mode, save_pt=False):
    
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
        Mc, selected_phonemes = phoneme_discretization(M, X_p, top_k)
        
        all_phonemes.extend(selected_phonemes)
        
        processed_ctr += 1
        
        # Save output
        
        base = os.path.splitext(os.path.basename(M_path))[0]
        print(f"{base} Done")
        
        if save_pt:
            
            out_file = os.path.join(output_dir, f"{base[:4]}.pt")
            torch.save(Mc, out_file)
            print(f"Saved binarized phoneme discretized saliency map to {out_file}")
    
    # change cc <-> cd depending on input
    plot_phoneme_hist(all_phonemes,processed_ctr, os.path.join(output_dir, f"cc_phoneme_hist_nosilence_k{top_k}.png"), top_k)
    

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
    
    run_batch(saliency_map_paths, ppg_paths, args.k, args.save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()