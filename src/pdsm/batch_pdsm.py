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
    plt.title(f"Flagged Phoneme Histogram (k={k}, Across {tot_samples} WAV)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_file)


def run_batch(saliency_dir, ppgs_dir, top_k, output_dir, preprocess_mode="threshold", pool_mode="sum"
              , threshold_val=0, save_pt=False, experiment_name="", save_hist=False):
    saliency_map_paths = sorted(Path(saliency_dir).glob("*_M.pt"))
    ppg_paths = sorted(Path(ppgs_dir).glob("*.pt"))
    all_phonemes = []
    processed_ctr = 0
    maxed_out = False
    os.makedirs(output_dir, exist_ok=True)
    k = top_k
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
        Mc, selected_phonemes, hit_max_phonemes = phoneme_discretization(M, X_p, top_k, preprocess_mode, threshold_val, pool_mode)
        
        # Set true once, stop once we max out on first sample.
        # Only want to run PDSM if we will get data for all samples
        if not maxed_out and hit_max_phonemes:
            maxed_out = maxed_out
        
            
        all_phonemes.extend(selected_phonemes)
        
        if k == 0:
            k = len(selected_phonemes)
            
        processed_ctr += 1
        
        # Save output
        
        base = os.path.splitext(os.path.basename(M_path))[0]
        print(f"{base} Done")
        
        if save_pt:
            
            out_file = os.path.join(output_dir, f"{base[:4]}.pt")
            torch.save(Mc, out_file)
            print(f"Saved binarized phoneme discretized saliency map to {out_file}")
    
    if save_hist:
        
        hist_fn = f"{experiment_name}_phoneme_hist_nosilence_k{k}.png" if experiment_name else f"phoneme_hist_nosilence_k{k}.png"
        plot_phoneme_hist(all_phonemes,processed_ctr, os.path.join(output_dir, hist_fn), k)
    
    # helpful for Faithfulness vs top_k experiment. 
    return k, maxed_out
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps", default="data/saliencies")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files", default="data/ppg_out")
    parser.add_argument("--k", type=int, default=0, help="Number of phonemes to keep. Defaults to 1/4 of total")
    parser.add_argument("--save_dir", type=str, default="data/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    saliency_map_paths = sorted(Path(args.M_path).glob("*_M.pt"))
    ppg_paths = sorted(Path(args.X_p_path).glob("*.pt"))
    
    run_batch(saliency_map_paths, ppg_paths, args.k, args.save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()