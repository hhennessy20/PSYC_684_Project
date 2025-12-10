import argparse
import os
import sys
import math
from pathlib import Path



import matplotlib
import pandas as pd
matplotlib.use('Agg')  # non-interactive backend for saving only
import matplotlib.pyplot as plt
from collections import Counter

import torch

from pdsm import phoneme_discretization





def run_batch(saliency_dir, ppgs_dir, top_k, output_dir, preprocess_mode="threshold", pool_mode="sum"
              , threshold_val=0, save_pt=False, experiment_name="", save_csv=False, verbose=False):
    saliency_map_paths = sorted(Path(saliency_dir).glob("*_M.pt"))
    ppg_paths = sorted(Path(ppgs_dir).glob("*.pt"))
    all_phonemes = []
    processed_ctr = 0
    maxed_out = False
    os.makedirs(output_dir, exist_ok=True)
    k = top_k
    print(f"Running PDSM batch with k={k}, preprocess_mode={preprocess_mode}, pool_mode={pool_mode}")
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
        Mc, selected_phonemes, hit_max_phonemes = phoneme_discretization(M, X_p, top_k, preprocess_mode, threshold_val, pool_mode, verbose)
        
        # Set true once, stop once we max out on first sample.
        # Only want to run PDSM if we will get data for all samples
        if not maxed_out and hit_max_phonemes:
            maxed_out = maxed_out
        
            
        all_phonemes.extend(selected_phonemes)
        
        processed_ctr += 1
        
        # Save output
        
        base = os.path.splitext(os.path.basename(M_path))[0]
        if verbose: print(f"{base} Done")
        
        if save_pt:
            
            out_file = os.path.join(output_dir, f"{base[:4]}.pt")
            torch.save(Mc, out_file)
            if verbose: print(f"Saved binarized phoneme discretized saliency map to {out_file}")
    
    if save_csv:
        
        csv_fn = f"{experiment_name}_k{k}_selected_phonemes.csv" if experiment_name else f"k{k}_selected_phonemes.csv"
        df = pd.DataFrame(all_phonemes)
        df.to_csv(os.path.join(output_dir, csv_fn), index=False)
        if verbose: print(f"Saved all selected phonemes to {csv_fn}")
    
    # helpful for Faithfulness vs top_k experiment. 
    return k, maxed_out
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps", default="data/saliencies")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files", default="data/ppg_out")
    parser.add_argument("--k", type=float, default=.1, help="Number of phonemes to keep. Defaults to 1/10 of total")
    parser.add_argument("--save_dir", type=str, default="data/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    saliency_map_paths = sorted(Path(args.M_path).glob("*_M.pt"))
    ppg_paths = sorted(Path(args.X_p_path).glob("*.pt"))
    
    run_batch(saliency_map_paths, ppg_paths, args.k, args.save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()