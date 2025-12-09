import argparse
from collections import Counter
import os

from batch_pdsm import run_batch

import pandas as pd
import matplotlib.pyplot as plt

from ppgs import PHONEMES
import pypar

def visualize_best_topk(csv_fn):
    """Generate graphs of mean Faithfulness vs topk for PDSM."""
    # Load CSV
    df = pd.read_csv(csv_fn)

    # Define columns we're interested in for averaging
    faithfulness_cols = ["pdsm_ff"]
    
    
    pretty_names = {
        "pdsm_ff": "PDSM"
    }

    # Group by top_k → mean and std
    grouped_mean = df.groupby("top_k")[faithfulness_cols].mean()
    grouped_std  = df.groupby("top_k")[faithfulness_cols].std()

    # Plotting
    plt.figure(figsize=(10, 6))

    for col in faithfulness_cols:
        plt.errorbar(
            grouped_mean.index,
            grouped_mean[col],
            yerr=grouped_std[col],
            marker='o',
            linestyle='-',
            capsize=4,
            label=pretty_names.get(col, col)
        )

    plt.title("Faithfulness vs Top-K")
    plt.xlabel("Top-K")
    plt.ylabel("Mean Faithfulness")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend()
    
    # Save figure
    plt.savefig("faithfulness_vs_topk.png", bbox_inches="tight")
    plt.close()
    
    
def plot_phoneme_hist(csv_fn, keep_silence):
    """
    Data model of the csv:
    {
        "phoneme_id": int,
        "phoneme_label": str,
        "start_frame": int,
        "end_frame": int
    })
        
        csv_fn = f"{experiment_name}_k{k}_selected_phonemes.csv" if experiment_name else f"k{k}_selected_phonemes.csv"
    """
    
    base_name = csv_fn.removesuffix("_selected_phonemes.csv")
    
    k = int(base_name.split("_k")[-1])
    
    selected_phonemes = pd.read_csv(csv_fn).to_dict(orient='records')
    
    if keep_silence:
        phoneme_counts = Counter(p['phoneme_id'] for p in selected_phonemes)
    else:
        phoneme_counts = Counter(p['phoneme_id'] for p in selected_phonemes if  PHONEMES[p['phoneme_id']] != pypar.SILENCE)

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
    plt.title(f"Flagged Phoneme Histogram (k={k})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{base_name}_phoneme_histogram.png")
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # ---- Subparser: phoneme_duration_hist ----
    p_pdh = subparsers.add_parser("phoneme_duration_hist", help="Plot phoneme duration histogram")
    p_pdh.add_argument("--csv_input", type=str, required=True, help="CSV file with durations")

    # ---- Subparser: phoneme_freq ----
    p_pf = subparsers.add_parser("phoneme_freq", help="Plot phoneme frequency")
    p_pf.add_argument("--csv_input", type=str, required=True, help="CSV file with frequencies")
    p_pf.add_argument("--keep_silence",  action="store_true")

    # ---- Subparser: best_topk ----
    p_topk = subparsers.add_parser("best_topk", help="Run best_topk experiment")
    p_topk.add_argument("--csv_input", type=str, required=True, help="CSV file with faithfulness for different k values")

    args = parser.parse_args()
    
    
    # Generate figures for use in report
    
    if args.experiment == "phoneme_duration_hist":
        pass
    elif args.experiment == "phoneme_freq":
        plot_phoneme_hist(args.csv_input, args.keep_silence)
    elif args.experiment == "best_topk":
        visualize_best_topk(args.csv_input)
    

if __name__ == "__main__":
    main()