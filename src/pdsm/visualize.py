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
    save_folder = os.path.dirname(csv_fn)
    plt.savefig(f"{save_folder}\\faithfulness_vs_topk.png", bbox_inches="tight")
    plt.close()
    
    
def plot_phoneme_hist(csv_fn, keep_silence):
    """
    Data model of the csv:
    {
        "phoneme_id": int,
        "phoneme_label": str,
        "start_frame": int,
        "end_frame": int,
        "selected": bool
    })
        
        csv_fn = f"{experiment_name}_k{k}_selected_phonemes.csv" if experiment_name else f"k{k}_selected_phonemes.csv"
    """
    
    base_name = csv_fn.removesuffix("_selected_phonemes.csv")
    
    k = base_name.split("k")[-1]
    
    selected_phonemes = pd.read_csv(csv_fn).to_dict(orient='records')
    
    if keep_silence:
        phoneme_counts = Counter(p['phoneme_id'] for p in selected_phonemes if p['selected'])
    else:
        phoneme_counts = Counter(p['phoneme_id'] for p in selected_phonemes if p['selected'] and PHONEMES[p['phoneme_id']] != pypar.SILENCE)

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
    
    
def plot_phoneme_flag_rate(csv_fn, keep_silence):

    df = pd.read_csv(csv_fn)

    print("Calculating phoneme flag rates from", csv_fn)
    # Optionally remove silence rows before any counting
    if not keep_silence:
        df = df[df["phoneme_label"] != pypar.SILENCE]

    # Count total occurrences per phoneme
    total_counts = df.groupby("phoneme_label").size()

    # Count flagged occurrences per phoneme
    flagged_counts = df[df["selected"]].groupby("phoneme_label").size()

    # Compute flag rate (fill missing flagged counts with zero)
    flag_rate = flagged_counts.reindex(total_counts.index, fill_value=0) / total_counts

    # Sort by flag rate
    flag_rate = flag_rate.sort_values(ascending=False)

    # Plot
    
    plt.figure(figsize=(14, 6))
    plt.bar(flag_rate.index, flag_rate.values)
    plt.xlabel("Phoneme")
    plt.ylabel("Flag Rate (Proportion Flagged)")
    plt.title("Phoneme Flag Rate (selection normalized by occurrence frequency)")
    plt.xticks(rotation=90)
    plt.ylim(0, 1)  # rates are probabilities
    plt.tight_layout()

    base_name = csv_fn.removesuffix("_selected_phonemes.csv")
    save_path = f"{base_name}_phoneme_flag_rate.png"
    print(f"Saving phoneme flag rate plot to {save_path}")
    plt.savefig(save_path)
    plt.close()
    
    
def plot_phoneme_duration_hist(csv_fn):
    # Load CSV
    df = pd.read_csv("phonemes.csv")

    # Compute duration
    df["duration"] = df["end_frame"] - df["start_frame"]

    # Split by selection flag
    df_selected = df[df["selected"] == True]
    df_unselected = df[df["selected"] == False]
    
    # ---- 1️⃣ Distribution for selected phonemes ---- #
    plt.figure(figsize=(8, 5))
    plt.hist(df_selected["duration"], bins=30, color="blue", alpha=0.7)
    plt.title("Duration Distribution of Selected Phonemes")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("selected_phonemes_duration.png")
    plt.close()
    
    # ---- 3️⃣ Distribution for all phonemes ---- #
    plt.figure(figsize=(8, 5))
    plt.hist(df["duration"], bins=30, color="green", alpha=0.7)
    plt.title("Duration Distribution of All Phonemes")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("all_phonemes_duration.png")
    plt.close()
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # ---- Subparser: phoneme_duration_hist ----
    p_pdh = subparsers.add_parser("phoneme_duration_hist", help="Plot phoneme duration histogram")
    p_pdh.add_argument("--csv_input", type=str, required=True, help="CSV file with selected phonemes")

    # ---- Subparser: phoneme_freq ----
    p_pf = subparsers.add_parser("phoneme_freq", help="Plot phoneme frequency")
    p_pf.add_argument("--csv_input", type=str, required=True, help="CSV file with selected phonemes")
    p_pf.add_argument("--keep_silence",  action="store_true")
    
     # ---- Subparser: phon_flag_rate ----
    p_pfr = subparsers.add_parser("phon_flag_rate", help="Plot phoneme flag rate")
    p_pfr.add_argument("--csv_input", type=str, required=True, help="CSV file with selected phonemes")
    p_pfr.add_argument("--keep_silence",  action="store_true")

    # ---- Subparser: best_topk ----
    p_topk = subparsers.add_parser("best_topk", help="Run best_topk experiment")
    p_topk.add_argument("--csv_input", type=str, required=True, help="CSV file with faithfulness for different k values")

    args = parser.parse_args()
    
    
    # Generate figures for use in report
    
    if args.experiment == "phoneme_duration_hist":
        pass
    elif args.experiment == "phoneme_freq":
        plot_phoneme_hist(args.csv_input, args.keep_silence)
    elif args.experiment == "phon_flag_rate":
        plot_phoneme_flag_rate(args.csv_input, args.keep_silence)
    elif args.experiment == "best_topk":
        visualize_best_topk(args.csv_input)
    

if __name__ == "__main__":
    main()