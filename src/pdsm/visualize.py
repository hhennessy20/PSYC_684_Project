import argparse
from collections import Counter
import os

import numpy as np

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
    plt.rcParams.update({
        "font.size": 18,         # General font size
        "axes.titlesize": 20,    # Title
        "axes.labelsize": 18,    # Axis labels
        "xtick.labelsize": 16,   # Tick labels
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })
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
    plt.rcParams.update({
        "font.size": 18,         # General font size
        "axes.titlesize": 20,    # Title
        "axes.labelsize": 18,    # Axis labels
        "xtick.labelsize": 16,   # Tick labels
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })
    
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
    plt.rcParams.update({
        "font.size": 18,         # General font size
        "axes.titlesize": 20,    # Title
        "axes.labelsize": 18,    # Axis labels
        "xtick.labelsize": 16,   # Tick labels
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })
    
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
    df = pd.read_csv(csv_fn)

    # Compute duration
    df["duration"] = df["end_frame"] - df["start_frame"]

    # Split by selection flag
    df_selected = df[df["selected"] == True]
    df_unselected = df[df["selected"] == False]
    
    plt.rcParams.update({
        "font.size": 18,         # General font size
        "axes.titlesize": 20,    # Title
        "axes.labelsize": 18,    # Axis labels
        "xtick.labelsize": 16,   # Tick labels
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })
    
    base_name = csv_fn.removesuffix("_selected_phonemes.csv")
    #  Distribution for selected phonemes ---- #
    plt.figure(figsize=(8, 5))
    plt.hist(df_selected["duration"], bins=100, color="blue", alpha=0.7)
    plt.title("Duration Distribution of Selected Phonemes")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    save_path = f"{base_name}_selected_phonemes_duration.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    
    # Phoneme duration plot
    mean_duration = df.groupby("phoneme_label")["duration"].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(mean_duration.index, mean_duration.values, alpha=0.8)
    plt.title("Mean Duration by Phoneme Label")
    plt.xlabel("Phoneme Label")
    plt.ylabel("Mean Duration (frames)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Improve readability for many phonemes
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = f"{base_name}_mean_duration_by_phoneme.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    plot_duration_flag_rate(csv_fn)
    
    
def plot_duration_flag_rate(csv_fn,  bin_size=20, max_duration=None):
    df = pd.read_csv(csv_fn)

    # Compute duration
    df["duration"] = df["end_frame"] - df["start_frame"]

    # Auto determine max duration if not provided
    if max_duration is None:
        max_duration = df["duration"].max()

    # Create duration bins
    bins = np.arange(0, max_duration + bin_size, bin_size)
    df["duration_bin"] = pd.cut(df["duration"], bins=bins, right=False)

    # Count total & selected in each bin
    total_counts = df.groupby("duration_bin").size()
    selected_counts = df[df["selected"]].groupby("duration_bin").size()
    flag_rate = selected_counts.reindex(total_counts.index, fill_value=0) / total_counts

    bin_centers = np.array([(interval.left + interval.right) / 2 for interval in flag_rate.index])


    # Plot setup for LaTeX two-column figures
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    })

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, flag_rate.values, width=bin_size * 0.9, alpha=0.8)
    plt.title("Selection Rate by Duration Bin")
    plt.xlabel("Duration (frames)")
    plt.ylabel("Selection Rate")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Use fewer ticks for readability
    tick_step = 250
    major_ticks = np.arange(0, max_duration + tick_step, tick_step)
    plt.xticks(major_ticks)

    plt.tight_layout()
    
    base_name = csv_fn.removesuffix("_selected_phonemes.csv")
    save_path = f"{base_name}_duration_flag_rate.png"
    plt.savefig(save_path, dpi=300)

    print(f"Saved: {save_path}")
    return flag_rate

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
        plot_phoneme_duration_hist(args.csv_input)
    elif args.experiment == "phoneme_freq":
        plot_phoneme_hist(args.csv_input, args.keep_silence)
    elif args.experiment == "phon_flag_rate":
        plot_phoneme_flag_rate(args.csv_input, args.keep_silence)
    elif args.experiment == "best_topk":
        visualize_best_topk(args.csv_input)
    

if __name__ == "__main__":
    main()