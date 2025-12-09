import argparse
from collections import Counter
import os

from batch_pdsm import run_batch

import pandas as pd
import matplotlib.pyplot as plt

from ppgs import PHONEMES
import pypar

def visualize_best_topk(csv_fn, output_dir):
    """Generate graphs of mean Faithfulness vs topk for gradshap and PDSM."""
    # Load CSV
    df = pd.read_csv(csv_fn)

    # Define columns we're interested in for averaging
    faithfulness_cols = [
        "gradshap_ff",
        "pdsm_ff"
    ]

    # Group by top_k and compute mean for relevant columns
    grouped_df = df.groupby("top_k")[faithfulness_cols].mean()

    # Plotting
    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    for col in faithfulness_cols:
        plt.plot(grouped_df.index, grouped_df[col], marker='o', linestyle='-', label=pretty_names.get(col, col))

    plt.title("Faithfulness vs Top-K")
    plt.xlabel("Top-K")
    plt.ylabel("Mean Faithfulness")
    plt.grid(True, linestyle='--', alpha=0.5)
    pretty_names = {
    "gradshap_ff": "GradSHAP Faithfulness",
    "pdsm_ff": "PDSM Faithfulness"
}
    plt.legend()
    
    # Save figure
    out_file = os.path.join(output_dir, "faithfulness_vs_topk.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    
    
def plot_phoneme_hist(csv_fn, out_file, keep_silence):
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
    parser.add_argument("--experiment",
                    type=str,
                    choices=["default", "phoneme_duration_hist", "phoneme_freq", "best_topk"],
                    default="default",
                    help="Select which experiment configuration to run."
                )
    parser.add_argument("--csv_input", type=str, required=False, help="Path to CSV input file for visualization experiments")
    args = parser.parse_args()
    
    os.makedirs(args.pdsm_save_dir, exist_ok=True)
    
    # Generate figures for use in report
    
    if args.experiment == "phoneme_duration_hist":
        pass
    elif args.experiment == "phoneme_freq":
        plot_phoneme_hist(args.csv_input, )
    elif args.experiment == "best_topk":
        pass
    else: # default options
        run_batch(args.M_path, args.X_p_path, args.k, args.pdsm_save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()