
import argparse, os
from pathlib import Path

import pandas as pd

from batch_pdsm import run_batch
from faithfulness import run_faithfulness

def experiment_preprocess_pool(saliency_dir, ppgs_dir, output_dir, top_k, save_pt, overwrite_pdsm_pt=True):
    
    # All 4 combinations
    preprocess_modes = ["threshold", "absolute"]
    pool_modes = ["sum", "mean"]

    csv_out_paths = []
    for preprocess in preprocess_modes:
        for pool in pool_modes:
            combo_name = f"{preprocess}_{pool}"
            combo_save_dir = os.path.join(output_dir, combo_name)


            print(f"Running combo: {combo_name}")
            k_top_phon = top_k if top_k > 0 else "auto"
            if overwrite_pdsm_pt:
                k_top_phon = run_batch(
                    saliency_dir,
                    ppgs_dir,
                    top_k,
                    combo_save_dir,
                    preprocess,
                    pool,
                    save_pt,
                    experiment_name=combo_name
                )
            csv_fn = os.path.join(combo_save_dir, f"{combo_name}_faithfulness_results_k{k_top_phon}.csv")
            csv_out_paths.append(csv_fn)
            run_faithfulness(Path(saliency_dir), Path(ppgs_dir), Path(combo_save_dir), k_top_phon, csv_output_fn=csv_fn)
            
    print_csv_results(csv_out_paths)
    

def print_csv_results(csv_paths):
    results = []

    for file in csv_paths:
        df = pd.read_csv(file)

        results.append({
            "filename": file,
            "gradshap_mean": df['gradshap_ff'].mean(),
            "gradshap_std": df['gradshap_ff'].std(),
            "pdsm_mean": df['pdsm_ff'].mean(),
            "pdsm_std": df['pdsm_ff'].std()
        })

    # Create a summary table
    summary_df = pd.DataFrame(results)

    # Format the output
    print(summary_df.to_string(index=False))

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps", default="data/saliencies")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files", default="data/ppg_out")
    parser.add_argument("--k", type=int, default=0, help="Number of phonemes to keep. Defaults to 1/4 of total")
    parser.add_argument("--save_dir", type=str, default="data/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    parser.add_argument("--overwrite_pdsm_pt",  action="store_true")
    parser.add_argument("--experiment",
                        type=str,
                        choices=["default", "preprocess_pool", "phoneme_duration", "test"],
                        default="default",
                        help="Select which experiment configuration to run."
                    )
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.experiment == "preprocess_pool":
        experiment_preprocess_pool(args.M_path, args.X_p_path, args.save_dir, args.k, args.save_pt, args.overwrite_pdsm_pt)
    else: # default options
        run_batch(args.M_path, args.X_p_path, args.k, args.save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()