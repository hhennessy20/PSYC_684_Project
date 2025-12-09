
from datetime import datetime
import glob
import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from batch_pdsm import run_batch
from faithfulness import run_faithfulness

def experiment_preprocess_pool(saliency_dir, ppgs_dir, output_dir, top_k, save_pt, use_existing_pdsm=False):
    """Tests the 4 possible combinations of preprocess_mode and pool_mode.
    
    Parameters:
    saliency_dir (str): Path to saliency maps
    ppgs_dir (str): Path to PPG .pt files
    output_dir (str): Where to save output
    top_k (int): Number of phonemes to keep. Defaults to 1/4
    save_pt (bool): Whether to save the pdsm .pt files
    overwrite_pdsm_pt (bool): Whether to overwrite existing pdsm .pt files
    """
    
    preprocess_modes = ["threshold", "absolute"]
    pool_modes = ["sum", "mean"]

    csv_out_paths = []
    for preprocess in preprocess_modes:
        for pool in pool_modes:
            combo_name = f"{preprocess}_{pool}"
            combo_save_dir = os.path.join(output_dir, combo_name)

            print(f"Running combo: {combo_name}")
            print("top_k:", top_k)
            k_top_phon = top_k 
            
            # Generate PDSM output
            if not use_existing_pdsm:
                k_top_phon = run_batch(
                    saliency_dir,
                    ppgs_dir,
                    top_k,
                    combo_save_dir,
                    preprocess,
                    pool,
                    save_pt=save_pt,
                    experiment_name=combo_name
                )
                
            k_fn = top_k if top_k > 0 else "auto"
            csv_fn = os.path.join(combo_save_dir, f"{combo_name}_faithfulness_results_k{k_fn}.csv")
            csv_out_paths.append(csv_fn)
            
            # Run faithfulness evaluation for this combo
            run_faithfulness(Path(saliency_dir), Path(ppgs_dir), Path(combo_save_dir), k_top_phon, csv_output_fn=csv_fn)
        
    # Print summary of results
    print_csv_results_prepool(csv_out_paths)
    

def print_csv_results_prepool(csv_paths):
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

def experiment_phon_in_mask(max_k=1000.0, starting_k=1, saliency_dir="", ppgs_dir="", output_dir=""
                            , save_pt=False, clean_temp=True):
    """This experiment computes faithfulness for different values of top_k. max_k defaults to the total number of phonemes."""
    
    full_df = pd.DataFrame()
    maxed_out = False
    
    if max_k <= 1.0:
        k_range = np.arange(0.05, max_k, 0.05)
        print(f"Will try {len(k_range)} top_k values from 5% to {int(max_k*100)}% of total phonemes")
    else:
        k_range = range(starting_k, int(max_k) + 1, 10)
        print(f"Will try {len(k_range)} top_k values from {starting_k} to {int(max_k)} phonemes")
        
    for top_k in tqdm(k_range, "Running top_k faithfulness experiments", leave=True):
        
        subdir_name = os.path.join(output_dir, f"topk_{top_k}") 
    
        # Generate PDSM output
        # Default preprocess_mode and pool_mode
        print(f"Generating PDSMs. Keeping {top_k} phonemes")
        k_top_phon, maxed_out = run_batch(
            saliency_dir,
            ppgs_dir,
            top_k,
            subdir_name,
            save_pt=save_pt
        )
        
        
        # Run faithfulness evaluation for this combo
        temp_df = run_faithfulness(Path(saliency_dir), Path(ppgs_dir), Path(subdir_name), k_top_phon, return_df=True)
        
        temp_df['top_k'] = top_k
        
        csv_fn_temp = os.path.join(subdir_name, f"TEMP_k{top_k}_ff_threshSum.csv")
        temp_df.to_csv(csv_fn_temp, index=False)
        
        full_df = pd.concat([full_df, temp_df], ignore_index=True)
        
        if maxed_out:
            print(f"Maxed out at top_k={top_k}. Ending experiment.")
            break
    
    if (maxed_out or top_k==max_k) and clean_temp:
        print("Experiment completed successfully. Cleaning up temp csv files...")
        pattern = os.path.join(output_dir, "**", "TEMP*.csv")
        for file_path in glob.iglob(pattern, recursive=True):
            print(f"Deleting: {file_path}")
            os.remove(file_path)
        
    csv_fn = os.path.join(output_dir, "experiment_topk_ff_threshSum.csv")
    full_df.to_csv(csv_fn, index=False)
    
    print_csv_results_topk(csv_fn)
    
def print_csv_results_topk(csv_path):
    df = pd.read_csv(csv_path)

    # Group by top_k and calculate mean and std
    summary_df = df.groupby('top_k').agg({
        'gradshap_ff': ['mean', 'std'],
        'pdsm_ff': ['mean', 'std']
    }).reset_index()

    # Flatten MultiIndex columns
    summary_df.columns = ['top_k', 'gradshap_mean', 'gradshap_std', 'pdsm_mean', 'pdsm_std']

    # Format the output
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps", default="data/saliencies")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files", default="data/ppg_out")
    parser.add_argument("--k", type=float, default=0.1, help="Number of phonemes to keep. Defaults to 1/10 of total")
    parser.add_argument("--threshold_val", type=float, default=0, help="Threshold value for preprocess_mode='threshold'")
    parser.add_argument("--pdsm_save_dir", type=str, default="data/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    parser.add_argument("--use_existing_pdsm",  action="store_true", help="Use existing PDSM .pt files instead of regenerating them")
    parser.add_argument("--experiment",
                        type=str,
                        choices=["single", "preprocess_pool", "phoneme_duration", "best_topk"],
                        default="single",
                        help="Select which experiment configuration to run."
                    )
    args = parser.parse_args()
    
    os.makedirs(args.pdsm_save_dir, exist_ok=True)
    
    if args.experiment == "preprocess_pool":
        experiment_preprocess_pool(args.M_path, args.X_p_path,  args.pdsm_save_dir, args.k, args.save_pt, args.use_existing_pdsm)
    elif args.experiment == "phoneme_duration":
        pass
    elif args.experiment == "best_topk":
        
        experiment_phon_in_mask(
            saliency_dir=args.M_path,
            ppgs_dir=args.X_p_path,
            output_dir=args.pdsm_save_dir,
            save_pt=args.save_pt,
            max_k=args.k
        )
    else: # experiment == "single"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.pdsm_save_dir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        k_top_phon, _ = run_batch(args.M_path, args.X_p_path, args.k, run_dir
                                , "threshold", "sum", args.threshold_val, save_pt=args.save_pt, save_csv=True, verbose=True)
        run_faithfulness(Path(args.M_path), Path( args.X_p_path), Path(run_dir), k_top_phon, csv_output_fn=run_dir / "faithfulness_results.csv")
    

if __name__ == "__main__":
    main()