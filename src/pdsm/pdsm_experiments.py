
import argparse, os
from pathlib import Path

from batch_pdsm import run_batch

def experiment_preprocess_pool(saliency_map_paths, ppg_paths, output_dir, top_k, save_pt):
    # All 4 combinations
        preprocess_modes = ["threshold", "absolute"]
        pool_modes = ["sum", "mean"]

        for preprocess in preprocess_modes:
            for pool in pool_modes:
                combo_name = f"{preprocess}_{pool}"
                combo_save_dir = os.path.join(output_dir, combo_name)


                print(f"Running combo: {combo_name}")
                run_batch(
                    saliency_map_paths,
                    ppg_paths,
                    top_k,
                    combo_save_dir,
                    preprocess,
                    pool,
                    save_pt,
                    experiment_name=combo_name
                )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M_path", type=str, required=True, help="Path to saliency maps", default="data/saliencies")
    parser.add_argument("--X_p_path", type=str, required=True, help="Path to PPG .pt files", default="data/ppg_out")
    parser.add_argument("--k", type=int, default=0, help="Number of phonemes to keep. Defaults to 1/4 of total")
    parser.add_argument("--save_dir", type=str, default="data/pdsm_out", help="Where to save output")
    parser.add_argument("--save_pt",  action="store_true")
    parser.add_argument("--experiment",
                        type=str,
                        choices=["default", "preprocess_pool", "phoneme_duration", "test"],
                        default="default",
                        help="Select which experiment configuration to run."
                    )
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    saliency_map_paths = sorted(Path(args.M_path).glob("*_M.pt"))
    ppg_paths = sorted(Path(args.X_p_path).glob("*.pt"))
    
    if args.experiment == "preprocess_pool":
        experiment_preprocess_pool(saliency_map_paths, ppg_paths, args.save_dir, args.k, args.save_pt)
    else: # default options
        run_batch(saliency_map_paths, ppg_paths, args.k, args.save_dir, "threshold", "sum", args.save_pt)
    

if __name__ == "__main__":
    main()