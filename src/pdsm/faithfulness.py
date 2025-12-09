# Faithfulness metric from Gupta et al. "Phoneme Discretized Saliency Maps"
# FF = f(X)_c - f(X * (1-M))_c
# Compares faithfulness of GradSHAP vs PDSM saliency maps

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODEL_CKPT_PATH, 
    ADRESS_DIARIZED_DIR, 
    SALIENCY_DIR, 
    PDSM_DIR,
    PPG_DIR,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "cnn"))
from train_adress_cnn import AudioCNN, N_MELS, SAMPLE_RATE, HOP_LENGTH, DURATION_SEC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate expected spectrogram window size (frames)
WINDOW_FRAMES = int((SAMPLE_RATE * DURATION_SEC) / HOP_LENGTH) + 1


def load_model(model_path, device=DEVICE):
    """
    Load AudioCNN with full checkpoint (including classifier).
    Returns the model and the expected input time dimension.
    """
    model = AudioCNN()
    model.to(device)

    state = torch.load(model_path, map_location=device, weights_only=True)
    
    # Figure out the expected input size from classifier weights
    # classifier.1 is the first Linear layer: Linear(flattened_features, 256)
    classifier_weight_key = "classifier.1.weight"
    if classifier_weight_key in state:
        expected_features = state[classifier_weight_key].shape[1]
        # After 3 conv+pool(2) layers with 128 channels and N_MELS=64:
        # Feature map: [128, N_MELS/8, T/8] = [128, 8, T/8]
        # Flattened: 128 * 8 * (T/8) = 1024 * T / 8
        # So: expected_features = 1024 * T / 8
        # Therefore: T = expected_features * 8 / 1024 = expected_features / 128
        expected_T = expected_features // 128
        print(f"  Checkpoint expects input with {expected_T} spectrogram frames")
    else:
        expected_T = WINDOW_FRAMES
    
    # Initialize classifier with correct size before loading weights
    dummy_input = torch.zeros(1, 1, N_MELS, expected_T).to(device)
    with torch.no_grad():
        model._init_fc(dummy_input)
    
    # Now load full state dict
    model.load_state_dict(state, strict=True)
    model.eval()
    
    return model, expected_T


def normalize_saliency_map(M, method="minmax"):
    """Normalize saliency map to [0, 1] range."""
    if method == "minmax":
        M_abs = torch.abs(M)
        M_min = M_abs.min()
        M_max = M_abs.max()
        if M_max - M_min > 1e-8:
            return (M_abs - M_min) / (M_max - M_min)
        else:
            return torch.zeros_like(M)
    elif method == "positive_only":
        M_pos = torch.clamp(M, min=0)
        M_max = M_pos.max()
        if M_max > 1e-8:
            return M_pos / M_max
        else:
            return torch.zeros_like(M)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_faithfulness(model, spec, saliency_map, window_frames, device=DEVICE, hop_fraction=0.5):
    """
    Compute faithfulness metric: FF = f(X)_c - f(X ⊙ (1-M))_c
    
    Uses sliding windows to handle full-length spectrograms, since the model
    was trained on fixed-size windows.
    
    Args:
        window_frames: Number of spectrogram frames the model expects
    """
    # Ensure 2D: [F, T]
    if spec.dim() == 3:
        spec = spec.squeeze(0)
    if saliency_map.dim() == 3:
        saliency_map = saliency_map.squeeze(0)
    
    F, T = spec.shape
    
    # If spectrogram fits in one window, process directly
    if T <= window_frames:
        # Pad if needed
        if T < window_frames:
            pad_size = window_frames - T
            spec = torch.nn.functional.pad(spec, (0, pad_size))
            saliency_map = torch.nn.functional.pad(saliency_map, (0, pad_size))
        
        spec_input = spec.unsqueeze(0).unsqueeze(0).to(device)
        saliency_input = saliency_map.unsqueeze(0).unsqueeze(0).to(device)
        
        inverted_mask = 1.0 - saliency_input
        spec_masked = spec_input * inverted_mask
        
        model.eval()
        with torch.no_grad():
            p_original = torch.sigmoid(model(spec_input)).item()
            p_masked = torch.sigmoid(model(spec_masked)).item()
        
        return p_original - p_masked, p_original, p_masked
    
    # Use sliding windows for longer spectrograms
    hop_frames = int(window_frames * hop_fraction)
    
    original_probs = []
    masked_probs = []
    
    inverted_mask_full = 1.0 - saliency_map
    spec_masked_full = spec * inverted_mask_full
    
    model.eval()
    with torch.no_grad():
        t = 0
        while t < T:
            t_end = min(t + window_frames, T)
            t_start = t
            
            # Handle last window
            if t_end - t_start < window_frames:
                t_start = max(0, T - window_frames)
                t_end = T
            
            # Extract windows
            spec_win = spec[:, t_start:t_end]
            spec_masked_win = spec_masked_full[:, t_start:t_end]
            
            # Pad if still too short
            if spec_win.shape[1] < window_frames:
                pad_size = window_frames - spec_win.shape[1]
                spec_win = torch.nn.functional.pad(spec_win, (0, pad_size))
                spec_masked_win = torch.nn.functional.pad(spec_masked_win, (0, pad_size))
            
            # Normalize each window
            spec_win = (spec_win - spec_win.mean()) / (spec_win.std() + 1e-8)
            spec_masked_win = (spec_masked_win - spec_masked_win.mean()) / (spec_masked_win.std() + 1e-8)
            
            # Add batch and channel dims: [1, 1, F, T]
            spec_input = spec_win.unsqueeze(0).unsqueeze(0).to(device)
            spec_masked_input = spec_masked_win.unsqueeze(0).unsqueeze(0).to(device)
            
            p_orig = torch.sigmoid(model(spec_input)).item()
            p_mask = torch.sigmoid(model(spec_masked_input)).item()
            
            original_probs.append(p_orig)
            masked_probs.append(p_mask)
            
            t += hop_frames
            if t_end >= T:
                break
    
    # Average across windows
    p_original = np.mean(original_probs)
    p_masked = np.mean(masked_probs)
    ff = p_original - p_masked
    
    return ff, p_original, p_masked


def generate_saliency_maps(output_dir, model_ckpt=None, device=DEVICE, val_split=0.2):
    """Generate spectrograms and GradSHAP saliency maps."""
    from interpret_gradshap_adress import get_saliency_maps_for_pdsm
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_ckpt = model_ckpt or MODEL_CKPT_PATH
    
    print(f"\nGenerating saliency maps (GradSHAP)")
    print(f"  Audio: {ADRESS_DIARIZED_DIR}")
    print(f"  Model: {model_ckpt}")
    print(f"  Output: {output_dir}")
    
    results = get_saliency_maps_for_pdsm(
        train_root=ADRESS_DIARIZED_DIR,
        model_ckpt=model_ckpt,
        val_split=val_split,
        k_most_confident=1000,
        k_least_confident=1000,
        hop_sec=2.0,
        device=device,
    )
    
    subject_ids = []
    seen_subjects = set()
    
    for item in results:
        subject = item["subject"]
        if subject in seen_subjects:
            continue
        seen_subjects.add(subject)
        subject_ids.append(subject)
        
        M_tensor = torch.tensor(item["M"])
        spec_tensor = torch.tensor(item["spec"])
        
        torch.save(M_tensor, output_dir / f"{subject}_M.pt")
        torch.save(spec_tensor, output_dir / f"{subject}_spec.pt")
        
        print(f"  Saved: {subject}_spec.pt, {subject}_M.pt")
    
    print(f"Generated saliency maps for {len(subject_ids)} subjects\n")
    return subject_ids


def generate_ppgs_for_subjects(subject_ids, output_dir, device=DEVICE):
    """Generate PPGs for specified subjects. Requires 'ppgs' library."""
    try:
        from batch_ppg import infer_ppg_from_wav, save_ppg
    except ImportError as e:
        print(f"ERROR: Could not import from batch_ppg.py: {e}")
        print("Install ppgs library: pip install ppgs")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating PPGs")
    print(f"  Output: {output_dir}")
    
    gpu_idx = 0 if device.type == "cuda" else None
    generated = []
    
    cc_dir = ADRESS_DIARIZED_DIR / "cc"
    cd_dir = ADRESS_DIARIZED_DIR / "cd"
    
    for subject in subject_ids:
        # Check if already exists (with subject ID naming)
        ppg_path = output_dir / f"{subject}.pt"
        if ppg_path.exists():
            print(f"  [EXISTS] {subject}.pt")
            generated.append(subject)
            continue
        
        # Find the audio file for this subject
        wav_path = None
        for search_dir in [cc_dir, cd_dir]:
            candidates = list(search_dir.glob(f"{subject}*.wav"))
            if candidates:
                wav_path = candidates[0]
                break
        
        if wav_path is None:
            print(f"  [SKIP] No audio found for {subject}")
            continue
        
        try:
            print(f"  Processing {subject}...")
            ppg = infer_ppg_from_wav(str(wav_path), gpu_idx)
            ppg = ppg.squeeze(0)
            
            # Save with subject ID as filename
            torch.save(ppg, ppg_path)
            print(f"  Saved: {subject}.pt")
            generated.append(subject)
            
        except Exception as e:
            print(f"  [ERROR] {subject}: {e}")
    
    print(f"Generated PPGs for {len(generated)} subjects\n")
    return generated


def generate_pdsms(subject_ids, saliency_dir, ppg_dir, output_dir, top_k=0.25):
    """Generate PDSMs for specified subjects. Requires 'ppgs' library."""
    try:
        from pdsm import phoneme_discretization
    except ImportError as e:
        print(f"ERROR: Could not import from pdsm.py: {e}")
        print("Install ppgs library: pip install ppgs")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saliency_dir = Path(saliency_dir)
    ppg_dir = Path(ppg_dir)
    
    print(f"\nGenerating PDSMs")
    print(f"  Saliency maps: {saliency_dir}")
    print(f"  PPGs: {ppg_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Top-k: {top_k if top_k > 0 else 'auto'}")
    
    generated = []
    
    for subject in subject_ids:
        pdsm_path = output_dir / f"{subject}.pt"
        
        if pdsm_path.exists():
            print(f"  [EXISTS] {subject}.pt")
            generated.append(subject)
            continue
        
        M_path = saliency_dir / f"{subject}_M.pt"
        ppg_path = ppg_dir / f"{subject}.pt"
        
        if not M_path.exists():
            print(f"  [SKIP] No saliency map for {subject}")
            continue
        if not ppg_path.exists():
            print(f"  [SKIP] No PPG for {subject}")
            continue
        
        try:
            M = torch.load(M_path, weights_only=True)
            X_p = torch.load(ppg_path, weights_only=True)
            
            # Align time dimensions if needed
            T_M = M.shape[-1]
            T_ppg = X_p.shape[-1]
            
            if T_M != T_ppg:
                X_p = torch.nn.functional.interpolate(
                    X_p.unsqueeze(0).unsqueeze(0),
                    size=(X_p.shape[0], T_M),
                    mode='nearest'
                ).squeeze(0).squeeze(0)
            
            print(f"  Processing {subject}...")
            Mc, _, _ = phoneme_discretization(M, X_p, k=top_k)
            
            torch.save(Mc, pdsm_path)
            print(f"  Saved: {subject}.pt")
            generated.append(subject)
            
        except Exception as e:
            print(f"  [ERROR] {subject}: {e}")
    
    print(f"Generated PDSMs for {len(generated)} subjects\n")
    return generated


def compute_faithfulness_for_subject(
    subject_id,
    spec_dir,
    gradshap_dir,
    pdsm_dir,
    model,
    window_frames,
    device=DEVICE,
):
    """Compute faithfulness for both GradSHAP and PDSM for a single subject."""
    spec_path = Path(spec_dir) / f"{subject_id}_spec.pt"
    gradshap_path = Path(gradshap_dir) / f"{subject_id}_M.pt"
    
    results = {"subject": subject_id}
    
    if not spec_path.exists():
        return None
    
    spec = torch.load(spec_path, map_location=device, weights_only=True)
    
    # GradSHAP faithfulness
    if gradshap_path.exists():
        M_gradshap = torch.load(gradshap_path, map_location=device, weights_only=True)
        M_gradshap_norm = normalize_saliency_map(M_gradshap, method="minmax")
        
        ff_gs, p_orig_gs, p_masked_gs = compute_faithfulness(
            model, spec, M_gradshap_norm, window_frames, device
        )
        results["gradshap"] = {
            "faithfulness": ff_gs,
            "p_original": p_orig_gs,
            "p_masked": p_masked_gs,
        }
    else:
        results["gradshap"] = None
    
    # PDSM faithfulness
    if pdsm_dir:
        pdsm_path = Path(pdsm_dir) / f"{subject_id}.pt"
        if pdsm_path.exists():
            M_pdsm = torch.load(pdsm_path, map_location=device, weights_only=True)
            M_pdsm = M_pdsm.float()
            
            ff_pdsm, p_orig_pdsm, p_masked_pdsm = compute_faithfulness(
                model, spec, M_pdsm, window_frames, device
            )
            results["pdsm"] = {
                "faithfulness": ff_pdsm,
                "p_original": p_orig_pdsm,
                "p_masked": p_masked_pdsm,
            }
        else:
            results["pdsm"] = None
    else:
        results["pdsm"] = None
    
    return results


def print_results(all_results, include_pdsm=True):
    """Print formatted faithfulness results."""
    print("\n" + "-" * 60)
    if include_pdsm:
        print("Faithfulness: GradSHAP vs PDSM")
    else:
        print("Faithfulness: GradSHAP")
    print("-" * 60)
    
    if include_pdsm:
        print(f"{'Subject':<10} {'GradSHAP':>12} {'PDSM':>12} {'Diff':>12}")
    else:
        print(f"{'Subject':<10} {'GradSHAP':>12} {'P(AD) orig':>12} {'P(AD) mask':>12}")
    print("-" * 60)
    
    gs_scores = []
    pdsm_scores = []
    
    for r in all_results:
        sid = r["subject"]
        gs = r.get("gradshap")
        pdsm = r.get("pdsm")
        
        gs_ff = gs["faithfulness"] if gs else None
        
        if include_pdsm:
            pdsm_ff = pdsm["faithfulness"] if pdsm else None
            gs_str = f"{gs_ff:.4f}" if gs_ff is not None else "N/A"
            pdsm_str = f"{pdsm_ff:.4f}" if pdsm_ff is not None else "N/A"
            
            if gs_ff is not None and pdsm_ff is not None:
                delta = pdsm_ff - gs_ff
                delta_str = f"{delta:+.4f}"
                gs_scores.append(gs_ff)
                pdsm_scores.append(pdsm_ff)
            else:
                delta_str = "N/A"
                if gs_ff is not None:
                    gs_scores.append(gs_ff)
            
            print(f"{sid:<10} {gs_str:>12} {pdsm_str:>12} {delta_str:>12}")
        else:
            if gs_ff is not None:
                gs_scores.append(gs_ff)
                print(f"{sid:<10} {gs_ff:>12.4f} {gs['p_original']:>12.4f} {gs['p_masked']:>12.4f}")
            else:
                print(f"{sid:<10} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    
    print("-" * 60)
    
    if gs_scores:
        gs_mean = np.mean(gs_scores)
        print(f"{'MEAN':<10} {gs_mean:>12.4f}", end="")
        
        if include_pdsm and pdsm_scores:
            pdsm_mean = np.mean(pdsm_scores)
            diff_mean = pdsm_mean - gs_mean
            print(f" {pdsm_mean:>12.4f} {diff_mean:>+12.4f}")
            print(f"{'STD':<10} {np.std(gs_scores):>12.4f} {np.std(pdsm_scores):>12.4f}")
        else:
            print()
            print(f"{'STD':<10} {np.std(gs_scores):>12.4f}")
    
    print("-" * 60)
    
    
    
def run_faithfulness(saliency_dir, ppg_dir, pdsm_dir, top_k, gradshap_dir="", no_generate=False
                     , skip_pdsm=False, model_path=MODEL_CKPT_PATH,  csv_output_fn="", val_split=0.2, return_df=False):
    import random
    subject_ids = []
    
    if not gradshap_dir:
        gradshap_dir = saliency_dir
    
    # Step 1: Check/generate saliency maps
    existing_specs = list(saliency_dir.glob("*_spec.pt")) if saliency_dir.exists() else []
    
    if existing_specs:
        subject_ids = [f.stem.replace("_spec", "") for f in existing_specs]
        print(f"\nFound {len(subject_ids)} existing saliency maps in {saliency_dir}")
    elif not no_generate:
        print(f"\nNo saliency maps found. Generating...")
        subject_ids = generate_saliency_maps(saliency_dir, model_path, DEVICE, val_split)
    else:
        print(f"\nNo saliency maps found in {saliency_dir}")
        return
    
    if not subject_ids:
        print("No subjects to process.")
        return
    
    # Step 2: Generate PPGs and PDSMs if needed (and not skipping PDSM)
    if not skip_pdsm and not no_generate:
        # Check for existing PPGs
        existing_ppgs = list(ppg_dir.glob("*.pt")) if ppg_dir.exists() else []
        existing_ppg_subjects = {f.stem for f in existing_ppgs}
        missing_ppg_subjects = [s for s in subject_ids if s not in existing_ppg_subjects]
        
        if missing_ppg_subjects:
            print(f"\n{len(missing_ppg_subjects)} subjects missing PPGs. Generating...")
            generate_ppgs_for_subjects(missing_ppg_subjects, ppg_dir, DEVICE)
        
        # Check for existing PDSMs
        existing_pdsms = list(pdsm_dir.glob("*.pt")) if pdsm_dir.exists() else []
        existing_pdsm_subjects = {f.stem for f in existing_pdsms}
        missing_pdsm_subjects = [s for s in subject_ids if s not in existing_pdsm_subjects]
        
        if missing_pdsm_subjects:
            print(f"\n{len(missing_pdsm_subjects)} subjects missing PDSMs. Generating...")
            generate_pdsms(missing_pdsm_subjects, saliency_dir, ppg_dir, pdsm_dir, top_k)
    
    # Step 3: Load model for faithfulness computation
    print(f"\nLoading model from {model_path}")
    model, window_frames = load_model(model_path, DEVICE)
    print(f"  Using window size: {window_frames} frames")
    
    # Compute faithfulness for each subject
    print(f"\nComputing faithfulness...")
    
    all_results = []
    for sid in subject_ids:
        result = compute_faithfulness_for_subject(
            sid, saliency_dir, gradshap_dir, pdsm_dir, model, window_frames, DEVICE,
        )
        if result:
            all_results.append(result)
            gs = result.get("gradshap")
            pdsm = result.get("pdsm")
            
            status = f"{sid}: "
            if gs:
                status += f"GS={gs['faithfulness']:.4f}"
            if pdsm:
                status += f" PDSM={pdsm['faithfulness']:.4f}"
            print(status)
    
    # Step 5: Print results
    include_pdsm = pdsm_dir is not None and any(r.get("pdsm") for r in all_results)
    print_results(all_results, include_pdsm=include_pdsm)
    
    # Optionally save to CSV
    if csv_output_fn:
        import csv
        with open(csv_output_fn, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "subject", 
                "gradshap_ff", "gradshap_p_orig", "gradshap_p_masked",
                "pdsm_ff", "pdsm_p_orig", "pdsm_p_masked"
            ])
            for r in all_results:
                gs = r.get("gradshap") or {}
                pdsm = r.get("pdsm") or {}
                writer.writerow([
                    r["subject"],
                    gs.get("faithfulness", ""),
                    gs.get("p_original", ""),
                    gs.get("p_masked", ""),
                    pdsm.get("faithfulness", ""),
                    pdsm.get("p_original", ""),
                    pdsm.get("p_masked", ""),
                ])
        print(f"\nResults saved to {csv_output_fn}")
    elif return_df:
        import pandas as pd
        data = []
        for r in all_results:
            gs = r.get("gradshap") or {}
            pdsm = r.get("pdsm") or {}
            data.append({
                "subject": r["subject"],
                "gradshap_ff": gs.get("faithfulness", None),
                "gradshap_p_orig": gs.get("p_original", None),
                "gradshap_p_masked": gs.get("p_masked", None),
                "pdsm_ff": pdsm.get("faithfulness", None),
                "pdsm_p_orig": pdsm.get("p_original", None),
                "pdsm_p_masked": pdsm.get("p_masked", None),
            })
        df = pd.DataFrame(data)
        return df


def main():
    parser = argparse.ArgumentParser(
        description="Compute faithfulness metric (Gupta et al.) for GradSHAP vs PDSM. "
                    "Auto-generates all components using existing project scripts if not provided."
    )
    parser.add_argument(
        "--spec_dir", default=None,
        help=f"Directory with spectrograms (*_spec.pt). Default: {SALIENCY_DIR}",
    )
    parser.add_argument(
        "--gradshap_dir", default=None,
        help=f"Directory with GradSHAP maps (*_M.pt). Default: {SALIENCY_DIR}",
    )
    parser.add_argument(
        "--ppg_dir", default=None,
        help=f"Directory with PPGs (*.pt). Default: {PPG_DIR}",
    )
    parser.add_argument(
        "--pdsm_dir", default=None,
        help=f"Directory with PDSMs (*.pt). Default: {PDSM_DIR}",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Model checkpoint. Default: {MODEL_CKPT_PATH}",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV file for results.",
    )
    parser.add_argument(
        "--top_k", type=int, default=0.25,
        help="Top-k phonemes for PDSM. Default: 1/4 of boundaries.",
    )
    parser.add_argument(
        "--skip_pdsm", action="store_true",
        help="Skip PDSM generation and comparison (GradSHAP only).",
    )
    parser.add_argument(
        "--no_generate", action="store_true",
        help="Don't auto-generate missing components, use only existing files.",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Top-k phonemes for PDSM. Default: 1/4 of boundaries.",
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    
    model_path = args.model or MODEL_CKPT_PATH
    
    # Set default directories
    saliency_dir = Path(args.spec_dir or SALIENCY_DIR)
    gradshap_dir = Path(args.gradshap_dir or SALIENCY_DIR)
    ppg_dir = Path(args.ppg_dir or PPG_DIR)
    pdsm_dir = Path(args.pdsm_dir or PDSM_DIR) if not args.skip_pdsm else None
    
    run_faithfulness(
        saliency_dir,
        ppg_dir,
        pdsm_dir,
        args.top_k,
        gradshap_dir,
        args.no_generate,
        args.skip_pdsm,
        model_path,
        args.output,
        args.val_split
    )
    


if __name__ == "__main__":
    main()
