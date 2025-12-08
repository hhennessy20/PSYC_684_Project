#
# For docs: https://github.com/interactiveaudiolab/ppgs/tree/master
#

from pathlib import Path
import os, sys

import torch
import soundfile as sf
import ppgs
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


PPG_SAMPLE_RATE = ppgs.SAMPLE_RATE


def plot_ppg(ppgs_tensor, wav_filename, out_dir):
    """Plot PPG tensor to file."""
    ppg = ppgs_tensor.float().numpy()

    plt.figure(figsize=(18, 18))
    plt.imshow(ppg, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Posterior Probability')
    plt.xlabel('Time Frame')
    plt.ylabel('Phoneme Index')
    plt.yticks(np.arange(len(ppgs.PHONEMES)), ppgs.PHONEMES)
    plt.title('Phonetic Posteriorgram (PPG)')
    plt.savefig(f"{out_dir}/ppg_plot_{wav_filename}.png")
    plt.close()
    

def load_audio_soundfile(wav_path):
    """Load audio file and convert to tensor."""
    audio_np, sr = sf.read(str(wav_path), dtype='float32')
    
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
    
    if sr != PPG_SAMPLE_RATE:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, PPG_SAMPLE_RATE)
        audio_tensor = resampler(audio_tensor)
    
    return audio_tensor


def generate_ppgs(wav_dir, output_dir, gpu_idx=None, savePlot=False):
    """Generate PPGs for all WAV files in directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(wav_dir):
        files = [f for f in files if f.lower().endswith(".wav")]
        for file in tqdm(files, desc="Processing WAVs"):
            wav_path = os.path.join(root, file)
            ppg = infer_ppg_from_wav(wav_path, gpu_idx)
            save_ppg(ppg.squeeze(0), wav_path, output_dir, savePlot)


def infer_ppg_from_wav(wav_path, gpu_idx=None):
    """Infer PPG from a WAV file."""
    base = os.path.splitext(os.path.basename(wav_path))[0]
    tqdm.write(f"Processing {base}")
    
    audio = load_audio_soundfile(wav_path)
    pp = ppgs.from_audio(audio, PPG_SAMPLE_RATE, gpu=gpu_idx)
    
    return pp


def save_ppg(ppg, wav_path, out_dir, savePlot=False):
    """Save PPG tensor to file."""
    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(out_dir, "PPG_" + base + ".pt")
    
    base_out = os.path.splitext(os.path.basename(out_path))[0]
    tqdm.write(f"Saving {base_out}")
    
    torch.save(ppg, out_path)
    
    if savePlot:
        plot_ppg(ppg, base, out_dir)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir_root",
        default="../data/train/Normalised_audio-chunks",
        help="Directory with WAV files. Must have subfolders 'cc' and 'cd'.",
    )
    parser.add_argument(
        "--output_dir",
        default="src/pdsm/ppgs",
        help="Directory to save PPG outputs.",
    )
    args = parser.parse_args()
    
    parent, folder = os.path.split(args.wav_dir_root)
    wav_dir_cd = os.path.join(args.wav_dir_root, "cd")
    wav_dir_cc = os.path.join(args.wav_dir_root, "cc")
    if not os.path.isdir(wav_dir_cd) or not os.path.isdir(wav_dir_cc):
        raise FileNotFoundError(f"Input directory must contain both 'cc' and 'cd' directories.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir_cd = os.path.join(args.output_dir, folder + "/cd")
    os.makedirs(output_dir_cd, exist_ok=True)
    generate_ppgs(wav_dir_cd, output_dir_cd)
    
    output_dir_cc = os.path.join(args.output_dir, folder + "/cc")
    os.makedirs(output_dir_cc, exist_ok=True)
    generate_ppgs(wav_dir_cc, output_dir_cc)
    


if __name__=="__main__":
    main()
