from pathlib import Path
import subprocess
import config


def standardize_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = config.TARGET_SR,
    channels: int = config.TARGET_CHANNELS,
    ffmpeg_bin: str = config.FFMPEG_BIN,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def standardize_all_in_dir(
    input_dir: Path,
) -> None:
    for wav in input_dir.glob("*.wav"):
        # overwrite input with standardized version (or change this logic if you want separate dir)
        tmp_out = wav.with_suffix(".tmp_16k_mono.wav")
        print(f"[AUDIO] Standardizing {wav.name}")
        standardize_audio(wav, tmp_out)
        wav.unlink()           # delete original
        tmp_out.rename(wav)    # rename standardized file to original name


if __name__ == "__main__":
    # Standardize both train and test audio in-place
    standardize_all_in_dir(config.TRAIN_AUDIO_DIR)
    standardize_all_in_dir(config.TEST_AUDIO_DIR)
