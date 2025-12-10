# Phoneme Posteriorgram (PPG)

Generates phoneme posteriorgrams using Montreal Forced Aligner (MFA) for phoneme-level analysis.

## Setup

```bash
pip install -r src/models/phoneme_posteriorgram/ppg_requirements.txt
```

### MFA Installation

Follow: https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html

Download required models:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## MFA Usage

### 1. Validate

```bash
mfa validate data/train_Data/Diarized_full_wave_enhanced_audio/cc english_us_arpa english_us_arpa
```

### 2. Align

```bash
mfa align data/train_Data/Diarized_full_wave_enhanced_audio/cc english_us_arpa english_us_arpa data/MFA_output_cc_diarized
mfa align data/train_Data/Diarized_full_wave_enhanced_audio/cd english_us_arpa english_us_arpa data/MFA_output_cd_diarized
```

## Output

- `MFA/*.TextGrid` - Phoneme alignment files
- `MFA/*.csv` - Alignment analysis
