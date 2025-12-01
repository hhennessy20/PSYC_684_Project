# Dependencies

pip install -r ppg_requirements.txt

For MFA installation: https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html

## MFA Usage

https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-pretrained

1. Download models
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa

2. Validate
mfa validate src/data/train/patient_audio_diarized/cc english_us_arpa english_us_arpa

3. Align (example)

mfa align src/data/train/patient_audio_diarized/cc english_us_arpa english_us_arpa src/data/train/patient_audio_diarized/cc/MFA_output