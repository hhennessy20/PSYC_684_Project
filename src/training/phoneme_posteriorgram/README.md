# Dependencies

pip install -r ppg_requirements.txt

For MFA installation: https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html

# Training PPG

## Prereqs

ADReSS phoneme alignments from montreal forced aligner
1. first run adress_phoneme_alignment.py to get input TextGrids compatible with MFA
2. install mfa and run mfa align on command line using your input TextGrids

## Training

fine_tune_ppg.ipynb

1. Train baseline ppg using ppg\[train\] (module)
    a. Evaluate, store metrics
2. Fine tune ppg using your phoneme alignments from step 1.
    a. Evaluate, compare with baseline