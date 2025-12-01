### Fine Tuning PPG Approach

1. Preprocess ADReSS dataset
    - Extract phoneme targets via Montreal Forced Aligner   <-- I'm Here
    - Convert into Mel-Spectrograms for PPG training
    - Create dataset compatible with PPG training module
2. Train PPG using custom dataset
3. Train PPG using provided dataset
4. Evaluate both and compare, collect metrics for final paper

### Get PPG Ready for PDSM (Not Started)

1. Create batch PPG inference.
2. Get PPGs for all data, stored locally.