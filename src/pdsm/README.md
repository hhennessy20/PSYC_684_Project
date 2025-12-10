# Setup

Run:
```python
conda env create -f pdsm_environment.yml
```

Depending on your operating system you may have to fix dependencies. We had trouble with torchcodec and torchaudio on Windows. 

# Generating data files for PDSM

There are two ways to generate the necessary data files. You can use the faithfulness.py script to generate all in one go, or you can generate each type individually.

## 1. One shot



### Generate outputs for ALL diarized wav files



```python
python .\src\pdsm\faithfulness.py --spec_dir data\full_output_diarized\saliencies --gradshap_dir data\full_output_diarized\saliencies --ppg_dir data\full_output_diarized\ppg_out --pdsm_dir data\full_output_diarized\pdsm_out --output faithfulness_ALLWAVS.csv --val_split 1.0
```

This will also compute faithfulness across the entire set, but we want to ignore that and separate the data into cc and cd directories after it has been generated, then run the below commands.

### Compute faithfulness for CC group

```python
python .\src\pdsm\faithfulness.py --spec_dir data\full_output_diarized\saliencies\cc --gradshap_dir data\full_output_diarized\saliencies\cc --ppg_dir data\full_output_diarized\ppg_out\cc --pdsm_dir data\full_output_diarized\pdsm_out\cc --output faithfulness_cc.csv 
```

### Compute faithfulness for CD group

```python
python .\src\pdsm\faithfulness.py --spec_dir data\full_output_diarized\saliencies\cd --gradshap_dir data\full_output_diarized\saliencies\cd --ppg_dir data\full_output_diarized\ppg_out\cd --pdsm_dir data\full_output_diarized\pdsm_out\cd --output faithfulness_cd.csv
```

## 2. One-at-a-time

### Generate PPGs

Run 'batch_ppg.py' to generate a ppgs/ directory and save .pt files representing phoneme posteriorgrams for each sound file in your input directory.

### Generate Saliency Maps

Run generate_saliency_maps.py with desired parameters to generate a number of saliency maps from which to produce PDSMs

### Create PDSMs for a saliency map <-> PPG pair

Run pdsm.py and pass to it a PPG, saliency map, and spectrogram each corresponding to the same sound file

# Running Experiments

Once you have the data files generated for your WAV samples, use ```pdsm_experiments.py``` to run the experiments we discuss in our paper.

## Example Experiments

More example runs are outlined in the "pdsm_experiments.py" section of the example_runs.txt file

### Experiment for testing preprocess and pool functions

```python
python ./src/pdsm/pdsm_experiments.py --experiment preprocess_pool --M_path data\full_output_diarized\saliencies\cd --X_p_path data\full_output_diarized\ppg_out\cd --pdsm_save_dir src\pdsm\experiment_results\experiment_prepool_cd --save_pt 
```

### Experiment for finding best top_k

```python
python ./src/pdsm/pdsm_experiments.py --experiment best_topk --M_path data\full_output_diarized\saliencies\cd --X_p_path data\full_output_diarized\ppg_out\cd --pdsm_save_dir src\pdsm\experiment_results\experiment_bestTopK__fractional_cd --save_pt --k 1.0
```

For generating the corresponding figure:

```python
python ./src/pdsm/visualize.py best_topk --csv_input src\pdsm\experiment_results\experiment_bestTopK__fractional_cd\experiment_topk_ff_threshSum.csv
```