# CNN Model

Trains a CNN for Alzheimer's detection on mel spectrograms and generates GradSHAP saliency maps.

## Setup

```bash
conda env create -f src/pdsm/pdsm_environment.yml
conda activate pdsm
```

## Scripts

### Train CNN

```bash
python src/models/cnn/train_adress_cnn.py
```

Trains a CNN on diarized ADReSS audio. Saves the best model to `data/models/best_adress_cnn.pt`.

### Generate Saliency Maps

```bash
python src/models/cnn/interpret_gradshap_adress.py
```

Generates GradSHAP saliency maps for the top-k most confident predictions. Saves visualizations to `gradshap_val_plots/`.

## Output

- `data/models/best_adress_cnn.pt` - Trained model checkpoint
- `gradshap_val_plots/*.png` - Saliency map visualizations
