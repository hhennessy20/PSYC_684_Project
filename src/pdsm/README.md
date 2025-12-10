# Setup

Run:
conda env create -f pdsm_environment.yml

# Generate PPGs

Run 'batch_ppg.py' to generate a ppgs/ directory and save .pt files representing phoneme posteriorgrams for each sound file in your input directory.

# Generate Saliency Maps

Run generate_saliency_maps.py with desired parameters to generate a number of saliency maps from which to produce PDSMs

# Create PDSMs for a saliency map <-> PPG pair

Run pdsm.py and give it a PPG, saliency map, and spectrogram each corresponding to the same sound file