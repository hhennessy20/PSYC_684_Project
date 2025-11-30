from interpret_gradshap_adress import get_saliency_maps_for_pdsm

results = get_saliency_maps_for_pdsm(
    train_root="train/Normalised_audio-chunks",
    model_ckpt="best_adress_cnn.pt",
    val_split=0.2,
    batch_size=32,
    num_examples=4,
)

for ex in results:
    M = ex["M"]        # (n_mels, T) — saliency map for PDSM
    spec = ex["spec"]  # (n_mels, T) — normalized log-mel
    label = ex["label"]
    p_ad = ex["p_ad"]

# Pass M into PDSM implementation...
