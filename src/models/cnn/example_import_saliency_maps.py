from interpret_gradshap_adress import get_saliency_maps_for_pdsm

results = get_saliency_maps_for_pdsm(
    train_root=None,
    model_ckpt=None,
    val_split=0.2,
    k_most_confident=4,
    k_least_confident=4,
)

for ex in results:
    M = ex["M"]        # saliency map (n_mels, T)
    spec = ex["spec"]  # log-mel spectrogram (n_mels, T)
    label = ex["label"]
    p_ad = ex["p_ad"]

# Pass M into PDSM implementation...
