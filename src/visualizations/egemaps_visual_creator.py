import pandas as pd
import matplotlib.pyplot as plt
from src.visualizations.egemaps_visuals import (
    plot_feature_boxplots_by_group,
    plot_pause_duration_distributions,
    plot_acoustic_fingerprint_heatmap,
    plot_word_timeline,
    plot_phoneme_feature_heatmap,

)

from src.config import (
    FIG_DIR
)

def save_fig(fig, name: str):
    out_path = FIG_DIR / name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved figure -> {out_path}")


if __name__ == "__main__":

    word_df = pd.read_csv("data/features/word_egemaps_diarized.csv")
    phon_df = pd.read_csv("data/features/phoneme_egemaps_diarized.csv")

    feat_list = [
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "loudness_sma3_amean",
        "MeanVoicedSegmentLengthSec",
    ]
    fig, axes = plot_feature_boxplots_by_group(word_df, feat_list, level_name="word")
    save_fig(fig, "boxplots_word_features_cc_vs_cd.png")

    # 2) pause distributions
    fig, axes = plot_pause_duration_distributions(word_df, pause_token="PAUSE")
    save_fig(fig, "pause_distributions_cc_vs_cd.png")

    # 3) acoustic fingerprint heatmap (word-level)
    fig, ax = plot_acoustic_fingerprint_heatmap(word_df, top_n=25)
    save_fig(fig, "acoustic_fingerprint_heatmap_word_level.png")

    # 4) word timeline for one example file
    example_file = word_df["file"].iloc[6120]
    print(example_file)
    fig, (ax_w, ax_f) = plot_word_timeline(
        word_df,
        example_file,
        feature="F0semitoneFrom27.5Hz_sma3nz_amean",
        pause_token="PAUSE",
    )
    save_fig(fig, f"word_timeline_{example_file}.png")

    # 5) phoneme feature heatmap (choose a few numeric features)
    numeric_cols = [c for c in phon_df.columns if phon_df[c].dtype != "O"]
    # pick a small set of features (adjust indices to something sensible in your file)
    phon_feat_list = numeric_cols[5:10]
    fig, ax = plot_phoneme_feature_heatmap(
        phon_df,
        feature_names=phon_feat_list,
        min_count=30,
    )
    save_fig(fig, "phoneme_feature_heatmap.png")


