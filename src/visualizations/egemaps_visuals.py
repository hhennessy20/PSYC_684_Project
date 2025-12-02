import math
from typing import List, Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. Box/violin-style feature comparison (cc vs cd)
# ============================================================

def plot_feature_boxplots_by_group(
    df: pd.DataFrame,
    feature_names: List[str],
    group_col: str = "group",
    level_name: str = "word",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Boxplots of selected features by group (e.g., cc vs cd).

    Parameters
    ----------
    df : DataFrame with at least [group_col] + feature_names
    feature_names : list of feature column names (eGeMAPS)
    group_col : column with 'cc' / 'cd'
    level_name : string for title annotation ("word" / "phoneme")

    Returns
    -------
    fig, axes : matplotlib Figure and Axes array
    """
    groups = sorted(df[group_col].dropna().unique())
    n_feats = len(feature_names)
    n_cols = min(3, n_feats)
    n_rows = math.ceil(n_feats / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, feat in enumerate(feature_names):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        vals = []
        labels = []
        for g in groups:
            arr = df.loc[df[group_col] == g, feat].dropna().values
            if arr.size > 0:
                vals.append(arr)
                labels.append(str(g))

        if vals:
            ax.boxplot(vals, labels=labels)

        ax.set_title(f"{feat} by {group_col} ({level_name}-level)")
        ax.set_ylabel(feat)

    # Hide unused axes
    for idx in range(n_feats, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    return fig, axes


# ============================================================
# 2. Pause-focused visualizations (duration + ratio)
# ============================================================

def compute_pause_stats_per_file(
    word_df: pd.DataFrame,
    pause_token: str = "PAUSE",
) -> pd.DataFrame:
    """
    Compute per-file pause stats:
        - total utterance duration
        - total pause duration
        - pause_ratio = pause_duration / total_duration
    """
    df = word_df.copy()
    is_pause = df["word"] == pause_token

    total_dur = df.groupby("file")["duration"].sum()
    pause_dur = df[is_pause].groupby("file")["duration"].sum()
    pause_dur = pause_dur.reindex(total_dur.index, fill_value=0.0)

    ratio = pause_dur / total_dur.replace(0, np.nan)
    file_group = df.groupby("file")["group"].first()

    stats = pd.DataFrame(
        {
            "file": total_dur.index,
            "group": file_group.reindex(total_dur.index),
            "total_duration": total_dur.values,
            "pause_duration": pause_dur.values,
            "pause_ratio": ratio.values,
        }
    )
    return stats


def plot_pause_duration_distributions(
    word_df: pd.DataFrame,
    pause_token: str = "PAUSE",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Two-panel figure:
      - Panel 1: histogram of pause segment durations (by group)
      - Panel 2: histogram of per-file pause ratio (by group)
    """
    pause_df = word_df[word_df["word"] == pause_token]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: raw pause durations
    ax1 = axes[0]
    for g in sorted(word_df["group"].dropna().unique()):
        vals = pause_df.loc[pause_df["group"] == g, "duration"].dropna().values
        if vals.size == 0:
            continue
        ax1.hist(vals, bins=30, alpha=0.5, label=str(g), density=True)
    ax1.set_title(f"Pause segment duration distribution ({pause_token})")
    ax1.set_xlabel("Pause duration (s)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Panel 2: per-file pause ratio
    stats = compute_pause_stats_per_file(word_df, pause_token=pause_token)
    ax2 = axes[1]
    for g in sorted(stats["group"].dropna().unique()):
        vals = stats.loc[stats["group"] == g, "pause_ratio"].dropna().values
        if vals.size == 0:
            continue
        ax2.hist(vals, bins=20, alpha=0.5, label=str(g), density=True)
    ax2.set_title("Per-file pause ratio (pause_time / total_time)")
    ax2.set_xlabel("Pause ratio")
    ax2.set_ylabel("Density")
    ax2.legend()

    fig.tight_layout()
    return fig, axes


# ============================================================
# 3. Acoustic fingerprint heatmap (effect sizes)
# ============================================================

def compute_effect_sizes_by_feature(
    df: pd.DataFrame,
    group_col: str = "group",
    group_a: str = "cc",
    group_b: str = "cd",
    exclude_cols: List[str] | None = None,
) -> pd.DataFrame:
    """
    Compute Cohen's d effect size (group_b - group_a) for each numeric feature.
    """
    if exclude_cols is None:
        exclude_cols = ["start", "end"]

    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype != "O" and c not in exclude_cols
    ]

    effects = []
    for feat in numeric_cols:
        a = df.loc[df[group_col] == group_a, feat].dropna().values
        b = df.loc[df[group_col] == group_b, feat].dropna().values
        if a.size < 2 or b.size < 2:
            continue

        mean_a, mean_b = a.mean(), b.mean()
        var_pooled = (
            (a.size - 1) * a.var(ddof=1)
            + (b.size - 1) * b.var(ddof=1)
        ) / (a.size + b.size - 2)

        if var_pooled <= 0:
            continue

        d = (mean_b - mean_a) / math.sqrt(var_pooled)
        effects.append((feat, d))

    eff_df = pd.DataFrame(effects, columns=["feature", "effect_size"])
    eff_df["abs_effect"] = eff_df["effect_size"].abs()
    eff_df.sort_values("abs_effect", ascending=False, inplace=True)
    return eff_df


def plot_acoustic_fingerprint_heatmap(
    df: pd.DataFrame,
    top_n: int = 30,
    group_col: str = "group",
    group_a: str = "cc",
    group_b: str = "cd",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of top-N features by effect size (group_b - group_a).
    """
    eff_df = compute_effect_sizes_by_feature(df, group_col, group_a, group_b)
    if eff_df.empty:
        raise ValueError("No numeric features to compute effect sizes.")

    eff_top = eff_df.head(top_n)

    fig, ax = plt.subplots(
        figsize=(6, max(4, 0.3 * len(eff_top)))
    )
    im = ax.imshow(
        eff_top[["effect_size"]].values,
        aspect="auto",
        cmap="coolwarm",
    )
    ax.set_yticks(range(len(eff_top)))
    ax.set_yticklabels(eff_top["feature"])
    ax.set_xticks([0])
    ax.set_xticklabels([f"{group_b} – {group_a} effect size"])

    cbar = plt.colorbar(im, ax=ax, label="Cohen's d")
    ax.set_title(f"Top {len(eff_top)} acoustic feature shifts ({group_b} vs {group_a})")
    fig.tight_layout()
    return fig, ax


# ============================================================
# 4. Word-level timeline (words + one feature over time)
# ============================================================

def plot_word_timeline(
    word_df: pd.DataFrame,
    file_id: str,
    feature: str = "F0semitoneFrom27.5Hz_sma3nz_amean",
    pause_token: str = "PAUSE",
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Timeline for a single file:
        - Top: words/PAUSE as colored bars over time
        - Bottom: selected acoustic feature at word midpoints
    """
    df_file = word_df[word_df["file"] == file_id].copy()
    if df_file.empty:
        raise ValueError(f"No rows for file_id={file_id}")
    df_file.sort_values("start", inplace=True)

    bars = []
    colors = []
    labels = []
    for _, row in df_file.iterrows():
        bars.append((row["start"], row["duration"]))
        labels.append(row["word"])
        colors.append(
            "lightblue" if row["word"] == pause_token else "lightgray"
        )

    fig, (ax_words, ax_feat) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True
    )

    # Word bars
    ax_words.broken_barh(bars, (0, 1), facecolors=colors, edgecolors="k")
    for (start, dur), txt in zip(bars, labels):
        if dur > 0.2 and txt != "PAUSE":  # avoid clutter for tiny segments
            ax_words.text(
                start + dur / 2,
                0.5,
                txt,
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,       # 🔹 vertical text
                rotation_mode="anchor"
            )
    ax_words.set_yticks([])
    ax_words.set_ylabel("Words / PAUSE")
    ax_words.set_title(f"Word timeline for {file_id}")

    # Feature line
    if feature not in df_file.columns:
        raise ValueError(f"Feature {feature} not in DataFrame.")
    midpoints = (df_file["start"].values + df_file["end"].values) / 2.0
    vals = df_file[feature].values
    ax_feat.plot(midpoints, vals, marker="o", linestyle="-")
    ax_feat.set_ylabel(feature)
    ax_feat.set_xlabel("Time (s)")

    fig.tight_layout()
    return fig, (ax_words, ax_feat)


# ============================================================
# 5. Generic word-importance bar chart
#    (you'll feed this with XGBoost SHAP or probabilities later)
# ============================================================

def plot_word_importance_bar(
    words: List[str],
    importances: np.ndarray,
    top_n: int = 20,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart of top-N words by importance.

    Parameters
    ----------
    words : list-like of word tokens
    importances : 1D array-like (same length as words)
    top_n : number of most important words to plot
    """
    if len(words) != len(importances):
        raise ValueError("words and importances must have same length.")

    importances = np.asarray(importances)
    words = np.asarray(words)

    idx = np.argsort(-np.abs(importances))[:top_n]
    top_words = words[idx]
    top_imps = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(top_words))))
    y = np.arange(len(top_words))
    ax.barh(y, top_imps)
    ax.set_yticks(y)
    ax.set_yticklabels(top_words)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top_words)} word importances")
    fig.tight_layout()
    return fig, ax


# ============================================================
# 6. PDSM vs XGBoost agreement scatter (generic)
# ============================================================

def plot_agreement_scatter(
    pdsm_energy: np.ndarray,
    xgb_importance: np.ndarray,
    title: str = "PDSM vs XGBoost importance",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of PDSM energy vs XGBoost importance for matched segments.
    """
    pdsm_energy = np.asarray(pdsm_energy)
    xgb_importance = np.asarray(xgb_importance)

    if pdsm_energy.shape != xgb_importance.shape:
        raise ValueError("pdsm_energy and xgb_importance must have same shape.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pdsm_energy, xgb_importance, alpha=0.5)
    ax.set_xlabel("PDSM energy")
    ax.set_ylabel("XGBoost importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ============================================================
# 7. Word-category scores (e.g., pause vs filler vs other)
#    This expects some score (e.g., mean SHAP contribution) per row.
# ============================================================

def default_word_category(word: str) -> str:
    """
    Very simple category mapping:
        - 'PAUSE' -> 'PAUSE'
        - 'uh', 'um', 'er', 'uhm' -> 'FILLER'
        - else -> 'OTHER'
    """
    w = str(word).lower()
    if w == "pause":
        return "PAUSE"
    if w in {"uh", "um", "er", "uhm"}:
        return "FILLER"
    return "OTHER"


def plot_word_category_scores(
    df: pd.DataFrame,
    score_col: str,
    category_fn: Callable[[str], str] = default_word_category,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Group mean score (e.g., AD contribution) by word category and group (cc/cd).

    df must have columns: 'word', 'group', score_col
    """
    if score_col not in df.columns:
        raise ValueError(f"{score_col} not found in df.")

    tmp = df[["word", "group", score_col]].dropna(subset=[score_col]).copy()
    tmp["category"] = tmp["word"].map(category_fn)
    agg = tmp.groupby(["group", "category"])[score_col].mean().reset_index()

    groups = sorted(agg["group"].unique())
    cats = sorted(agg["category"].unique())

    mat = np.zeros((len(cats), len(groups)))
    for i, cat in enumerate(cats):
        for j, g in enumerate(groups):
            val = agg.loc[
                (agg["group"] == g) & (agg["category"] == cat),
                score_col,
            ]
            mat[i, j] = val.iloc[0] if not val.empty else np.nan

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(cats))))
    bar_width = 0.35
    x = np.arange(len(cats))

    for j, g in enumerate(groups):
        offset = (j - (len(groups) - 1) / 2) * bar_width
        ax.bar(x + offset, mat[:, j], width=bar_width, label=str(g))

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(score_col)
    ax.set_title(f"Average {score_col} by word category and group")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ============================================================
# 8. Phoneme-level feature heatmap (AD–HC differences)
# ============================================================

def plot_phoneme_feature_heatmap(
    phon_df: pd.DataFrame,
    feature_names: List[str],
    min_count: int = 50,
    group_col: str = "group",
    group_a: str = "cc",
    group_b: str = "cd",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    For each phoneme with at least min_count tokens, compute:
        mean(feature) in group_b - mean(feature) in group_a
    for each feature in feature_names, and visualize as a heatmap.

    Parameters
    ----------
    phon_df : DataFrame from phoneme_egemaps_diarized.csv
    feature_names : list of eGeMAPS feature names
    min_count : minimum token count per phoneme to include
    """
    counts = phon_df["phoneme"].value_counts()
    keep_phonemes = counts[counts >= min_count].index.tolist()
    sub = phon_df[phon_df["phoneme"].isin(keep_phonemes)].copy()
    if sub.empty:
        raise ValueError("No phonemes meet the min_count threshold.")

    phonemes = sorted(keep_phonemes)
    mat = np.zeros((len(phonemes), len(feature_names)))

    for i, ph in enumerate(phonemes):
        df_ph = sub[sub["phoneme"] == ph]
        for j, feat in enumerate(feature_names):
            if feat not in df_ph.columns:
                raise ValueError(f"Feature {feat} not in phoneme DataFrame.")
            a = df_ph.loc[df_ph[group_col] == group_a, feat].dropna().values
            b = df_ph.loc[df_ph[group_col] == group_b, feat].dropna().values
            if a.size == 0 or b.size == 0:
                mat[i, j] = np.nan
            else:
                mat[i, j] = b.mean() - a.mean()

    fig, ax = plt.subplots(
        figsize=(1.5 * len(feature_names), max(4, 0.25 * len(phonemes)))
    )
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Mean({group_b}) - Mean({group_a})")

    ax.set_title("Phoneme-wise feature differences")
    fig.tight_layout()
    return fig, ax
