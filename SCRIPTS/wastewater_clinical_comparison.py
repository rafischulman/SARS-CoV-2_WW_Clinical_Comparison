#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wastewater_clinical_comparison.py

Generates all figures and statistical results for:
"Wastewater genomic surveillance detects SARS-CoV-2 variants earlier and more
comprehensively than clinical sequencing: a statewide systems comparison in
New York, 2023-2025"

Figures are written to ../FIGURES/
Tabular results are written to ../RESULTS/

Input data required (see Data paths section below):
  - NY State COVID-19 testing data (NYS DOH)
  - GISAID clinical sequencing metadata
  - BIOBOT wastewater sequencing and concentration data
  - NYS sewershed metadata
  - Lineage aggregation key (lineage-map.csv)
  - NY State county shapefile
  - NY State locality hierarchy
  - County population estimates
"""

# ============================================================
# Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import cohen_kappa_score

# ============================================================
# Global figure style
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# ============================================================
# Configuration
# ============================================================

start_date = "2023-01-01"
end_date   = "2025-10-01"

NY_POP = 20_201_249          # 2020 Census NY State population

WW_CUTOFF     = 0.01         # Primary wastewater relative abundance threshold
WW_CUTOFFS    = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]  # Sensitivity analysis thresholds
NUM_LINEAGES  = 10           # Number of top lineages for timeline and emergence figures
ROLLING_WINDOW = 7           # Days for rolling average smoothing

MERGE_CLINICAL_GROUPS = True  # Aggregate clinical lineages to parent groups
MERGE_WW_GROUPS       = True  # Aggregate wastewater lineages to parent groups
MERGE_SUM             = True  # Sum variant_pct within aggregated groups per sample

DPI = 600

FIGURES_DIR = "../FIGURES"
RESULTS_DIR = "../RESULTS"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Data paths
# ============================================================

NY_CASES_FILE    = "../DATA/New_York_State_Statewide_COVID-19_Testing_20251207.csv"
# Derived aggregate files produced by prepare_clinical_aggregates.py from
# the raw GISAID export -- see that script's docstring for why aggregates
# are used instead of the raw per-specimen file, and for the EPI_SET that
# identifies the exact clinical record set analyzed.
CLINICAL_LINEAGE_COUNTS_FILE = "../DATA/clinical_lineage_daily_counts.csv.gz"
CLINICAL_COUNTY_COUNTS_FILE  = "../DATA/clinical_county_counts.csv"
WW_SEQ_FILE      = "../DATA/sars2-genetic-sequencing_20251128.csv"
WW_META_FILE     = "../DATA/concentration_20251130.csv"
SEWERSHEDS_FILE  = "../DATA/nys-wws-sewersheds_20251130.csv"
LINEAGE_MAP_FILE = "../DATA/lineage-map.csv"
SHAPEFILE        = "../DATA/shapefile/Counties_Shoreline.shp"
COUNTY_POP_FILE  = "../DATA/county_population_metadata.csv"

# ============================================================
# Utility functions
# ============================================================

def save_fig(fig, path):
    """Save figure at publication resolution with white background."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")


def rolling_mean(series, window=7):
    """Centered rolling mean with minimum 1 period to avoid edge NaNs."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def annotate_correlation(ax, pearson_r, spearman_r, loc=(0.02, 0.98)):
    """Annotate an axes with Pearson r and Spearman rho."""
    text = f"Pearson r = {pearson_r:.2f}\nSpearman ρ = {spearman_r:.2f}"
    ax.text(
        loc[0], loc[1], text,
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
    )


def plot_fd_histogram(ax, data, xlabel, bar_color, ylabel="Number of counties"):
    """
    Histogram using Freedman-Diaconis bin width selection.
    Falls back to 10 bins if IQR is zero.
    """
    data = data.dropna().values
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(data) ** (-1 / 3))
    bins = 10 if bin_width <= 0 else int(np.ceil((data.max() - data.min()) / bin_width))
    ax.hist(data, bins=bins, color=bar_color, alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")


def plot_timeline_panel(ax, dates, counts, moving_average, ylabel,
                        start_date, end_date, bar_color, line_color):
    """
    Single panel for the three-part sampling timeline figure.
    Plots daily bars, 7-day rolling average, and median reference line.
    """
    ax.bar(dates, counts, color=bar_color, alpha=0.5, width=1.0, label="Daily")
    ax.plot(moving_average.index, moving_average, color=line_color,
            linewidth=2.0, label="7-day average")
    ax.axhline(counts.median(), color="black", linestyle=":", linewidth=1.2, label="Median")
    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    data_in_range = counts.loc[(dates >= start_date) & (dates <= end_date)]
    ax.set_ylim(0, data_in_range.max() * 1.1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")


def select_top_lineages_by_peak(df, num_lineages, window):
    """
    Select the top N most common lineages from clinical data, then sort
    them by the date of their smoothed peak abundance. Returns an ordered
    list of lineage names from earliest to latest peak.
    """
    lineage_counts = df["pangolin_lineage"].value_counts()
    top_lineages   = lineage_counts.head(num_lineages).index.tolist()
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    peak_dates = {}
    for lin in top_lineages:
        series = (
            df[df["pangolin_lineage"] == lin]
            .groupby("date").size()
            .reindex(full_range, fill_value=0)
        )
        smoothed = rolling_mean(series, window)
        peak_dates[lin] = smoothed.idxmax()
    return sorted(peak_dates, key=lambda x: peak_dates[x])


def plot_combined_lineage_timelines_weighted(
    clinical_df,
    ww_df,
    lineages,
    lineage_colors,
    start_date,
    end_date,
    output_path,
    ww_cutoff=0.0,
):
    """
    Two-panel figure showing lineage-level temporal dynamics for the top N lineages.

    Clinical panel: daily counts smoothed with a rolling average.
    Wastewater panel: daily sum of per-sample relative abundance (variant_pct)
    for detections above ww_cutoff, smoothed with a rolling average.
    Each lineage is plotted in its assigned color from the lineage key.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    clinical_df = clinical_df[
        (clinical_df["date"] >= start_date) & (clinical_df["date"] <= end_date)
    ]
    ww_df = ww_df[
        (ww_df["sample_collect_date"] >= start_date) &
        (ww_df["sample_collect_date"] <= end_date)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for lin in lineages:
        clin_series = (
            clinical_df[clinical_df["pangolin_lineage"] == lin]
            .groupby("date").size()
            .reindex(full_range, fill_value=0)
        )
        axes[0].plot(full_range, rolling_mean(clin_series),
                     color=lineage_colors.get(lin, "#999999"), linewidth=1.3, alpha=0.9)
    axes[0].set_ylabel("Clinical\ncases")
    axes[0].set_title("Clinical lineage timelines (rolling average)")
    axes[0].grid(False)

    for lin in lineages:
        ww_series = (
            ww_df[(ww_df["variant"] == lin) & (ww_df["variant_pct"] > ww_cutoff)]
            .groupby("sample_collect_date")["variant_pct"]
            .sum()
            .reindex(full_range, fill_value=0)
        )
        axes[1].plot(full_range, rolling_mean(ww_series),
                     color=lineage_colors.get(lin, "#999999"), linewidth=1.3, alpha=0.9)
    axes[1].set_ylabel("Wastewater\ncounts (weighted)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(False)

    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)


def plot_combined_lineage_timelines_weighted_multi_cutoff(
    clinical_df,
    ww_df,
    lineages,
    lineage_colors,
    start_date,
    end_date,
    output_path,
    ww_cutoffs,
):
    """
    Supplementary version of plot_combined_lineage_timelines_weighted: a
    single clinical panel on top, followed by one wastewater panel per
    threshold in ww_cutoffs stacked below it -- same row-stacking layout
    as the lineage_emergence_multi_cutoff figure (n_rows = 1 + len(cutoffs),
    sharex, clinical row first). Lets a reader see, cutoff by cutoff, how
    the shape of each lineage's wastewater timeline changes with
    sensitivity, rather than only its first-detection date.
    """
    full_range = pd.date_range(start=start_date, end=end_date, freq="D")
    clinical_df = clinical_df[
        (clinical_df["date"] >= start_date) & (clinical_df["date"] <= end_date)
    ]
    ww_df = ww_df[
        (ww_df["sample_collect_date"] >= start_date) &
        (ww_df["sample_collect_date"] <= end_date)
    ]

    n_rows = 1 + len(ww_cutoffs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.2 * n_rows), sharex=True)

    for lin in lineages:
        clin_series = (
            clinical_df[clinical_df["pangolin_lineage"] == lin]
            .groupby("date").size()
            .reindex(full_range, fill_value=0)
        )
        axes[0].plot(full_range, rolling_mean(clin_series),
                     color=lineage_colors.get(lin, "#999999"), linewidth=1.3, alpha=0.9)
    axes[0].set_ylabel("Clinical\ncases")
    axes[0].set_title("Clinical lineage timelines (rolling average)")
    axes[0].grid(False)

    for i, cutoff in enumerate(ww_cutoffs, start=1):
        ax = axes[i]
        for lin in lineages:
            ww_series = (
                ww_df[(ww_df["variant"] == lin) & (ww_df["variant_pct"] > cutoff)]
                .groupby("sample_collect_date")["variant_pct"]
                .sum()
                .reindex(full_range, fill_value=0)
            )
            ax.plot(full_range, rolling_mean(ww_series),
                    color=lineage_colors.get(lin, "#999999"), linewidth=1.3, alpha=0.9)
        ax.set_ylabel(f"WW ≥ {cutoff:.2f}\n(weighted)")
        ax.grid(False)

    axes[-1].tick_params(axis="x", rotation=45)
    plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    plt.tight_layout()
    save_fig(fig, output_path)
    plt.close(fig)


def get_cumulative_clinical(df, merge=False):
    """
    Compute cumulative unique lineages seen over time in clinical data.
    If merge=True, lineages are first aggregated to parent groups via the lineage map.
    Returns a DataFrame with columns: date, cumulative_lineages.
    """
    if merge:
        df = df.merge(
            lineage_map[["lineage", "callout_group"]],
            left_on="pangolin_lineage", right_on="lineage", how="left"
        ).rename(columns={
            "pangolin_lineage": "pangolin_lineage_original",
            "callout_group": "pangolin_lineage"
        })
    df = df[
        (df["date"] >= pd.Timestamp(start_date)) &
        (df["date"] <= pd.Timestamp(end_date))
    ].sort_values("date")
    seen, cumulative = set(), []
    for _, row in df.iterrows():
        lin = row["pangolin_lineage"]
        seen.add(lin)
        cumulative.append(len(seen))
    return (
        pd.DataFrame({"date": df["date"], "cumulative_lineages": cumulative})
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )


# ============================================================
# Load and preprocess shared data
# ============================================================

# --- NY State reported COVID-19 cases ---
ny_df = pd.read_csv(NY_CASES_FILE)
ny_df = ny_df[ny_df["Geography Level"] == "COUNTY"]
ny_df["Test Date"] = pd.to_datetime(ny_df["Test Date"], errors="coerce")
ny_df = ny_df.dropna(subset=["Test Date"])
ny_df = ny_df.rename(columns={"Test Date": "test_date", "Geography Description": "county"})
ny_df["Total New Positives"] = pd.to_numeric(ny_df["Total New Positives"], errors="coerce")
ny_df = ny_df[(ny_df["test_date"] >= start_date) & (ny_df["test_date"] <= end_date)]

ny_cases_daily = ny_df.groupby("test_date")["Total New Positives"].sum().sort_index()
ny_cases_daily = (
    pd.Series(0, index=pd.date_range(ny_cases_daily.index.min(),
                                     ny_cases_daily.index.max(), freq="D"))
    .add(ny_cases_daily, fill_value=0)
)
ny_cases_daily = ny_cases_daily / NY_POP * 100_000
ny_cases_7day  = ny_cases_daily.rolling(7).mean()

# --- Clinical sequencing (reconstructed from aggregate GISAID-derived data) ---
# clinical_lineage_daily_counts.csv.gz is a (date, pangolin_lineage, n) count
# table -- see prepare_clinical_aggregates.py. We expand it back into one row
# per specimen so every downstream groupby / value_counts / min / max
# computation below is identical to what it would be on a genuine
# per-specimen frame -- no per-specimen data ever leaves GISAID's platform,
# but the analysis code doesn't need to know that.
lineage_daily_counts = pd.read_csv(CLINICAL_LINEAGE_COUNTS_FILE, compression="gzip")
lineage_daily_counts["date"] = pd.to_datetime(lineage_daily_counts["date"], errors="coerce")
lineage_daily_counts = lineage_daily_counts.dropna(subset=["date"])
lineage_daily_counts = lineage_daily_counts[
    (lineage_daily_counts["date"] >= pd.Timestamp(start_date)) &
    (lineage_daily_counts["date"] <= pd.Timestamp(end_date))
]

clin_df = (
    lineage_daily_counts.loc[lineage_daily_counts.index.repeat(lineage_daily_counts["n"])]
    [["date", "pangolin_lineage"]]
    .reset_index(drop=True)
)

clinical_daily = clin_df["date"].value_counts().sort_index()
clinical_7day  = clinical_daily.rolling(7).mean()

# --- Wastewater sequencing ---
ww_seq_df  = pd.read_csv(WW_SEQ_FILE)
ww_meta_df = pd.read_csv(WW_META_FILE)

ww_meta_df["sample_collect_date"] = pd.to_datetime(
    ww_meta_df["sample_collect_date"], errors="coerce"
)
ww_meta_df = ww_meta_df[
    (ww_meta_df["sample_collect_date"] >= start_date) &
    (ww_meta_df["sample_collect_date"] <= end_date)
]

ww_merged = pd.merge(ww_seq_df, ww_meta_df, on="sample_id").drop_duplicates(subset="sample_id")
ww_merged = ww_merged[
    (ww_merged["sample_collect_date"] >= pd.Timestamp(start_date)) &
    (ww_merged["sample_collect_date"] <= pd.Timestamp(end_date))
]

ww_daily = ww_merged["sample_collect_date"].value_counts().sort_index()
ww_daily = (
    pd.Series(0, index=pd.date_range(ww_daily.index.min(), ww_daily.index.max()))
    .add(ww_daily, fill_value=0)
    .astype(int)
)
ww_7day = ww_daily.rolling(7).mean()

# --- Lineage aggregation key ---
lineage_map    = pd.read_csv(LINEAGE_MAP_FILE)
lineage_colors = (
    lineage_map.dropna(subset=["callout_group", "hex_code"])
    .set_index("callout_group")["hex_code"]
    .to_dict()
)

# --- Wastewater sequencing with lineage metadata (for lineage-level analyses) ---
ww_full = pd.merge(ww_seq_df, ww_meta_df, on="sample_id")
ww_full["sample_collect_date"] = pd.to_datetime(ww_full["sample_collect_date"], errors="coerce")
ww_full = ww_full[
    (ww_full["sample_collect_date"] >= pd.Timestamp(start_date)) &
    (ww_full["sample_collect_date"] <= pd.Timestamp(end_date))
]
ww_full = ww_full[ww_full["variant"] != "Unidentified"]

# Aggregate wastewater lineages to parent groups and sum variant_pct within groups
if MERGE_WW_GROUPS:
    ww_full = ww_full.merge(
        lineage_map[["lineage", "callout_group"]],
        left_on="variant", right_on="lineage", how="left"
    ).rename(columns={"variant": "variant_original", "callout_group": "variant"})
    ww_full["variant_pct_original"] = ww_full["variant_pct"]
    if MERGE_SUM:
        sums = ww_full.groupby(["sample_id", "variant"])["variant_pct"].transform("sum")
        mask = ww_full.duplicated(subset=["sample_id", "variant"], keep="first")
        ww_full["variant_pct"] = 0
        ww_full.loc[~mask, "variant_pct"] = sums[~mask]

# Aggregate clinical lineages to parent groups
clin_merged = clin_df.copy()
if MERGE_CLINICAL_GROUPS:
    clin_merged = clin_merged.merge(
        lineage_map[["lineage", "callout_group"]],
        left_on="pangolin_lineage", right_on="lineage", how="left"
    ).rename(columns={
        "pangolin_lineage": "pangolin_lineage_original",
        "callout_group": "pangolin_lineage"
    })

# Apply primary abundance threshold to wastewater data
ww_filtered = ww_full[ww_full["variant_pct"] >= WW_CUTOFF]


# ============================================================
# Figure: Three-panel sampling timeline
# NY cases per 100k / clinical sequences / wastewater sequences
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)

plot_timeline_panel(
    axes[0], ny_cases_daily.index, ny_cases_daily, ny_cases_7day,
    "NY statewide new cases\n(per 100k)", start_date, end_date,
    bar_color="black", line_color="black",
)
axes[0].set_title("A", loc="left", fontweight="bold")

plot_timeline_panel(
    axes[1], clinical_daily.index, clinical_daily, clinical_7day,
    "Clinical samples sequenced", start_date, end_date,
    bar_color="#1f4e79", line_color="#1f4e79",
)
axes[1].set_title("B", loc="left", fontweight="bold")

plot_timeline_panel(
    axes[2], ww_daily.index, ww_daily, ww_7day,
    "Wastewater samples sequenced", start_date, end_date,
    bar_color="#8b1a1a", line_color="#8b1a1a",
)
axes[2].set_title("C", loc="left", fontweight="bold")

date_ticks = pd.date_range(start=start_date, end=end_date, freq="4MS")
axes[2].set_xticks(date_ticks)
axes[2].set_xticklabels([d.strftime("%Y-%m") for d in date_ticks], rotation=45, ha="right")

plt.tight_layout(h_pad=1.2)
save_fig(fig, f"{FIGURES_DIR}/sampling_timeline.pdf")

# ============================================================
# Weekly aggregation (used for scatter plots, histograms, and correlations)
# ============================================================

weekly_df = pd.DataFrame({
    "cases_weekly":    ny_cases_daily.resample("W-MON").sum(),
    "clinical_weekly": clinical_daily.resample("W-MON").sum(),
    "ww_weekly":       ww_daily.resample("W-MON").sum(),
}).dropna()

# ============================================================
# Figure: Scatter — weekly sequences vs weekly cases
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(4.5, 7.0), sharex=True)

axes[0].scatter(
    weekly_df["cases_weekly"], weekly_df["clinical_weekly"],
    s=24, alpha=0.7, color="#1f4e79", edgecolor="none",
)
axes[0].set_ylabel("Clinical sequences\n(weekly total)")

axes[1].scatter(
    weekly_df["cases_weekly"], weekly_df["ww_weekly"],
    s=24, alpha=0.7, color="#8b1a1a", edgecolor="none",
)
axes[1].set_ylabel("Wastewater sequences\n(weekly total)")
axes[1].set_xlabel("NY statewide new cases per 100k (weekly total)")

for ax in axes:
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)

plt.tight_layout(h_pad=1.2)
save_fig(fig, f"{FIGURES_DIR}/sampling_vs_cases_scatter.pdf")

# ============================================================
# Pearson and Spearman correlations: weekly sequences vs cases
# ============================================================

pearson_clin,   pearson_clin_p  = pearsonr(weekly_df["cases_weekly"], weekly_df["clinical_weekly"])
spearman_clin,  spearman_clin_p = spearmanr(weekly_df["cases_weekly"], weekly_df["clinical_weekly"])
pearson_ww,     pearson_ww_p    = pearsonr(weekly_df["cases_weekly"], weekly_df["ww_weekly"])
spearman_ww,    spearman_ww_p   = spearmanr(weekly_df["cases_weekly"], weekly_df["ww_weekly"])

print("\nWeekly correlation statistics")
print("=" * 45)
print(f"Clinical vs cases:    Pearson r = {pearson_clin:.3f}  (p={pearson_clin_p:.3g}), "
      f"Spearman ρ = {spearman_clin:.3f}  (p={spearman_clin_p:.3g})")
print(f"Wastewater vs cases:  Pearson r = {pearson_ww:.3f}  (p={pearson_ww_p:.3g}), "
      f"Spearman ρ = {spearman_ww:.3f}  (p={spearman_ww_p:.3g})")

# ============================================================
# Figures: Histograms of sequences per week
# ============================================================

fig, ax = plt.subplots(figsize=(4.5, 3.5))
plot_fd_histogram(
    ax,
    weekly_df["clinical_weekly"],
    xlabel="Clinical sequences per week",
    bar_color="#1f4e79",
    ylabel="Number of weeks",
)
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/clinical_sequences_per_week_histogram.pdf")

fig, ax = plt.subplots(figsize=(4.5, 3.5))
plot_fd_histogram(
    ax,
    weekly_df["ww_weekly"],
    xlabel="Wastewater sequences per week",
    bar_color="#8b1a1a",
    ylabel="Number of weeks",
)
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/ww_sequences_per_week_histogram.pdf")

# ============================================================
# Figure: Choropleth — SARS-CoV-2 reported cases per county (per 100,000)
# ============================================================

counties_gdf = gpd.read_file(SHAPEFILE)
pop_df = pd.read_csv(COUNTY_POP_FILE)

cases_per_county = (
    ny_df.groupby("county")["Total New Positives"].sum()
    .rename("clinical_cases").reset_index()
)
cases_per_county["county"] = (
    cases_per_county["county"].str.replace("St. Lawrence", "St Lawrence")
)

cases_per_county = cases_per_county.merge(
    pop_df, left_on="county", right_on="County", how="left"
)
cases_per_county = cases_per_county.dropna(subset=["2020 Population Estimate"])
cases_per_county["2020 Population Estimate"] = (
    cases_per_county["2020 Population Estimate"]
    .astype(str).str.replace(",", "", regex=False).astype(float)
)
cases_per_county["clinical_cases"] = (
    cases_per_county["clinical_cases"] /
    cases_per_county["2020 Population Estimate"] * 100_000
)

counties_cases_merged = counties_gdf.merge(
    cases_per_county, left_on="NAME", right_on="county", how="left"
)
counties_cases_merged["clinical_cases"] = counties_cases_merged["clinical_cases"].fillna(0)

fig, ax = plt.subplots(figsize=(8, 8))
counties_cases_merged.plot(
    column="clinical_cases", ax=ax, cmap="Greys",
    edgecolor="black", linewidth=0.5, legend=True,
)
ax.set_title("SARS-CoV-2 Reported Cases per County (per 100,000)", fontweight="bold")
ax.axis("off")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/county_cases_map_per100k.pdf")

# ============================================================
# Figure: Choropleth — wastewater sequences per county
# ============================================================

sewersheds_df = pd.read_csv(SEWERSHEDS_FILE)

ww_county_df = ww_meta_df.merge(sewersheds_df, on="site_id", how="left").dropna(subset=["county"])
ww_per_county = ww_county_df.groupby("county").size().rename("SampleCount").reset_index()

counties_ww_merged = counties_gdf.merge(
    ww_per_county, left_on="NAME", right_on="county", how="left"
)
counties_ww_merged["SampleCount"] = counties_ww_merged["SampleCount"].fillna(0)

fig, ax = plt.subplots(figsize=(8, 8))
counties_ww_merged.plot(
    column="SampleCount", ax=ax, cmap="Reds",
    edgecolor="black", linewidth=0.5, legend=True,
)
ax.set_title("Number of Wastewater Samples per County", fontweight="bold")
ax.axis("off")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/county_map_ww.pdf")

# ============================================================
# Figure: Histogram — wastewater sequences per county
# ============================================================

fig, ax = plt.subplots(figsize=(4.5, 3.5))
plot_fd_histogram(ax, ww_per_county["SampleCount"],
                  xlabel="Wastewater sequences per county", bar_color="#8b1a1a")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/ww_sequences_per_county_histogram.pdf")

# ============================================================
# Figure: Choropleth — clinical sequences per county
# County harmonization (mapping GISAID locality strings to official
# county names) is done upstream in prepare_clinical_aggregates.py, so we
# only ever load already-harmonized totals here.
# ============================================================

official_counties = sorted(counties_gdf["NAME"].str.title())

clinical_df_merge = pd.read_csv(CLINICAL_COUNTY_COUNTS_FILE)
clinical_df_merge.columns = ["NAME", "SampleCount"]

clinical_series = (
    clinical_df_merge.set_index("NAME")["SampleCount"]
    .reindex(official_counties, fill_value=0)
)

counties_clin_merged = counties_gdf.merge(clinical_df_merge, on="NAME", how="left")
counties_clin_merged["SampleCount"] = counties_clin_merged["SampleCount"].fillna(0)

fig, ax = plt.subplots(figsize=(8, 8))
counties_clin_merged.plot(
    column="SampleCount", ax=ax, cmap="Blues",
    edgecolor="black", linewidth=0.5, legend=True,
)
ax.set_title("Number of Clinical Sequences per County", fontweight="bold")
ax.axis("off")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/county_clinical_map.pdf")

# ============================================================
# Figure: Histogram — clinical sequences per county
# ============================================================

fig, ax = plt.subplots(figsize=(4.5, 3.5))
plot_fd_histogram(ax, clinical_series,
                  xlabel="Clinical sequences per county", bar_color="#1f4e79")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/county_clinical_counts_hist.pdf")

# ============================================================
# Figure: Cumulative unique lineages detected over time
# Generated for both raw (unaggregated) and aggregated lineage labels.
# Wastewater lines are stratified by relative abundance cutoff.
# Clinical is shown as a single line.
# ============================================================

ww_seq_full = pd.merge(ww_seq_df, ww_meta_df, on="sample_id")
ww_seq_full["sample_collect_date"] = pd.to_datetime(
    ww_seq_full["sample_collect_date"], errors="coerce"
)

for mode in ["raw", "merged"]:
    cumulative_clinical = get_cumulative_clinical(clin_df, merge=(mode == "merged"))

    if mode == "raw":
        ww_plot_df = ww_seq_full.copy()
    else:
        ww_plot_df = ww_seq_full.merge(
            lineage_map[["lineage", "callout_group"]],
            left_on="variant", right_on="lineage", how="left"
        ).rename(columns={"variant": "variant_original", "callout_group": "variant"})
        sums = ww_plot_df.groupby(["sample_id", "variant"])["variant_pct"].transform("sum")
        mask = ww_plot_df.duplicated(subset=["sample_id", "variant"], keep="first")
        ww_plot_df["variant_pct"] = 0
        ww_plot_df.loc[~mask, "variant_pct"] = sums[~mask]

    ww_plot_df = ww_plot_df[
        (ww_plot_df["sample_collect_date"] >= pd.Timestamp(start_date)) &
        (ww_plot_df["sample_collect_date"] <= pd.Timestamp(end_date))
    ].sort_values("sample_collect_date")

    cutoffs = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
    reds = plt.get_cmap("Reds")
    colors = [reds(x) for x in [0.4, 0.55, 0.7, 0.8, 0.9, 1.0]]

    plt.figure(figsize=(14, 7))

    for cutoff, color in zip(cutoffs, colors):
        high = ww_plot_df[ww_plot_df["variant_pct"] > cutoff]
        seen, cumulative = set(), []
        for _, row in high.iterrows():
            seen.add(row["variant"])
            cumulative.append(len(seen))
        cum_df = (
            pd.DataFrame({"date": high["sample_collect_date"], "count": cumulative})
            .drop_duplicates(subset="date", keep="last")
            .reset_index(drop=True)
        )
        plt.plot(cum_df["date"], cum_df["count"],
                 marker="o", linestyle="-", markersize=5, color=color)

    plt.plot(
        cumulative_clinical["date"], cumulative_clinical["cumulative_lineages"],
        marker="o", linestyle="None", markersize=5, color="blue",
        label=f"Clinical ({'Aggregated' if mode == 'merged' else 'Unaggregated'})"
    )

    cutoff_patches = [
        mpatches.Patch(color=color, label=f"WW pct > {cutoff}")
        for cutoff, color in zip(cutoffs, colors)
    ]

    plt.ylim(0, 120 if mode == "merged" else 3500)
    plt.title(f"Cumulative Variants ({'Aggregated' if mode == 'merged' else 'Unaggregated'} lineages)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Total unique variants / lineages seen to date")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/cumulative_variants_{mode}.pdf", dpi=DPI)

# ============================================================
# Figure: Diverging bar — lineage overlap between clinical and wastewater
# Uses pre-aggregation (raw) lineage labels for both systems.
# Bars show counts of lineages unique to clinical, shared, or unique to wastewater,
# stratified by wastewater relative abundance cutoff.
# ============================================================

ww_overlap_df = pd.merge(ww_seq_df, ww_meta_df, on="sample_id")
ww_overlap_df["sample_collect_date"] = pd.to_datetime(
    ww_overlap_df["sample_collect_date"], errors="coerce"
)
ww_overlap_df = ww_overlap_df[
    (ww_overlap_df["sample_collect_date"] >= pd.Timestamp(start_date)) &
    (ww_overlap_df["sample_collect_date"] <= pd.Timestamp(end_date))
]
ww_overlap_df = ww_overlap_df[ww_overlap_df["variant"] != "Unidentified"]

clin_overlap_df = clin_df.copy()

clinical_lineages = set(clin_overlap_df["pangolin_lineage"].dropna().unique())
total_clinical    = len(clinical_lineages)

bar_rows = []
for cutoff in WW_CUTOFFS:
    ww_lineages = set(
        ww_overlap_df.loc[ww_overlap_df["variant_pct"] > cutoff, "variant"].unique()
    )
    bar_rows.append({
        "label": str(cutoff),
        "Clinical_only": -len(clinical_lineages - ww_lineages),
        "Both": len(clinical_lineages & ww_lineages),
        "WW_only": len(ww_lineages - clinical_lineages),
    })

bar_df = pd.DataFrame(bar_rows).iloc[::-1].reset_index(drop=True)
plt.style.use("classic")
fig, ax = plt.subplots(figsize=(7, 6))
y = range(len(bar_df))

ax.barh(y, bar_df["Clinical_only"], color="#1f4e79", label="Clinical only")
ax.barh(y, bar_df["Both"], left=0, color="gray", label="Both")
ax.barh(y, bar_df["WW_only"], left=bar_df["Both"], color="#8b1a1a", label="WW only")

for i in y:
    if bar_df.loc[i, "Clinical_only"] != 0:
        ax.text(
            bar_df.loc[i, "Clinical_only"] / 2, i,
            str(abs(bar_df.loc[i, "Clinical_only"])),
            ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            path_effects=[path_effects.Stroke(linewidth=1.5, foreground="black"),
                          path_effects.Normal()],
        )
    if bar_df.loc[i, "Both"] > 0:
        ax.text(
            bar_df.loc[i, "Both"] / 2, i, str(bar_df.loc[i, "Both"]),
            ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            path_effects=[path_effects.Stroke(linewidth=1.5, foreground="black"),
                          path_effects.Normal()],
        )
    if bar_df.loc[i, "WW_only"] > 0:
        ax.text(
            bar_df.loc[i, "Both"] + bar_df.loc[i, "WW_only"] / 2, i,
            str(bar_df.loc[i, "WW_only"]),
            ha="center", va="center", color="white", fontsize=10, fontweight="bold",
            path_effects=[path_effects.Stroke(linewidth=1.5, foreground="black"),
                          path_effects.Normal()],
        )

step = 1000
max_x = int(bar_df[["Clinical_only", "Both", "WW_only"]].apply(abs).sum(axis=1).max())
for xpos in range(0, max_x + step, step):
    ax.axvline(x=xpos,  color="black", linestyle="--", alpha=0.2)
    ax.axvline(x=-xpos, color="black", linestyle="--", alpha=0.2)
ax.axvline(0, color="black", linewidth=1)

ax.set_yticks(list(y))
ax.set_yticklabels(bar_df["label"])
ax.set_xlabel("Number of lineages")
ax.set_ylabel("Wastewater relative abundance cutoff")
ax.set_title("Overlap of Clinical and Wastewater Lineages by Variant Cutoff")
ax.set_xlim(bar_df["Clinical_only"].min() * 1.2,
            (bar_df["Both"] + bar_df["WW_only"]).max() * 1.05)

plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_overlap_diverging_barplot.pdf")

# ============================================================
# Figure: Proportion of detected lineages modified by taxonomic aggregation
# For clinical specimens and, per WW_CUTOFFS threshold, wastewater
# detections: what fraction of individually detected lineages above the
# cutoff have a raw label that differs from their parent lineage (per
# lineage_map) vs. already match their own parent (no reassignment
# needed). Computed at the detection/specimen level, not the
# unique-lineage level. No relative-abundance summation is applied here --
# for wastewater, the cutoff is applied to each row's raw variant_pct
# BEFORE merging in the parent lineage, and each surviving row is counted
# as one detection. (Summation across co-grouped lineages within a sample,
# per MERGE_SUM above, is a separate analysis -- recovering one combined
# abundance per parent lineage per sample -- and is not part of this
# proportion.)
# ============================================================

# --- Clinical: no cutoff applies. Reuse clin_merged, but fill any
# lineage with no entry in lineage_map back to its own name so it counts
# as "unaggregated" rather than silently becoming NaN. ---
clin_agg_check = clin_merged.copy()
clin_agg_check["pangolin_lineage"] = clin_agg_check["pangolin_lineage"].fillna(
    clin_agg_check["pangolin_lineage_original"]
)
clinical_total = len(clin_agg_check)
clinical_aggregated = (
    clin_agg_check["pangolin_lineage_original"] != clin_agg_check["pangolin_lineage"]
).sum()
clinical_unaggregated = clinical_total - clinical_aggregated
clinical_agg_pct = 100 * clinical_aggregated / clinical_total
clinical_unagg_pct = 100 * clinical_unaggregated / clinical_total

# --- Wastewater: for each cutoff, filter raw detections FIRST, then merge
# lineage_map onto only the surviving rows and compare labels.
# Cutoffs ordered descending (0.32 -> 0.01) so the plot reads high-to-low. ---
cutoffs_desc = sorted(WW_CUTOFFS, reverse=True)
ww_agg_pct_list, ww_unagg_pct_list, ww_agg_n_list = [], [], []
for cutoff in cutoffs_desc:
    ww_at_cutoff = ww_overlap_df[ww_overlap_df["variant_pct"] >= cutoff].copy()
    ww_at_cutoff = ww_at_cutoff.merge(
        lineage_map[["lineage", "callout_group"]],
        left_on="variant", right_on="lineage", how="left"
    ).rename(columns={"variant": "variant_original", "callout_group": "variant"})
    ww_at_cutoff["variant"] = ww_at_cutoff["variant"].fillna(ww_at_cutoff["variant_original"])

    n = len(ww_at_cutoff)
    aggregated = (ww_at_cutoff["variant_original"] != ww_at_cutoff["variant"]).sum()
    unaggregated = n - aggregated
    ww_agg_pct_list.append(100 * aggregated / n if n else 0)
    ww_unagg_pct_list.append(100 * unaggregated / n if n else 0)
    ww_agg_n_list.append(n)

# --- Plot: equidistant vertical stack, Clinical on top,
# aggregated on left / unaggregated on right ---
agg_labels = ["Clinical"] + [f"{c:g}" for c in cutoffs_desc]
aggregated_values = [clinical_agg_pct] + ww_agg_pct_list
unaggregated_values = [clinical_unagg_pct] + ww_unagg_pct_list

colors_agg = ["#99CCFF"] + ["#FF9999"] * len(WW_CUTOFFS)
colors_unagg = ["#CCCCFF"] + ["#FFCCCC"] * len(WW_CUTOFFS)

y_agg = np.arange(len(agg_labels))
height = 0.6

fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(y_agg, aggregated_values, height, color=colors_agg, edgecolor="black")
ax.barh(y_agg, unaggregated_values, height, left=aggregated_values, color=colors_unagg, edgecolor="black")

for i in range(len(agg_labels)):
    if aggregated_values[i] > 0:
        ax.text(aggregated_values[i] / 2, y_agg[i], f"{aggregated_values[i]:.1f}%",
                ha="center", va="center", fontsize=10)
    if unaggregated_values[i] > 0:
        ax.text(aggregated_values[i] + unaggregated_values[i] / 2, y_agg[i],
                f"{unaggregated_values[i]:.1f}%", ha="center", va="center", fontsize=10)

ax.set_yticks(y_agg)
ax.set_yticklabels(agg_labels)
ax.invert_yaxis()  # Clinical (index 0) at top
ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of Detected Lineages")
ax.set_title("Aggregated vs Unaggregated Lineages: Clinical & Wastewater", fontsize=14)
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_aggregation_proportion.pdf")
plt.close(fig)

legend_elements = [
    mpatches.Patch(facecolor="#99CCFF", edgecolor="black", label="Clinical Unaggregated Lineages"),
    mpatches.Patch(facecolor="#CCCCFF", edgecolor="black", label="Clinical Aggregated Lineages"),
    mpatches.Patch(facecolor="#FF9999", edgecolor="black", label="WW Unaggregated Lineages"),
    mpatches.Patch(facecolor="#FFCCCC", edgecolor="black", label="WW Aggregated Lineages"),
]
fig_legend = plt.figure(figsize=(4, 2))
ax_leg = fig_legend.add_subplot(111)
ax_leg.axis("off")
ax_leg.legend(handles=legend_elements, loc="center", frameon=False)
save_fig(fig_legend, f"{FIGURES_DIR}/lineage_aggregation_proportion_legend.pdf")
plt.close(fig_legend)

print("\nProportion of detected lineages modified by taxonomic aggregation:")
print(f"  Clinical: {clinical_agg_pct:.1f}% aggregated, {clinical_unagg_pct:.1f}% unaggregated "
      f"(n={clinical_total})")
for cutoff, agg, unagg, n in zip(cutoffs_desc, ww_agg_pct_list, ww_unagg_pct_list, ww_agg_n_list):
    print(f"  WW cutoff {cutoff}: {agg:.1f}% aggregated, {unagg:.1f}% unaggregated (n={n})")

# ============================================================
# Total lineage observations table (total_lineage_observations.csv):
# Total unique lineages detected -- clinical and
# per-WW-cutoff, pre- and post-taxonomic-aggregation ("unaggregated" =
# distinct raw lineage/variant names; "aggregated" = distinct parent
# lineage names after merging via lineage_map). "Other" (a residual
# catch-all category, not a real lineage) is excluded from the aggregated
# counts; no equivalent exclusion applies to unaggregated counts, since
# raw lineage calls are never literally named "Other".
# ============================================================

clin_unaggregated_n = clin_merged["pangolin_lineage_original"].nunique(dropna=True)
clin_aggregated_n = clin_merged.loc[
    clin_merged["pangolin_lineage"] != "Other", "pangolin_lineage"
].nunique(dropna=True)

richness_records = [{
    "system": "Clinical",
    "ww_cutoff": None,
    "n_unaggregated_lineages": clin_unaggregated_n,
    "n_aggregated_lineages": clin_aggregated_n,
}]

for cutoff in WW_CUTOFFS:
    ww_at_cutoff = ww_overlap_df[ww_overlap_df["variant_pct"] >= cutoff].copy()
    ww_unaggregated_n = ww_at_cutoff["variant"].nunique(dropna=True)

    ww_at_cutoff = ww_at_cutoff.merge(
        lineage_map[["lineage", "callout_group"]],
        left_on="variant", right_on="lineage", how="left"
    ).rename(columns={"variant": "variant_original", "callout_group": "variant"})
    ww_at_cutoff["variant"] = ww_at_cutoff["variant"].fillna(ww_at_cutoff["variant_original"])
    ww_aggregated_n = ww_at_cutoff.loc[
        ww_at_cutoff["variant"] != "Other", "variant"
    ].nunique(dropna=True)

    richness_records.append({
        "system": "Wastewater",
        "ww_cutoff": cutoff,
        "n_unaggregated_lineages": ww_unaggregated_n,
        "n_aggregated_lineages": ww_aggregated_n,
    })

richness_df = pd.DataFrame(richness_records)
richness_path = f"{RESULTS_DIR}/total_lineage_observations.csv"
richness_df.to_csv(richness_path, index=False)
print(f"\nTotal lineages detected (clinical + per-WW-cutoff, aggregated/unaggregated) "
      f"saved to {richness_path}")
print(richness_df.to_string(index=False))

# ============================================================
# Supplementary Table 1: Detection counts and first-detection date for
# every lineage (taxonomically aggregated) seen in clinical sequencing
# and/or wastewater at the most sensitive cutoff (0.01), reported for
# clinical and independently at each WW_CUTOFFS threshold. One row per
# lineage, wide format (one count + first-date + adjusted-% column set
# per system/cutoff). "Other" excluded, consistent with the aggregated
# counts above. Adjusted percentages use the same methodology as
# lineage_observation_table: clinical % = row count / total clinical
# specimens; WW % = summed variant_pct (abundance-weighted) / total
# summed variant_pct across all lineages passing that cutoff -- NOT a
# raw detection-count percentage.
# ============================================================

clin_lineages_all = set(
    clin_merged.loc[clin_merged["pangolin_lineage"] != "Other", "pangolin_lineage"].dropna().unique()
)
ww_lineages_at_0_01 = set(
    ww_full.loc[
        (ww_full["variant_pct"] >= 0.01) & (ww_full["variant"] != "Other"), "variant"
    ].dropna().unique()
)
supp2_lineages = sorted(clin_lineages_all | ww_lineages_at_0_01)

clin_total_count = len(clin_merged)
ww_total_abundance_by_cutoff = {
    cutoff: ww_full.loc[ww_full["variant_pct"] >= cutoff, "variant_pct"].sum()
    for cutoff in WW_CUTOFFS
}

supp2_records = []
for lin in supp2_lineages:
    clin_subset = clin_merged[clin_merged["pangolin_lineage"] == lin]
    clin_count = len(clin_subset)
    row = {
        "lineage": lin,
        "clinical_count": clin_count,
        "clinical_adjusted_pct": 100 * clin_count / clin_total_count if clin_total_count else 0,
        "clinical_first_detection_date": clin_subset["date"].min() if clin_count else pd.NaT,
    }
    for cutoff in WW_CUTOFFS:
        ww_subset = ww_full[(ww_full["variant"] == lin) & (ww_full["variant_pct"] >= cutoff)]
        ww_abundance_sum = ww_subset["variant_pct"].sum()
        total_at_cutoff = ww_total_abundance_by_cutoff[cutoff]
        row[f"ww_count_{cutoff:g}"] = len(ww_subset)
        row[f"ww_adjusted_pct_{cutoff:g}"] = (
            100 * ww_abundance_sum / total_at_cutoff if total_at_cutoff else 0
        )
        row[f"ww_first_detection_date_{cutoff:g}"] = (
            ww_subset["sample_collect_date"].min() if len(ww_subset) else pd.NaT
        )
    supp2_records.append(row)

supp2_df = pd.DataFrame(supp2_records)
supp1_path = f"{RESULTS_DIR}/Supp1_lineage_detection_counts_and_first_dates.csv"
supp2_df.to_csv(supp1_path, index=False)
print(f"\nSupplemental Table 1 ({len(supp2_df)} lineages: clinical union WW at 0.01) "
      f"saved to {supp1_path}")

# ============================================================
# Supplementary Table 2: Same as Supplemental Table 1, but UNAGGREGATED --
# raw (pre-taxonomic-aggregation) lineage/variant names, not the merged
# parent lineage names. Clinical uses pangolin_lineage_original. For
# wastewater, this filters on variant_pct_original (the true per-raw-
# sub-lineage abundance, preserved before MERGE_SUM redistributes/zeroes
# variant_pct across co-grouped rows) and groups by variant_original --
# using the post-aggregation variant_pct here would incorrectly zero out
# any raw sub-lineage that shares a parent with another sub-lineage
# detected in the same sample. No "Other" exclusion is needed since raw
# lineage/variant calls are never literally named "Other" (that label
# only exists as a post-aggregation callout_group value).
# ============================================================

clin_lineages_all_raw = set(clin_merged["pangolin_lineage_original"].dropna().unique())
ww_lineages_at_0_01_raw = set(
    ww_full.loc[ww_full["variant_pct_original"] >= 0.01, "variant_original"].dropna().unique()
)
supp3_lineages = sorted(clin_lineages_all_raw | ww_lineages_at_0_01_raw)

ww_total_abundance_by_cutoff_raw = {
    cutoff: ww_full.loc[ww_full["variant_pct_original"] >= cutoff, "variant_pct_original"].sum()
    for cutoff in WW_CUTOFFS
}

supp3_records = []
for lin in supp3_lineages:
    clin_subset = clin_merged[clin_merged["pangolin_lineage_original"] == lin]
    clin_count = len(clin_subset)
    row = {
        "lineage": lin,
        "clinical_count": clin_count,
        "clinical_adjusted_pct": 100 * clin_count / clin_total_count if clin_total_count else 0,
        "clinical_first_detection_date": clin_subset["date"].min() if clin_count else pd.NaT,
    }
    for cutoff in WW_CUTOFFS:
        ww_subset = ww_full[
            (ww_full["variant_original"] == lin) & (ww_full["variant_pct_original"] >= cutoff)
        ]
        ww_abundance_sum_raw = ww_subset["variant_pct_original"].sum()
        total_at_cutoff_raw = ww_total_abundance_by_cutoff_raw[cutoff]
        row[f"ww_count_{cutoff:g}"] = len(ww_subset)
        row[f"ww_adjusted_pct_{cutoff:g}"] = (
            100 * ww_abundance_sum_raw / total_at_cutoff_raw if total_at_cutoff_raw else 0
        )
        row[f"ww_first_detection_date_{cutoff:g}"] = (
            ww_subset["sample_collect_date"].min() if len(ww_subset) else pd.NaT
        )
    supp3_records.append(row)

supp3_df = pd.DataFrame(supp3_records)
supp2_path = f"{RESULTS_DIR}/Supp2_lineage_detection_counts_and_first_dates_unaggregated.csv"
supp3_df.to_csv(supp2_path, index=False)
print(f"\nSupplemental Table 2, unaggregated ({len(supp3_df)} lineages: clinical union WW at 0.01) "
      f"saved to {supp2_path}")

# ============================================================
# Figure: Violin — first detection lead time (Δt) at each WW abundance cutoff
# Δt = date of first clinical detection − date of first wastewater detection
# Positive values indicate wastewater detected the lineage first.
# ============================================================

clinical_first_dates = (
    clin_merged.dropna(subset=["pangolin_lineage"])
    .groupby("pangolin_lineage")["date"]
    .min()
    .to_dict()
)

ww_first_dates = {}
for cutoff in WW_CUTOFFS:
    ww_first_dates[cutoff] = (
        ww_full[ww_full["variant_pct"] >= cutoff]
        .groupby("variant")["sample_collect_date"]
        .min()
        .to_dict()
    )

records = []
for cutoff in WW_CUTOFFS:
    shared = (set(clinical_first_dates) & set(ww_first_dates[cutoff])) - {"Other"}
    for lin in shared:
        records.append({
            "lineage": lin,
            "ww_cutoff": cutoff,
            "delta_days": (clinical_first_dates[lin] - ww_first_dates[cutoff][lin]).days,
        })

timing_df = pd.DataFrame(records)
timing_df.to_csv(f"{RESULTS_DIR}/ww_clinical_timing_table.csv", index=False)

# --- Summary statistics per cutoff: median/IQR lead time, n, and a
# one-sample Wilcoxon signed-rank test of delta_days against 0 (i.e. is
# the median lead time significantly different from simultaneous
# detection?). Positive delta_days = wastewater detected first. ---
timing_summary_records = []
for cutoff in WW_CUTOFFS:
    deltas = timing_df.loc[timing_df["ww_cutoff"] == cutoff, "delta_days"]
    n = len(deltas)
    median = deltas.median()
    q1, q3 = deltas.quantile([0.25, 0.75])

    nonzero = deltas[deltas != 0]
    if len(nonzero) >= 1:
        stat, p_value = wilcoxon(nonzero)
    else:
        stat, p_value = float("nan"), float("nan")

    timing_summary_records.append({
        "ww_cutoff": cutoff,
        "n": n,
        "median_delta_days": median,
        "q1_delta_days": q1,
        "q3_delta_days": q3,
        "wilcoxon_stat": stat,
        "wilcoxon_p": p_value,
    })

timing_summary_df = pd.DataFrame(timing_summary_records)
timing_summary_df.to_csv(f"{RESULTS_DIR}/ww_clinical_timing_summary.csv", index=False)

print("\nLead time (Δt = clinical − wastewater detection) summary by WW cutoff:")
for row in timing_summary_df.itertuples():
    direction = "WW earlier" if row.median_delta_days > 0 else (
        "clinical earlier" if row.median_delta_days < 0 else "simultaneous"
    )
    print(f"  cutoff {row.ww_cutoff}: n={row.n}, median={row.median_delta_days:.0f}d "
          f"(IQR {row.q1_delta_days:.0f} to {row.q3_delta_days:.0f}), "
          f"{direction}, Wilcoxon p={row.wilcoxon_p:.4g}")

fig, ax = plt.subplots(figsize=(5, 4))

violin_data = [
    timing_df.loc[timing_df["ww_cutoff"] == c, "delta_days"] for c in WW_CUTOFFS
]
violin = ax.violinplot(
    violin_data, positions=range(len(WW_CUTOFFS)),
    showmeans=True, showmedians=True, widths=0.8,
)
for body in violin["bodies"]:
    body.set_facecolor("#7a1f1f")
    body.set_alpha(0.6)
    body.set_edgecolor("black")

violin["cmedians"].set_color("black")
violin["cmedians"].set_linewidth(2)
violin["cmeans"].set_color("#8B4513")
violin["cmeans"].set_linewidth(2)

for part in ["cbars", "cmins", "cmaxes"]:
    violin[part].set_color("steelblue")
    violin[part].set_linewidth(1.5)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks(range(len(WW_CUTOFFS)))
ax.set_xticklabels([str(c) for c in WW_CUTOFFS])
ax.set_xlabel("Wastewater detection threshold (variant fraction)")
ax.set_ylabel("Δt = Clinical − Wastewater detection (days)")

plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/ww_clinical_timing_violin.pdf")

# ============================================================
# Figure: Weighted lineage timelines (top N lineages)
# ============================================================

lineages_timeline = select_top_lineages_by_peak(clin_merged, NUM_LINEAGES, ROLLING_WINDOW)

plot_combined_lineage_timelines_weighted(
    clinical_df=clin_merged,
    ww_df=ww_filtered,
    lineages=lineages_timeline,
    lineage_colors=lineage_colors,
    start_date=start_date,
    end_date=end_date,
    output_path=f"{FIGURES_DIR}/combined_lineage_timelines_weighted.pdf",
    ww_cutoff=WW_CUTOFF,
)

# ============================================================
# Supplementary Figure: Weighted lineage timelines across all WW cutoffs
# Clinical panel on top, one wastewater panel per WW_CUTOFFS threshold
# below it -- same row-stacking layout as lineage_emergence_multi_cutoff.
# ============================================================

plot_combined_lineage_timelines_weighted_multi_cutoff(
    clinical_df=clin_merged,
    ww_df=ww_full,
    lineages=lineages_timeline,
    lineage_colors=lineage_colors,
    start_date=start_date,
    end_date=end_date,
    output_path=f"{FIGURES_DIR}/combined_lineage_timelines_weighted_multi_cutoff.pdf",
    ww_cutoffs=WW_CUTOFFS,
)

# ============================================================
# Figure: Lineage emergence timeline
# One row per surveillance system/cutoff; each dot marks the date of
# first detection of a lineage. Clinical row on top, followed by
# wastewater rows at decreasing abundance thresholds.
# ============================================================

ww_cutoffs_emergence = [0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
n_rows = 1 + len(ww_cutoffs_emergence)

fig, axes = plt.subplots(
    n_rows, 1,
    figsize=(8, 0.7 * n_rows + 1),
    sharex=True,
)

ax = axes[0]
ax.annotate("", xy=(pd.Timestamp(end_date), 0), xytext=(pd.Timestamp(start_date), 0),
            arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black"))
for lin in lineages_timeline:
    color = lineage_colors.get(lin, "#999999")
    clin_dates = clin_merged.loc[clin_merged["pangolin_lineage"] == lin, "date"]
    first_clin = clin_dates.min() if not clin_dates.empty else pd.NaT
    if pd.notna(first_clin):
        ax.scatter(first_clin, 0, color=color, s=40, zorder=3)
        ax.text(first_clin, 0.05, lin, color=color, ha="center",
                va="bottom", rotation=45, fontsize=8)

ax.set_ylim(-0.3, 0.3)
ax.set_yticks([])
ax.set_title("Lineage Emergence Timeline: Clinical + Wastewater Cutoffs")
ax.grid(False)
for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

for i, cutoff in enumerate(ww_cutoffs_emergence, start=1):
    ax = axes[i]
    ax.annotate("", xy=(pd.Timestamp(end_date), 0), xytext=(pd.Timestamp(start_date), 0),
                arrowprops=dict(arrowstyle="->", linewidth=1.5, color="black"))
    for lin in lineages_timeline:
        color = lineage_colors.get(lin, "#999999")
        ww_dates = ww_full.loc[
            (ww_full["variant"] == lin) & (ww_full["variant_pct"] >= cutoff),
            "sample_collect_date",
        ]
        first_ww = ww_dates.min() if not ww_dates.empty else pd.NaT
        if pd.notna(first_ww):
            ax.scatter(first_ww, 0, color=color, s=40, zorder=3)
            ax.text(first_ww, 0.05, lin, color=color, ha="center",
                    va="bottom", rotation=45, fontsize=8)
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_ylabel(f"WW ≥ {cutoff:.2f}", rotation=0, labelpad=35, fontsize=9, va="center")
    ax.grid(False)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

axes[-1].tick_params(axis="x", rotation=45)
plt.xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_emergence_multi_cutoff.pdf")

# ============================================================
# Table: First-detection dates for each lineage shown on the emergence plot
# One row per (lineage, WW cutoff), matching the plot's granularity:
#   - first_ww_sample_collect_date: earliest sample_collect_date among WW
#     detections of this lineage at/above this cutoff
#   - first_clinical_sample_collect_date: earliest clinical specimen
#     collection date for this lineage (cutoff-independent, repeated
#     across rows for convenience)
# ============================================================

emergence_records = []
for lin in lineages_timeline:
    clin_mask = clin_merged["pangolin_lineage"] == lin
    clin_dates = clin_merged.loc[clin_mask, "date"]
    first_clin_collect = clin_dates.min() if not clin_dates.empty else pd.NaT

    for cutoff in ww_cutoffs_emergence:
        ww_dates = ww_full.loc[
            (ww_full["variant"] == lin) & (ww_full["variant_pct"] >= cutoff),
            "sample_collect_date",
        ]
        first_ww = ww_dates.min() if not ww_dates.empty else pd.NaT

        emergence_records.append({
            "lineage": lin,
            "ww_cutoff": cutoff,
            "first_ww_sample_collect_date": first_ww,
            "first_clinical_sample_collect_date": first_clin_collect,
        })

emergence_dates_df = pd.DataFrame(emergence_records)
emergence_dates_df.to_csv(f"{RESULTS_DIR}/lineage_emergence_dates.csv", index=False)
print(f"\nLineage emergence dates saved to {RESULTS_DIR}/lineage_emergence_dates.csv")

# ============================================================
# Figure: Lineage relative abundance scatter (clinical vs wastewater)
# Displays lineages comprising ≥1% of total observations in both systems.
# Pearson correlation computed on the union of the top 20 lineages from
# each system, excluding lineages with zero abundance in either.
# Regression line fit in log10 space.
# ============================================================

clin_counts  = clin_merged["pangolin_lineage"].value_counts()
clin_percent = 100 * clin_counts / clin_counts.sum()

ww_weighted = (
    ww_filtered[ww_filtered["variant_pct"] > WW_CUTOFF]
    .groupby("variant")["variant_pct"]
    .sum()
)
ww_percent = 100 * ww_weighted / ww_weighted.sum()

top_clin     = clin_percent.head(20).index
top_ww       = ww_percent.head(20).index
top_lineages = top_clin.union(top_ww)

clin_counts_top  = clin_counts.reindex(top_lineages, fill_value=0)
clin_percent_top = clin_percent.reindex(top_lineages, fill_value=0)
ww_weighted_top  = ww_weighted.reindex(top_lineages, fill_value=0)
ww_percent_top   = ww_percent.reindex(top_lineages, fill_value=0)

summary_df = pd.DataFrame({
    "Lineage":            top_lineages,
    "Clinical counts":    clin_counts_top.values,
    "Clinical %":         clin_percent_top.values,
    "Adjusted WW counts": ww_weighted_top.values,
    "Adjusted WW %":      ww_percent_top.values,
})
summary_df.to_csv(f"{RESULTS_DIR}/lineage_observation_table.csv", index=False)

corr_mask = (clin_percent_top > 0) & (ww_percent_top > 0)
x_corr = clin_percent_top[corr_mask]
y_corr = ww_percent_top[corr_mask]

r, p = pearsonr(x_corr.values, y_corr.values)
logx = np.log10(x_corr.values)
logy = np.log10(y_corr.values)
slope, intercept = np.polyfit(logx, logy, 1)
x_line = np.logspace(np.log10(x_corr.min()), np.log10(x_corr.max()), 100)
y_line = 10 ** (intercept + slope * np.log10(x_line))

plot_mask = (clin_percent_top >= 1) & (ww_percent_top >= 1)
x = clin_percent_top[plot_mask]
y = ww_percent_top[plot_mask]

point_colors = [lineage_colors.get(lin, "#999999") for lin in x.index]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x.values, y.values, c=point_colors,
           edgecolors="black", linewidths=0.4, alpha=0.85)
ax.plot([1e-3, 1e2], [1e-3, 1e2], linestyle="-", color="grey",
        linewidth=2, alpha=0.6, label="x = y", zorder=0)
ax.plot(x_line, y_line, linestyle="--", color="grey", linewidth=2,
        label=f"Pearson fit (r = {r:.2f}, p = {p:.2e})")

for lin, xi, yi in zip(x.index, x.values, y.values):
    if xi > 1 and yi > 1:
        ax.text(xi, yi, lin, fontsize=10, ha="center", va="bottom", rotation=30,
                color=lineage_colors.get(lin, "#333333"))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Clinical % of total")
ax.set_ylabel("Wastewater weighted % of total")
ax.set_title("Lineage Relative Abundance: Clinical vs Wastewater (≥1%)")
ax.set_xlim(1, 20)
ax.set_ylim(1, 20)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(loc="upper left")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_scatter_percent_log.pdf")
plt.close(fig)

# ============================================================
# Supplementary Figure: Lineage relative abundance scatter across all WW
# cutoffs. Same logic as the single-cutoff version above (top-20-per-
# system candidate pool, Pearson correlation on nonzero-both-systems
# lineages, ≥1%-both-systems lineages displayed), recomputed independently
# at each WW_CUTOFFS threshold and stacked into one panel per cutoff.
# ============================================================

n_cutoffs = len(WW_CUTOFFS)
fig, axes = plt.subplots(n_cutoffs, 1, figsize=(6, 6 * n_cutoffs))

for ax, cutoff in zip(axes, WW_CUTOFFS):
    ww_weighted_c = (
        ww_full[ww_full["variant_pct"] > cutoff]
        .groupby("variant")["variant_pct"]
        .sum()
    )
    ww_percent_c = 100 * ww_weighted_c / ww_weighted_c.sum()

    top_clin_c     = clin_percent.head(20).index
    top_ww_c       = ww_percent_c.head(20).index
    top_lineages_c = top_clin_c.union(top_ww_c)

    clin_percent_top_c = clin_percent.reindex(top_lineages_c, fill_value=0)
    ww_percent_top_c   = ww_percent_c.reindex(top_lineages_c, fill_value=0)

    corr_mask_c = (clin_percent_top_c > 0) & (ww_percent_top_c > 0)
    x_corr_c = clin_percent_top_c[corr_mask_c]
    y_corr_c = ww_percent_top_c[corr_mask_c]

    r_c, p_c = pearsonr(x_corr_c.values, y_corr_c.values)
    logx_c = np.log10(x_corr_c.values)
    logy_c = np.log10(y_corr_c.values)
    slope_c, intercept_c = np.polyfit(logx_c, logy_c, 1)
    x_line_c = np.logspace(np.log10(x_corr_c.min()), np.log10(x_corr_c.max()), 100)
    y_line_c = 10 ** (intercept_c + slope_c * np.log10(x_line_c))

    plot_mask_c = (clin_percent_top_c >= 1) & (ww_percent_top_c >= 1)
    x_c = clin_percent_top_c[plot_mask_c]
    y_c = ww_percent_top_c[plot_mask_c]

    point_colors_c = [lineage_colors.get(lin, "#999999") for lin in x_c.index]

    ax.scatter(x_c.values, y_c.values, c=point_colors_c,
               edgecolors="black", linewidths=0.4, alpha=0.85)
    ax.plot([1e-3, 1e2], [1e-3, 1e2], linestyle="-", color="grey",
            linewidth=2, alpha=0.6, label="x = y", zorder=0)
    ax.plot(x_line_c, y_line_c, linestyle="--", color="grey", linewidth=2,
            label=f"Pearson fit (r = {r_c:.2f}, p = {p_c:.2e})")

    for lin, xi, yi in zip(x_c.index, x_c.values, y_c.values):
        if xi > 1 and yi > 1:
            ax.text(xi, yi, lin, fontsize=9, ha="center", va="bottom", rotation=30,
                    color=lineage_colors.get(lin, "#333333"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clinical % of total")
    ax.set_ylabel("Wastewater weighted % of total")
    ax.set_title(f"WW cutoff {cutoff:g} (≥1% both systems, N={len(x_c)})")
    ax.set_xlim(1, 20)
    ax.set_ylim(1, 20)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_scatter_percent_log_multi_cutoff.pdf")
plt.close(fig)

# ============================================================
# Figure: Violin — lead time (Δt) at WW_CUTOFF for lineages with nonzero
# (>0%) relative abundance in both clinical and wastewater surveillance
# -- the full lineage universe from both systems, not restricted to the
# top-20-per-system pool used for the scatter plot. Mean, median, IQR,
# and a one-sample Wilcoxon signed-rank test (against 0) are annotated
# directly on the figure.
# ============================================================

all_lineages_idx = clin_percent.index.union(ww_percent.index)
clin_percent_full = clin_percent.reindex(all_lineages_idx, fill_value=0)
ww_percent_full   = ww_percent.reindex(all_lineages_idx, fill_value=0)

nonzero_mask = (clin_percent_full > 0) & (ww_percent_full > 0)
lineages_nonzero = set(clin_percent_full[nonzero_mask].index) - {"Other"}

fig3_timing_lineages = [
    lin for lin in lineages_nonzero
    if lin in clinical_first_dates and lin in ww_first_dates[WW_CUTOFF]
]
fig3_deltas = pd.Series([
    (clinical_first_dates[lin] - ww_first_dates[WW_CUTOFF][lin]).days
    for lin in fig3_timing_lineages
])

n_fig3 = len(fig3_deltas)
median_fig3 = fig3_deltas.median()
mean_fig3 = fig3_deltas.mean()
q1_fig3, q3_fig3 = fig3_deltas.quantile([0.25, 0.75])

nonzero_deltas_fig3 = fig3_deltas[fig3_deltas != 0]
if len(nonzero_deltas_fig3) >= 1:
    wstat_fig3, wp_fig3 = wilcoxon(nonzero_deltas_fig3)
else:
    wstat_fig3, wp_fig3 = float("nan"), float("nan")

fig, ax = plt.subplots(figsize=(4, 5))

violin = ax.violinplot(
    [fig3_deltas], positions=[0],
    showmeans=True, showmedians=True, widths=0.8,
)
for body in violin["bodies"]:
    body.set_facecolor("#7a1f1f")
    body.set_alpha(0.6)
    body.set_edgecolor("black")

violin["cmedians"].set_color("black")
violin["cmedians"].set_linewidth(2)
violin["cmeans"].set_color("#8B4513")
violin["cmeans"].set_linewidth(2)

for part in ["cbars", "cmins", "cmaxes"]:
    violin[part].set_color("steelblue")
    violin[part].set_linewidth(1.5)

ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_xticks([0])
ax.set_xticklabels([f"Nonzero (>0%)\nboth systems\n(N={n_fig3})"])
ax.set_xlim(-1, 1)
ax.set_ylabel("Δt = Clinical − Wastewater detection (days)")
ax.set_title(f"Lead Time, Shared Lineages (WW cutoff {WW_CUTOFF:g})")

stats_text = (
    f"Median = {median_fig3:.1f}d\n"
    f"Mean = {mean_fig3:.1f}d\n"
    f"IQR = {q1_fig3:.1f} to {q3_fig3:.1f}d\n"
    f"Wilcoxon p = {wp_fig3:.3g}"
)
ax.text(
    0.98, 0.02, stats_text, transform=ax.transAxes,
    ha="right", va="bottom", fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="black"),
)

plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/fig3_timing_violin.pdf")
plt.close(fig)

print(f"\nFigure 3 timing violin: N={n_fig3}, median Δt={median_fig3:.1f}d, "
      f"mean Δt={mean_fig3:.1f}d (IQR {q1_fig3:.1f} to {q3_fig3:.1f}), "
      f"Wilcoxon stat={wstat_fig3:.1f}, p={wp_fig3:.4g}")

# ============================================================
# Cohen's kappa: weekly co-detection agreement, computed for three
# lineage groups: top 10 clinical, top 50 clinical, and all lineages
# seen in both systems (no cap). Weekly presence/absence binarized for
# each system independently, then kappa computed per lineage.
# "Other" excluded from all three groups.
# ============================================================

weekly_range = pd.date_range(start=start_date, end=end_date, freq="W-SUN")


def compute_weekly_kappa(lineages, print_each=False):
    """Returns a DataFrame of per-lineage Cohen's kappa (weekly
    presence/absence, WW_CUTOFF) for the given list of lineages."""
    results = []
    for lin in lineages:
        clin_series = (
            clin_merged[clin_merged["pangolin_lineage"] == lin]
            .groupby(pd.Grouper(key="date", freq="W-SUN"))
            .size()
            .reindex(weekly_range, fill_value=0)
        )
        ww_series = (
            ww_full[(ww_full["variant"] == lin) & (ww_full["variant_pct"] >= WW_CUTOFF)]
            .groupby(pd.Grouper(key="sample_collect_date", freq="W-SUN"))["variant_pct"]
            .sum()
            .reindex(weekly_range, fill_value=0)
        )
        clin_bin = (clin_series > 0).astype(int).values
        ww_bin   = (ww_series  > 0).astype(int).values
        kappa = cohen_kappa_score(clin_bin, ww_bin)
        results.append({"lineage": lin, "cohen_kappa": kappa})
        if print_each:
            print(f"{lin}: κ = {kappa:.3f}")
    return pd.DataFrame(results)


def summarize_kappa(kappa_df, label, out_path):
    kappa_df.to_csv(out_path, index=False)
    values = kappa_df["cohen_kappa"].dropna()
    print(f"\n{label} (WW cutoff {WW_CUTOFF:g}): N={len(values)}, "
          f"mean={values.mean():.3f}, median={values.median():.3f} "
          f"(IQR {values.quantile(0.25):.3f}-{values.quantile(0.75):.3f})")
    print(f"  Saved to {out_path}")
    return values


clin_lineages_ranked = (
    clin_merged.loc[clin_merged["pangolin_lineage"] != "Other", "pangolin_lineage"]
    .value_counts()
)
top10_lineages = clin_lineages_ranked.head(10).index.tolist()
top50_lineages = clin_lineages_ranked.head(50).index.tolist()

ww_lineages_seen = set(
    ww_full.loc[
        (ww_full["variant_pct"] >= WW_CUTOFF) & (ww_full["variant"] != "Other"), "variant"
    ].dropna().unique()
)
all_shared_lineages = sorted(set(clin_lineages_ranked.index) & ww_lineages_seen)

kappa_df_top10 = compute_weekly_kappa(top10_lineages, print_each=True)
top10_kappa_values = summarize_kappa(
    kappa_df_top10, "1) Cohen's kappa, top 10 clinical lineages",
    f"{RESULTS_DIR}/cohen_kappa_top10.csv"
)

kappa_df_top50 = compute_weekly_kappa(top50_lineages, print_each=True)
top50_kappa_values = summarize_kappa(
    kappa_df_top50, "2) Cohen's kappa, top 50 clinical lineages",
    f"{RESULTS_DIR}/cohen_kappa_top50.csv"
)
kappa_df = kappa_df_top50  # preserved name for downstream sections that reference kappa_df

kappa_df_all = compute_weekly_kappa(all_shared_lineages, print_each=False)
all_kappa_values = summarize_kappa(
    kappa_df_all, "3) Cohen's kappa, ALL lineages seen in both clinical and wastewater",
    f"{RESULTS_DIR}/cohen_kappa_all_shared.csv"
)



# ============================================================
# Figure: Cohen's kappa distribution histogram
# Bars subdivided by horizontal lines; agreement regions annotated.
# ============================================================

kappa_df = pd.read_csv(f"{RESULTS_DIR}/cohen_kappa_top50.csv")
kappa_df = kappa_df[kappa_df["lineage"] != "Other"]
kappa_values = kappa_df["cohen_kappa"].dropna()

mean_kappa   = kappa_values.mean()
median_kappa = kappa_values.median()

fig, ax = plt.subplots(figsize=(12, 7))

bins = np.linspace(-1, 1, 41)
counts, bin_edges, patches = ax.hist(
    kappa_values, bins=bins, edgecolor="black", alpha=0.75, zorder=2,
)

ax.set_xlim(-1, 1)
ax.set_xlabel("Cohen's Kappa", fontsize=12)
ax.set_ylabel("Number of lineages", fontsize=12)
ax.set_title("Distribution of Cohen's Kappa Scores", fontsize=14)

for count, left_edge, right_edge in zip(counts, bin_edges[:-1], bin_edges[1:]):
    for y_val in range(1, int(count) + 1):
        ax.hlines(y_val, xmin=left_edge, xmax=right_edge,
                  colors="black", linewidth=0.6, alpha=0.6, zorder=3)

ax.axvline(mean_kappa,   linestyle="-",  linewidth=2, label=f"Mean = {mean_kappa:.2f}",   zorder=4)
ax.axvline(median_kappa, linestyle="--", linewidth=2, label=f"Median = {median_kappa:.2f}", zorder=4)

agreement_regions = [
    (-1.00, 0.00, "Disagreement"),
    (0.00,  0.20, "Slight"),
    (0.20,  0.40, "Fair"),
    (0.40,  0.60, "Moderate"),
    (0.60,  0.80, "Substantial"),
    (0.80,  1.00, "Almost Perfect"),
]

y_max = max(counts)
for start, end, label in agreement_regions:
    ax.axvspan(start, end, alpha=0.05, zorder=1)
    ax.text((start + end) / 2, y_max * 0.95, label,
            ha="center", va="top", rotation=90, fontsize=9, zorder=5)

for boundary in [-1, 0, 0.2, 0.4, 0.6, 0.8, 1]:
    ax.axvline(boundary, linestyle=":", linewidth=0.8, zorder=3)

ax.legend(loc="upper left")
plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/kappa_histogram.pdf")

# ============================================================
# Figure: Clinical and wastewater lineage heatmaps (top 50 lineages, weekly)
# Rows sorted by date of smoothed peak clinical abundance.
# Clinical panel: weekly counts (Blues colormap).
# Wastewater panel: weekly sum of variant_pct (Reds colormap).
# Colorbar scaled to 0.5th–99th percentile to reduce outlier influence.
# ============================================================

full_range_heatmap = pd.date_range(start=start_date, end=end_date, freq="D")

peak_dates_heatmap = {}
for lin in top50_lineages:
    series = (
        clin_merged[clin_merged["pangolin_lineage"] == lin]
        .groupby("date").size()
        .reindex(full_range_heatmap, fill_value=0)
    )
    peak_dates_heatmap[lin] = rolling_mean(series, ROLLING_WINDOW).idxmax()

top50_lineages_sorted = sorted(top50_lineages, key=lambda x: peak_dates_heatmap[x])

clin_df_heat = clin_merged[
    (clin_merged["date"] >= pd.Timestamp(start_date)) &
    (clin_merged["date"] <= pd.Timestamp(end_date))
]
ww_df_heat = ww_filtered[
    (ww_filtered["sample_collect_date"] >= pd.Timestamp(start_date)) &
    (ww_filtered["sample_collect_date"] <= pd.Timestamp(end_date))
]

ww_heatmap   = []
clin_heatmap = []
lin_labels   = []

for lin in top50_lineages_sorted:
    clin_series = (
        clin_df_heat[clin_df_heat["pangolin_lineage"] == lin]
        .groupby("date").size()
        .reindex(full_range_heatmap, fill_value=0)
    )
    ww_series = (
        ww_df_heat[(ww_df_heat["variant"] == lin) & (ww_df_heat["variant_pct"] > WW_CUTOFF)]
        .groupby("sample_collect_date")["variant_pct"]
        .sum()
        .reindex(full_range_heatmap, fill_value=0)
    )
    clin_heatmap.append(clin_series.resample("W").sum())
    ww_heatmap.append(ww_series.resample("W").sum())
    lin_labels.append(lin)

n_weeks = len(clin_heatmap[0])
dates = pd.date_range(start=start_date, periods=n_weeks, freq="W")
desired_months = [1, 5, 9]
tick_idx    = [i for i, d in enumerate(dates) if d.month in desired_months and d.day <= 7]
tick_labels = [dates[i].strftime("%Y-%m") for i in tick_idx]

clin_vmin, clin_vmax = np.percentile(clin_heatmap, [0.5, 99])
ww_vmin,   ww_vmax   = np.percentile(ww_heatmap,   [0.5, 99])

n_rows = len(lin_labels)
y_ticks = np.arange(n_rows)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

im1 = axes[0].imshow(
    clin_heatmap, cmap="Blues", interpolation="nearest",
    vmin=clin_vmin, vmax=clin_vmax, aspect=5,
)
axes[0].set_title("Clinical")
axes[0].set_yticks(y_ticks)
axes[0].set_yticklabels(lin_labels, fontsize=6)
axes[0].set_xticks(tick_idx)
axes[0].set_xticklabels(tick_labels, rotation=45, ha="right")

im2 = axes[1].imshow(
    ww_heatmap, cmap="Reds", interpolation="nearest",
    vmin=ww_vmin, vmax=ww_vmax, aspect=5,
)
axes[1].set_title("Wastewater")
axes[1].set_yticks(y_ticks)
axes[1].set_yticklabels(lin_labels, fontsize=6)
axes[1].set_xticks(tick_idx)
axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")

plt.tight_layout()
save_fig(fig, f"{FIGURES_DIR}/lineage_heatmap_clinical_ww.pdf")
plt.close(fig)

print("\nAll manuscript figures generated.")
