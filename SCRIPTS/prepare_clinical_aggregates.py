#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_clinical_aggregates.py

Collapses the raw GISAID clinical sequencing export down to small
AGGREGATE tables -- the minimum needed to reproduce every clinical-
sequencing figure and table in wastewater_clinical_comparison.py, with no
per-specimen row surviving into the public output at all.

Outputs (all safe to publish)
------------------------------
1. clinical_lineage_daily_counts.csv.gz
       columns: date, pangolin_lineage, n

2. clinical_county_counts.csv
       columns: county, n_sequences

Study window
------------
Rows are restricted to the study period (collection date between
START_DATE and END_DATE, currently 2023-01-01 to 2025-10-01) before any
aggregation. These constants must be kept in sync with start_date/end_date
in wastewater_clinical_comparison.py -- if you change one, change the other.

Deduplication
-------------
276 rows (within the study period) were found to be exact duplicate
records sharing the same accession_id (see check_duplicate_accessions.py)
-- an artifact of overlapping GISAID search queries at export time, not
genuine differing metadata. Dropped via
drop_duplicates(subset="accession_id") before aggregating.

Exclusion criteria
------------------
Rows are excluded from all aggregate tables if `division` != "New York".
An audit of this dataset found 9 such rows -- specimens from Tennessee,
New Jersey (x3), Connecticut (x3), California, and Belgium -- each
confirmed non-NY by cross-checking strain name, submitting lab, and
division_exposure. Their raw `location` field happened to contain a
"New York"-ish string by coincidence, which is why a location-only filter
would have missed them.

The raw GISAID export (GISAID_Clinical.txt.gz) is NOT included in the
public repository. Researchers with an approved GISAID account may
request it directly from the corresponding author, per GISAID's Database
Access Agreement. The manuscript cites a GISAID EPI_SET identifying the
exact clinical record set used in the analysis.

Usage
-----
    python prepare_clinical_aggregates.py

Inputs:
    ../DATA/GISAID_Clinical.txt.gz     (raw GISAID export; NOT redistributed)
    ../DATA/New_York_State_Locality_Hierarchy_with_Websites_20240926.csv
    ../DATA/shapefile/Counties_Shoreline.shp
"""

import pandas as pd
import geopandas as gpd

# ============================================================
# Configuration
# ============================================================

RAW_CLINICAL_FILE = "../DATA/GISAID_Clinical.txt.gz"
LOCALITY_FILE      = "../DATA/New_York_State_Locality_Hierarchy_with_Websites_20240926.csv"
SHAPEFILE           = "../DATA/shapefile/Counties_Shoreline.shp"

LINEAGE_DAILY_COUNTS_FILE = "../DATA/clinical_lineage_daily_counts.csv.gz"
COUNTY_COUNTS_FILE        = "../DATA/clinical_county_counts.csv"

# Must match start_date/end_date in wastewater_clinical_comparison.py --
# restricting here means the aggregate files only ever reflect specimens
# actually used in the analysis, rather than the full history of the raw
# export.
START_DATE = pd.Timestamp("2023-01-01")
END_DATE   = pd.Timestamp("2025-10-01")

# Raw columns actually read from the GISAID export. `accession_id` is used
# only for deduplication and `division` only for the out-of-state
# exclusion -- both are dropped before any aggregate is written.
RAW_COLUMNS = ["accession_id", "date", "pangolin_lineage", "location", "division"]


def harmonize_county(clin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw GISAID `location` strings to official NY county names.
    Returns a two-column DataFrame: county, n_sequences.
    """
    counties_gdf = gpd.read_file(SHAPEFILE)
    official_counties = sorted(counties_gdf["NAME"].str.title())

    locality_df = pd.read_csv(LOCALITY_FILE)
    county_names = {
        name.strip().title(): name.strip().title()
        for name in locality_df["County Name"].dropna().unique()
    }
    locality_mapping = {}
    for _, row in locality_df.iterrows():
        county = county_names.get(
            row["County Name"].strip().title(), row["County Name"].strip().title()
        )
        for col in ["City Name", "Town Name", "Village Name"]:
            val = row.get(col)
            if pd.notna(val):
                locality_mapping[val.strip().lower()] = county

    clin_loc = clin_df.copy()
    clin_loc["location"] = (
        clin_loc["location"]
        .str.replace(r"(?i)\s*county$", "", regex=True)
        .str.strip()
        .replace({"New York City": "New York", "St. Lawrence": "St Lawrence"})
    )

    is_matched = clin_loc["location"].isin(official_counties)
    clin_loc.loc[~is_matched, "location"] = (
        clin_loc.loc[~is_matched, "location"]
        .apply(lambda v: locality_mapping.get(v.lower(), v) if pd.notna(v) else v)
    )
    clin_loc["location"] = clin_loc["location"].where(
        clin_loc["location"].isin(official_counties), other=pd.NA
    )

    county_counts = (
        clin_loc["location"].value_counts()
        .reindex(official_counties, fill_value=0)
        .reset_index()
    )
    county_counts.columns = ["county", "n_sequences"]
    return county_counts


def main():
    df = pd.read_csv(RAW_CLINICAL_FILE, sep="\t", compression="gzip", low_memory=False)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected column(s) missing from {RAW_CLINICAL_FILE}: {missing}. "
            "Has the raw GISAID export format changed?"
        )
    df = df[RAW_COLUMNS].copy()
    n_total = len(df)

    # --- Deduplicate exact duplicate records sharing an accession_id ---
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset="accession_id")
    n_dupes_dropped = n_before_dedup - len(df)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    n_dropped_date = df["date"].isna().sum()
    df = df.dropna(subset=["date"])

    # --- Restrict to the study window (by collection date) ---
    in_window = (df["date"] >= START_DATE) & (df["date"] <= END_DATE)
    n_out_of_window = (~in_window).sum()
    df = df.loc[in_window]

    # --- Exclude confirmed out-of-state/out-of-country specimens ---
    is_ny = df["division"].astype(str).str.strip() == "New York"
    n_excluded = (~is_ny).sum()
    df = df.loc[is_ny].drop(columns=["division", "accession_id"])

    # --- Aggregate 1: date x pangolin_lineage counts ---
    lineage_daily = (
        df.groupby(["date", "pangolin_lineage"])
        .size()
        .reset_index(name="n")
        .sort_values(["date", "pangolin_lineage"])
    )
    lineage_daily["date"] = lineage_daily["date"].dt.strftime("%Y-%m-%d")
    lineage_daily.to_csv(LINEAGE_DAILY_COUNTS_FILE, index=False, compression="gzip")

    # --- Aggregate 2: harmonized county counts ---
    county_counts = harmonize_county(df[["location"]].assign(location=df["location"]))
    county_counts.to_csv(COUNTY_COUNTS_FILE, index=False)

    print(f"Read {n_total:,} rows from {RAW_CLINICAL_FILE}")
    print(f"Dropped {n_dupes_dropped:,} exact duplicate rows (same accession_id)")
    print(f"Dropped {n_dropped_date:,} rows with unparseable dates")
    print(f"Excluded {n_out_of_window:,} rows outside study window "
          f"({START_DATE.date()} to {END_DATE.date()})")
    print(f"Excluded {n_excluded:,} rows as non-NY by division")
    print(f"Wrote {len(lineage_daily):,} (date, lineage) rows -> {LINEAGE_DAILY_COUNTS_FILE}")
    print(f"Wrote {len(county_counts):,} county rows -> {COUNTY_COUNTS_FILE} "
          f"(total sequences: {county_counts['n_sequences'].sum():,})")


if __name__ == "__main__":
    main()
