"""
generate_synthetic_data.py
==========================
Creates a realistic synthetic CSV for testing the NanoBioSense-PMI pipeline.

Each row represents one biosensor measurement taken from a cadaver at a known
post-mortem interval (PMI).  Multiple samples may come from the same cadaver
(grouped by `case_id`).

The synthetic relationships loosely mimic real-world expectations:
  - Microbial metabolite production (→ current) increases with PMI and temperature.
  - Body surface temperature converges toward ambient temperature over time.
  - Sample pH drifts as decomposition progresses.

Usage:
    python generate_synthetic_data.py          # writes data/synthetic_pmi_data.csv
    python generate_synthetic_data.py --n 1000 # custom sample count
"""

import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = "synthetic_pmi_data.csv"

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(n_samples: int = 500, n_cases: int = 80, seed: int = SEED) -> pd.DataFrame:
    """Return a DataFrame with synthetic biosensor + environmental data."""
    rng = np.random.default_rng(seed)

    # ---- Case assignment (multiple samples per cadaver) -------------------
    case_ids = rng.integers(1, n_cases + 1, size=n_samples)

    # ---- True PMI (hours): log-uniform between 2 h and 240 h (~10 days) --
    pmi_hours = np.exp(rng.uniform(np.log(2), np.log(240), size=n_samples))

    # ---- Environmental covariates -----------------------------------------
    # Tropical ambient temperature: 24 – 40 °C
    ambient_temp = rng.uniform(24, 40, size=n_samples)

    # Relative humidity: 55 – 100 %
    humidity = rng.uniform(55, 100, size=n_samples)

    # Body surface temperature converges to ambient over time
    #   body_temp ≈ ambient + (37 - ambient) * exp(-k * PMI)
    k_cooling = rng.uniform(0.04, 0.08, size=n_samples)         # cooling rate
    body_temp = ambient_temp + (37.0 - ambient_temp) * np.exp(-k_cooling * pmi_hours)
    body_temp += rng.normal(0, 0.5, size=n_samples)              # noise

    # Sample pH drifts from ~7.0 toward acidic (≈5.5) with decomposition
    sample_pH = 7.0 - 1.5 * (1 - np.exp(-0.01 * pmi_hours)) + rng.normal(0, 0.15, size=n_samples)

    # ---- Amperometric current (µA) ----------------------------------------
    # Loosely: current ∝ metabolite production ∝ f(PMI, temperature)
    # Use a non-linear relationship so Random Forest has something to learn.
    log_current = (
        0.8 * np.log1p(pmi_hours)
        + 0.03 * ambient_temp
        + 0.005 * humidity
        - 0.1 * sample_pH
        + rng.normal(0, 0.25, size=n_samples)
    )
    current_uA = np.exp(log_current)  # always positive
    current_uA = np.round(current_uA, 4)

    # ---- Introduce ~3 % missing values at random --------------------------
    df = pd.DataFrame({
        "case_id":        case_ids,
        "current_uA":     current_uA,
        "ambient_temp_C": np.round(ambient_temp, 2),
        "humidity_pct":   np.round(humidity, 2),
        "body_temp_C":    np.round(body_temp, 2),
        "sample_pH":      np.round(sample_pH, 3),
        "PMI_hours":      np.round(pmi_hours, 2),
    })

    # Randomly set ~3 % of feature cells to NaN (not target or case_id)
    feature_cols = ["current_uA", "ambient_temp_C", "humidity_pct", "body_temp_C", "sample_pH"]
    mask = rng.random((n_samples, len(feature_cols))) < 0.03
    for i, col in enumerate(feature_cols):
        df.loc[mask[:, i], col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NanoBioSense-PMI data")
    parser.add_argument("--n", type=int, default=500, help="Number of samples (default 500)")
    parser.add_argument("--cases", type=int, default=80, help="Number of unique cadaver cases")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    df = generate(n_samples=args.n, n_cases=args.cases, seed=args.seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()
