#!/usr/bin/env python3
"""
nanobiosense_pmi.py
===================
NanoBioSense-PMI – Machine Learning Pipeline
---------------------------------------------
Predicts Post-Mortem Interval (PMI, hours) from electrochemical biosensor
amperometric current and environmental covariates, with 95 % confidence
intervals, full evaluation suite, and SHAP-based interpretability.

Sections
--------
  §1  Configuration & Data Loading
  §2  Exploratory Data Analysis (EDA)
  §3  Train / Validation / Test Split
  §4  Preprocessing Pipeline
  §5  Baseline Models
  §6  Hyperparameter Tuning (Random Forest)
  §7  Evaluation
  §8  Confidence Intervals
  §9  Interpretability (Feature Importance, SHAP, PDP)
  §10 Save / Load
  §11 predict_PMI() – field-ready prediction function

Usage
-----
    python nanobiosense_pmi.py                        # default CSV path
    python nanobiosense_pmi.py --data path/to/data.csv

Author : NanoBioSense-PMI Team
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Optional: SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.inspection import PartialDependenceDisplay

warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# [S1]  CONFIGURATION & DATA LOADING
# ==============================================================================

# ---- Defaults (overridable via CLI or direct editing) ---------------------
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "refined_training_data_2000.csv")
FEATURE_COLS = ["current_ua", "ambient_c", "humidity_pct",
                "body_temp_c", "sample_ph"]
TARGET_COL   = "hours"
GROUP_COL    = "id"          # Updated to match india_all_weather_sensor_data.csv

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
RANDOM_STATE = 42

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV, validate expected columns, print summary."""
    print(f"\n{'='*70}")
    print(f"  [S1] Loading data from: {csv_path}")
    print(f"{'='*70}")
    df = pd.read_csv(csv_path)
    required = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")
    print(f"  Shape : {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ==============================================================================
# [S2]  EXPLORATORY DATA ANALYSIS
# ==============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """Print summaries and save exploratory plots."""
    print(f"\n{'='*70}")
    print("  [S2] Exploratory Data Analysis")
    print(f"{'='*70}")

    # -- Basic stats --------------------------------------------------------
    print("\n>> First 5 rows:")
    print(df.head().to_string())
    print("\n>> Descriptive statistics:")
    print(df.describe().to_string())
    print("\n>> Missing values:")
    print(df.isnull().sum().to_string())

    # -- Missing-value heatmap ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df[FEATURE_COLS + [TARGET_COL]].isnull(), cbar=False,
                yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Missing-Value Map")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_missing_values.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eda_missing_values.png")

    # -- PMI distribution ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[TARGET_COL].dropna(), bins=40, color="#2196F3", edgecolor="white")
    axes[0].set_xlabel("PMI (hours)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("PMI Distribution")
    axes[1].hist(np.log1p(df[TARGET_COL].dropna()), bins=40,
                 color="#FF9800", edgecolor="white")
    axes[1].set_xlabel("log(1 + PMI)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log-PMI Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_pmi_distribution.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eda_pmi_distribution.png")

    # -- Feature distributions ----------------------------------------------
    fig, axes = plt.subplots(1, len(FEATURE_COLS), figsize=(4 * len(FEATURE_COLS), 4))
    for ax, col in zip(axes, FEATURE_COLS):
        ax.hist(df[col].dropna(), bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
        ax.set_title(col)
    fig.suptitle("Feature Distributions", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_feature_distributions.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eda_feature_distributions.png")

    # -- Correlation matrix -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature-Target Correlation Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_matrix.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eda_correlation_matrix.png")

    # -- Scatter: each feature vs PMI ---------------------------------------
    fig, axes = plt.subplots(1, len(FEATURE_COLS),
                             figsize=(4 * len(FEATURE_COLS), 4))
    for ax, col in zip(axes, FEATURE_COLS):
        ax.scatter(df[col], df[TARGET_COL], s=8, alpha=0.5, color="#673AB7")
        ax.set_xlabel(col)
        ax.set_ylabel("PMI (hours)")
        ax.set_title(f"{col} vs PMI")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eda_features_vs_pmi.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eda_features_vs_pmi.png")


# ==============================================================================
# [S3]  TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================

def split_data(df: pd.DataFrame):
    """
    Split into train (70 %), validation (15 %), test (15 %).
    Uses GroupShuffleSplit on `case_id` if present to prevent data leakage.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    print(f"\n{'='*70}")
    print("  [S3] Train / Validation / Test Split")
    print(f"{'='*70}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    has_groups = GROUP_COL is not None and GROUP_COL in df.columns
    groups = df[GROUP_COL] if has_groups else None

    if has_groups:
        print(f"  Grouping by '{GROUP_COL}' -- {groups.nunique()} unique groups")
        # First split: 70 % train+val, 30 % test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30,
                                random_state=RANDOM_STATE)
        trainval_idx, test_idx = next(gss1.split(X, y, groups))
        X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
        y_trainval, y_test = y.iloc[trainval_idx], y.iloc[test_idx]
        g_trainval = groups.iloc[trainval_idx]

        # Second split: ~50 % of 30 % -> 15 % val, 15 % implicitly
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5,
                                random_state=RANDOM_STATE)
        # Actually split trainval into train (70%) and val (remainder ~15%)
        gss2b = GroupShuffleSplit(n_splits=1, test_size=0.2143,
                                 random_state=RANDOM_STATE)  # 15/70 ≈ 0.2143
        train_idx, val_idx = next(gss2b.split(X_trainval, y_trainval,
                                              g_trainval))
        X_train = X_trainval.iloc[train_idx]
        X_val   = X_trainval.iloc[val_idx]
        y_train = y_trainval.iloc[train_idx]
        y_val   = y_trainval.iloc[val_idx]
    else:
        print("  No group column — using random stratified split")
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_STATE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.176,       # 0.15/0.85 ≈ 0.176
            random_state=RANDOM_STATE)

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==============================================================================
# [S4]  PREPROCESSING PIPELINE
# ==============================================================================

def _add_engineered_features(X: np.ndarray) -> np.ndarray:
    """
    Feature engineering hook applied AFTER imputation+scaling.
    Input columns order: current_uA, ambient_temp_C, humidity_pct,
                         body_temp_C, sample_pH
    Adds:
      - log(1 + current_uA)
      - current_uA x ambient_temp_C  (interaction)
    """
    log_current = np.log1p(np.abs(X[:, 0])).reshape(-1, 1)
    interaction = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    return np.hstack([X, log_current, interaction])


def build_preprocessor(use_scaling: bool = True,
                       use_feature_engineering: bool = True) -> Pipeline:
    """
    Build a scikit-learn Pipeline:
      1. SimpleImputer (median)
      2. Optional StandardScaler
      3. Optional feature engineering
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if use_scaling:
        steps.append(("scaler", StandardScaler()))

    if use_feature_engineering:
        steps.append(("feature_eng",
                       FunctionTransformer(_add_engineered_features,
                                           validate=False)))

    return Pipeline(steps)


# ==============================================================================
# [S5]  BASELINE MODELS
# ==============================================================================

def train_baselines(preprocessor, X_train, y_train, X_val, y_val):
    """Train baseline models and report validation MAE & RMSE."""
    print(f"\n{'='*70}")
    print("  [S5] Baseline Models")
    print(f"{'='*70}")

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest":     RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        mae  = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2   = r2_score(y_val, preds)
        results[name] = {"pipe": pipe, "mae": mae, "rmse": rmse, "r2": r2}
        print(f"  {name:25s}  MAE={mae:8.2f} h  RMSE={rmse:8.2f} h  R2={r2:.4f}")

    return results


# ==============================================================================
# [S6]  HYPERPARAMETER TUNING  (Random Forest)
# ==============================================================================

def tune_random_forest(preprocessor, X_train, y_train):
    """
    RandomizedSearchCV on Random Forest hyperparameters.
    Optimises for negative MAE (equiv. to minimising MAE).
    Returns the best Pipeline.
    """
    print(f"\n{'='*70}")
    print("  [S6] Hyperparameter Tuning (RandomizedSearchCV)")
    print(f"{'='*70}")

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_distributions = {
        "model__n_estimators":      [100, 200, 300, 500, 700],
        "model__max_depth":         [5, 10, 15, 20, 30, None],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf":  [1, 2, 4, 8],
        "model__max_features":      ["sqrt", "log2", 0.5, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions,
        n_iter=60,
        scoring="neg_mean_absolute_error",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best MAE (CV): {-search.best_score_:.2f} hours")
    print(f"  Best params : {search.best_params_}")

    return search.best_estimator_


# ==============================================================================
# [S7]  EVALUATION
# ==============================================================================

def evaluate_model(pipe, X_test, y_test, X_test_raw_df: pd.DataFrame = None):
    """
    Evaluate best model on the held-out test set.
    - Overall MAE, RMSE, R2
    - Scatter plot: true vs predicted (with y=x)
    - Error by PMI window and temperature band
    """
    print(f"\n{'='*70}")
    print("  [S7] Test-Set Evaluation")
    print(f"{'='*70}")

    preds = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"  MAE  = {mae:.2f} hours")
    print(f"  RMSE = {rmse:.2f} hours")
    print(f"  R2   = {r2:.4f}")

    # -- Scatter: true vs predicted -----------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, preds, s=18, alpha=0.6, color="#1976D2", edgecolors="white",
               linewidth=0.3, label="Predictions")
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x (ideal)")
    ax.set_xlabel("True PMI (hours)", fontsize=12)
    ax.set_ylabel("Predicted PMI (hours)", fontsize=12)
    ax.set_title(f"True vs Predicted PMI  |  MAE={mae:.1f} h, R2={r2:.3f}",
                 fontsize=13)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eval_true_vs_pred.png"), dpi=150)
    plt.close(fig)
    print("  [saved] eval_true_vs_pred.png")

    # -- Error by PMI window ------------------------------------------------
    _error_by_group(y_test, preds, bins=[0, 24, 120, 1e6],
                    labels=["0-24 h", "1-5 days", ">5 days"],
                    group_name="PMI Window",
                    filename="eval_error_by_pmi_window.png")

    # -- Error by temperature band ------------------------------------------
    if X_test_raw_df is not None and "ambient_temp_C" in X_test_raw_df.columns:
        temps = X_test_raw_df["ambient_temp_C"].values
        _error_by_group(y_test, preds, values=temps,
                        bins=[0, 25, 35, 100],
                        labels=["<25 C", "25-35 C", ">35 C"],
                        group_name="Temperature Band",
                        filename="eval_error_by_temp_band.png")

    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": preds}


def _error_by_group(y_true, y_pred, bins, labels, group_name, filename,
                    values=None):
    """Print and plot error stratified by binned groups."""
    if values is None:
        values = np.array(y_true)
    groups = pd.cut(values, bins=bins, labels=labels, right=True)
    df_err = pd.DataFrame({"true": np.array(y_true), "pred": y_pred,
                           "group": groups})

    print(f"\n  Error by {group_name}:")
    summary_rows = []
    for label in labels:
        sub = df_err[df_err["group"] == label]
        if len(sub) == 0:
            continue
        m = mean_absolute_error(sub["true"], sub["pred"])
        r = np.sqrt(mean_squared_error(sub["true"], sub["pred"]))
        print(f"    {label:15s}  n={len(sub):4d}  MAE={m:8.2f} h  RMSE={r:8.2f} h")
        summary_rows.append({"Group": label, "n": len(sub), "MAE": m, "RMSE": r})

    # Bar chart
    if summary_rows:
        sr = pd.DataFrame(summary_rows)
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(sr))
        ax.bar(x - 0.18, sr["MAE"], width=0.35, label="MAE",
               color="#2196F3", edgecolor="white")
        ax.bar(x + 0.18, sr["RMSE"], width=0.35, label="RMSE",
               color="#FF5722", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(sr["Group"])
        ax.set_ylabel("Error (hours)")
        ax.set_title(f"Prediction Error by {group_name}")
        ax.legend()
        for i, row in sr.iterrows():
            ax.text(i - 0.18, row["MAE"] + 0.5, f'{row["MAE"]:.1f}',
                    ha="center", fontsize=8)
            ax.text(i + 0.18, row["RMSE"] + 0.5, f'{row["RMSE"]:.1f}',
                    ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close(fig)
        print(f"  [saved] {filename}")


# ==============================================================================
# [S8]  CONFIDENCE INTERVALS  (per-tree in Random Forest)
# ==============================================================================

def compute_confidence_intervals(pipe, X, confidence: float = 0.95):
    """
    Use individual tree predictions inside the Random Forest to compute
    an approximate confidence interval for each sample.

    Returns
    -------
    ci : pd.DataFrame with columns [mean, std, lower, upper]
    """
    print(f"\n{'='*70}")
    print("  [S8] Confidence Intervals (per-tree)")
    print(f"{'='*70}")

    # Navigate the pipeline to the RF model
    rf_model = pipe.named_steps["model"]
    preprocess = pipe.named_steps["preprocess"]
    X_proc = preprocess.transform(X)

    # Collect predictions from every tree
    tree_preds = np.array([tree.predict(X_proc) for tree in rf_model.estimators_])
    # shape: (n_trees, n_samples)

    means = tree_preds.mean(axis=0)
    stds  = tree_preds.std(axis=0)

    z = 1.96 if confidence == 0.95 else __import__("scipy").stats.norm.ppf(
        0.5 + confidence / 2)

    ci = pd.DataFrame({
        "mean":  means,
        "std":   stds,
        "lower": means - z * stds,
        "upper": means + z * stds,
    })

    # Clip lower bound to 0 (PMI >= 0)
    ci["lower"] = ci["lower"].clip(lower=0)

    print(f"  Computed {confidence*100:.0f} % CI for {len(ci)} samples")
    print(ci.head(10).to_string())

    return ci


# ==============================================================================
# [S9]  INTERPRETABILITY
# ==============================================================================

def plot_feature_importances(pipe):
    """Bar chart of Random Forest feature importances."""
    print(f"\n{'='*70}")
    print("  [S9a] Feature Importances")
    print(f"{'='*70}")

    rf = pipe.named_steps["model"]
    importances = rf.feature_importances_

    # Feature names (original + engineered)
    names = list(FEATURE_COLS) + ["log_current", "current_x_temp"]
    # In case engineered features were not used, trim to length
    names = names[:len(importances)]

    idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), importances[idx], color="#00897B", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("Importance (MDI)")
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "interpret_feature_importances.png"), dpi=150)
    plt.close(fig)
    print("  [saved] interpret_feature_importances.png")


def plot_shap_and_pdp(pipe, X_train):
    """SHAP summary plot + Partial Dependence Plots."""
    print(f"\n{'='*70}")
    print("  [S9b] SHAP & Partial Dependence Plots")
    print(f"{'='*70}")

    preprocess = pipe.named_steps["preprocess"]
    rf_model   = pipe.named_steps["model"]
    X_proc     = preprocess.transform(X_train)

    feature_names = list(FEATURE_COLS) + ["log_current", "current_x_temp"]
    feature_names = feature_names[:X_proc.shape[1]]

    # -- SHAP ---------------------------------------------------------------
    if HAS_SHAP:
        print("  Computing SHAP values (this may take a minute)...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_proc[:300])  # subsample for speed

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_proc[:300],
                          feature_names=feature_names, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "interpret_shap_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  [saved] interpret_shap_summary.png")
    else:
        print("  [skip] SHAP not installed – run `pip install shap` to enable")

    # -- Partial Dependence Plots -------------------------------------------
    print("  Computing Partial Dependence Plots...")
    fig, ax = plt.subplots(figsize=(14, 4))
    # PDP for the first 3 original features via the full pipeline
    # We need to use the raw pipeline for PDP
    features_for_pdp = [0, 1, 2]   # current, ambient_temp, humidity
    try:
        PartialDependenceDisplay.from_estimator(
            pipe, X_train, features=features_for_pdp,
            feature_names=FEATURE_COLS,
            grid_resolution=30, ax=ax,
        )
        fig = plt.gcf()
        fig.suptitle("Partial Dependence Plots", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "interpret_pdp.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  [saved] interpret_pdp.png")
    except Exception as e:
        print(f"  [warn] PDP generation failed: {e}")
        plt.close(fig)


# ==============================================================================
# [S10]  SAVE / LOAD
# ==============================================================================

def save_model(pipe, path: str = None):
    """Persist the full pipeline (preprocessing + model) via joblib."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    joblib.dump(pipe, path)
    print(f"\n  [saved] Model pipeline -> {path}")
    return path


def load_model(path: str = None):
    """Load a previously saved pipeline."""
    if path is None:
        path = os.path.join(OUTPUT_DIR, "best_model.joblib")
    pipe = joblib.load(path)
    print(f"  [loaded] Model pipeline <- {path}")
    return pipe


# ==============================================================================
# [S11]  predict_PMI()  --  FIELD-READY PREDICTION FUNCTION
# ==============================================================================

def predict_PMI(val_current: float,
                val_ambient: float,
                val_humidity: float,
                val_body: float,
                val_ph: float,
                model_path: str = None,
                confidence: float = 0.95,
                _pipe=None) -> dict:
    """
    Predict PMI from a single biosensor measurement.
    Generic parameter names (val_*) are used to allow switching dataset schemas.

    Parameters
    ----------
    val_current      : Amperometric current (first item in FEATURE_COLS).
    val_ambient      : Ambient temperature (second item in FEATURE_COLS).
    val_humidity     : Relative humidity (third item in FEATURE_COLS).
    val_body         : Body surface temperature (fourth item in FEATURE_COLS).
    val_ph           : pH of the sample (fifth item in FEATURE_COLS).
    model_path       : Path to saved .joblib pipeline.
    confidence       : Confidence level for interval (default 0.95).
    _pipe            : Pre-loaded pipeline object.

    Returns
    -------
    dict with PMI point estimate and CI bounds.
    """
    # Load or reuse the pipeline
    if _pipe is None:
        _pipe = load_model(model_path)

    # Build a single-row DataFrame matching the EXACT names used during training
    X = pd.DataFrame([{
        FEATURE_COLS[0]: val_current,
        FEATURE_COLS[1]: val_ambient,
        FEATURE_COLS[2]: val_humidity,
        FEATURE_COLS[3]: val_body,
        FEATURE_COLS[4]: val_ph,
    }])

    # Point prediction
    point = float(_pipe.predict(X)[0])

    # Confidence interval via per-tree predictions
    rf = _pipe.named_steps["model"]
    preprocess = _pipe.named_steps["preprocess"]
    X_proc = preprocess.transform(X)
    tree_preds = np.array([t.predict(X_proc)[0] for t in rf.estimators_])
    mean_pred = tree_preds.mean()
    std_pred  = tree_preds.std()

    z = 1.96 if confidence == 0.95 else __import__("scipy").stats.norm.ppf(
        0.5 + confidence / 2)

    ci_lower = max(0.0, mean_pred - z * std_pred)
    ci_upper = mean_pred + z * std_pred

    return {
        "PMI_hours": round(point, 2),
        "CI_lower":  round(ci_lower, 2),
        "CI_upper":  round(ci_upper, 2),
    }


# ==============================================================================
# MAIN  --  Orchestrates the full pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NanoBioSense-PMI: ML pipeline for PMI estimation")
    parser.add_argument("--data", type=str, default=DEFAULT_CSV,
                        help="Path to CSV dataset")
    parser.add_argument("--no-eda", action="store_true",
                        help="Skip EDA plots")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (use baseline RF)")
    parser.add_argument("--field", action="store_true",
                        help="Run in interactive field prediction mode")
    args = parser.parse_args()

    # -- Interactive Field Mode --------------------------------------------
    if args.field:
        print("\n" + "="*70)
        print("  NanoBioSense-PMI: INTERACTIVE FIELD MODE")
        print("="*70)
        
        model_path = os.path.join(OUTPUT_DIR, "best_model.joblib")
        
        # Train-if-missing logic
        if not os.path.exists(model_path):
            print("\n[!] saved model not found. Training on default dataset now...")
            if not os.path.exists(args.data):
                sys.exit(f"[ERROR] Training data not found at {args.data}. "
                         "Please run 'python generate_synthetic_data.py' first.")
            
            df = load_data(args.data)
            X_train, _, _, y_train, _, _ = split_data(df)
            preprocessor = build_preprocessor(use_scaling=True, 
                                              use_feature_engineering=True)
            # Use baseline RF for speed in field setup
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1))
            ])
            pipe.fit(X_train, y_train)
            save_model(pipe, model_path)
            print("[OK] Model trained and saved.")

        try:
            print("\nPlease enter the following measurements:")
            curr = float(input(f"  >> {FEATURE_COLS[0]:<30}: "))
            temp = float(input(f"  >> {FEATURE_COLS[1]:<30}: "))
            hum  = float(input(f"  >> {FEATURE_COLS[2]:<30}: "))
            body = float(input(f"  >> {FEATURE_COLS[3]:<30}: "))
            ph   = float(input(f"  >> {FEATURE_COLS[4]:<30}: "))
            
            result = predict_PMI(curr, temp, hum, body, ph, model_path=model_path)
            
            print("\n" + "-"*50)
            print(f"  ESTIMATED PMI       : {result['PMI_hours']} hours")
            print(f"  95% CONF. INTERVAL  : [{result['CI_lower']} - {result['CI_upper']}] hours")
            print("-"*50 + "\n")
        except ValueError:
            print("\n[ERROR] Invalid input. Please enter numeric values (e.g., 12.5).")
        except KeyboardInterrupt:
            print("\n\nExiting Field Mode.")
        return

    # -- [S1] Load data ----------------------------------------------------
    df = load_data(args.data)

    # -- [S2] EDA ----------------------------------------------------------
    if not args.no_eda:
        run_eda(df)

    # -- [S3] Split --------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # -- [S4] Preprocessor -------------------------------------------------
    preprocessor = build_preprocessor(use_scaling=True,
                                      use_feature_engineering=True)

    # -- [S5] Baselines ----------------------------------------------------
    baselines = train_baselines(preprocessor, X_train, y_train, X_val, y_val)

    # -- [S6] Tune RF ------------------------------------------------------
    if not args.no_tune:
        best_pipe = tune_random_forest(preprocessor, X_train, y_train)
    else:
        best_pipe = baselines["RandomForest"]["pipe"]
        print("\n  [skip] Tuning skipped -- using baseline Random Forest")

    # -- [S7] Evaluate on test set -----------------------------------------
    test_metrics = evaluate_model(best_pipe, X_test, y_test,
                                  X_test_raw_df=X_test)

    # -- [S8] Confidence intervals -----------------------------------------
    ci = compute_confidence_intervals(best_pipe, X_test)

    # -- [S9] Interpretability ---------------------------------------------
    plot_feature_importances(best_pipe)
    plot_shap_and_pdp(best_pipe, X_train)

    # -- [S10] Save model --------------------------------------------------
    model_path = save_model(best_pipe)

    # -- [S11] Demo prediction ---------------------------------------------
    print(f"\n{'='*70}")
    print("  [S11] Demo: predict_PMI()")
    print(f"{'='*70}")
    demo = predict_PMI(
        val_current   = 12.5,
        val_ambient   = 32.0,
        val_humidity  = 85.0,
        val_body      = 30.5,
        val_ph        = 6.2,
        _pipe=best_pipe,
    )
    print(f"  Input : {FEATURE_COLS[0]}=12.5, {FEATURE_COLS[1]}=32, "
          f"{FEATURE_COLS[2]}=85, {FEATURE_COLS[3]}=30.5, {FEATURE_COLS[4]}=6.2")
    print(f"  Output: PMI = {demo['PMI_hours']:.2f} hours "
          f"[95 % CI: {demo['CI_lower']:.2f} - {demo['CI_upper']:.2f}]")

    print(f"\n{'='*70}")
    print("  [OK]  Pipeline complete. Outputs saved to:", OUTPUT_DIR)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
