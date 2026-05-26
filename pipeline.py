import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS", "0")

import hashlib

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

_TMP_CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "taecac_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMP_CACHE_ROOT / "xdg-cache"))

import numpy as np
import pandas as pd
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

mpl.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 21,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "lines.linewidth": 3.0,
        "lines.markersize": 9.0,
        "lines.antialiased": True,
        "patch.antialiased": True,
        "text.antialiased": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

import tsfel
import shap
from typing import Iterable, Sequence
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import spearmanr, kendalltau, randint, uniform, loguniform

from catboost import CatBoostRegressor

try:
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    _HAS_STATSFORECAST = True
except Exception:
    _HAS_STATSFORECAST = False

from neuralforecast import NeuralForecast

from neuralforecast.auto import (
    AutoKAN,
    AutoNHITS,
)

AUTO_MODELS = {
    "AutoKAN": AutoKAN,
    "AutoNHITS": AutoNHITS,
}


import joblib

from fm_utils.PrepareDataset import PrepareDataset

warnings.simplefilter(action="ignore", category=FutureWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

DS_COL = "ds"
ID_COL = "unique_id"
Y_COL = "y"

DEFAULT_NF_INPUT_MULT = 2
DEFAULT_NF_BATCH_SIZE = 32
DEFAULT_CV_REFIT = True
DEFAULT_CV_VAL_SIZE = 0
DEFAULT_CV_TEST_SIZE = None

DEFAULT_TSFEL_NAN_COL_THRESH = 0.2
DEFAULT_TSFEL_FS = 1
DEFAULT_META_N_JOBS = -1
DEFAULT_TUNE_ITER = 30
DEFAULT_TUNE_CV_FOLDS = 5

DEFAULT_DS2_STEP = None
DEFAULT_DIFF_WARMUP = 0
DEFAULT_DS2_HOLDOUT_FRAC = 0.6
DEFAULT_DS2_HOLDOUT_MIN_WINDOWS = 1

DEFAULT_RANDOM_REPS = 50
DEFAULT_SHAP_MAX_ROWS = 5000
DEFAULT_SHAP_MAX_DISPLAY = 10
DEFAULT_ONLINE_FIT_WINDOWS = 1
DEFAULT_ONLINE_CALIBRATION_WINDOWS = 1
DEFAULT_ONLINE_N_EXAMPLE_SERIES = 5
DEFAULT_ONLINE_DISTRIBUTION_CLIP_QUANTILE = 0.95

DEFAULT_UQ_PI_LEVEL = 90
DEFAULT_UQ_PI_AGG = "mean"
DEFAULT_UQ_RESID_STAT = "mad"


def _dbg(label: str, msg: str) -> None:
    print(f"[{label}] {msg}", flush=True)


def _hash_config(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _save_paper_figure(fig, outpath: Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        outpath,
        format=(outpath.suffix.lstrip(".") or "pdf"),
        dpi=300,
        bbox_inches="tight",
    )


def _split_plot_title(title: str) -> tuple[str, str]:
    parts = [part.strip() for part in str(title or "").split("|")]
    parts = [part for part in parts if part]
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " | ".join(parts[1:])


def _try_load_nf(path: Path) -> NeuralForecast | None:
    try:
        if hasattr(NeuralForecast, "load"):
            try:
                return NeuralForecast.load(path=str(path))
            except TypeError:
                return NeuralForecast.load(str(path))
    except Exception:
        return None
    return None


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return val if np.isfinite(val) else None
    return value


def _normalize_ds_dtype(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_val = df[DS_COL].iloc[0]
    if isinstance(first_val, (int, np.integer, np.int32, np.int64)) or np.issubdtype(
        df[DS_COL].dtype, np.integer
    ):
        df[DS_COL] = df[DS_COL].astype(int)
    else:
        if not np.issubdtype(df[DS_COL].dtype, np.datetime64):
            df[DS_COL] = pd.to_datetime(df[DS_COL])
    return df


def _infer_freq_and_seasonality(dataset_name: str, dataset_obj):
    frequency = 1 if dataset_name == "M4" else getattr(dataset_obj, "frequency", 1)
    freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
    if isinstance(frequency, str):
        key = frequency.upper()
        frequency = freq_map.get(key, frequency)
    seasonality = int(getattr(dataset_obj, "seasonality", 1))
    return frequency, seasonality


def _canonicalize_frequency(freq):
    if isinstance(freq, (int, np.integer)):
        return int(freq)

    if not isinstance(freq, str):
        return freq

    key = freq.strip().upper()
    alias_map = {
        "H": "H",
        "HOURLY": "H",
        "15T": "15T",
        "15MIN": "15T",
        "15MINUTE": "15T",
        "15MINUTES": "15T",
        "10T": "10T",
        "10MIN": "10T",
        "10MINUTE": "10T",
        "10MINUTES": "10T",
        "D": "D",
        "DAILY": "D",
        "B": "B",
        "BUSINESS": "B",
        "W": "W",
        "WEEKLY": "W",
        "M": "ME",
        "ME": "ME",
        "MS": "ME",
        "MONTHLY": "ME",
        "Q": "QE",
        "QE": "QE",
        "QS": "QE",
        "QUARTERLY": "QE",
        "Y": "YE",
        "YE": "YE",
        "YS": "YE",
        "A": "YE",
        "ANNUAL": "YE",
        "YEARLY": "YE",
    }
    return alias_map.get(key, key)


def _validate_transfer_frequency_match(
    *,
    ds1_label: str,
    ds2_label: str,
    freq1,
    freq2,
) -> None:
    left = _canonicalize_frequency(freq1)
    right = _canonicalize_frequency(freq2)
    if left == right:
        return

    raise ValueError(
        "DS1 -> DS2 transfer requires matching data frequency/group. "
        f"Resolved frequencies differ: DS1 {ds1_label!r} -> {freq1!r}, "
        f"DS2 {ds2_label!r} -> {freq2!r}. "
        "For datasets such as M1/M3/M4/Tourism, use matching groups like "
        "Monthly->Monthly or Quarterly->Quarterly. "
        "For datasetsforecast long-horizon datasets, group is the dataset name "
        "(for example ETTh1, ETTm1, Traffic), and frequency is inferred from it."
    )


def _infer_nf_freq_from_df(df: pd.DataFrame):
    s = df[df[ID_COL] == df[ID_COL].iloc[0]].sort_values(DS_COL)[DS_COL]
    if isinstance(s.iloc[0], (int, np.integer)) or np.issubdtype(s.dtype, np.integer):
        return 1
    s = pd.to_datetime(s)
    f = pd.infer_freq(s)
    if f is None:
        raise RuntimeError(
            "Could not infer pandas frequency from ds; ds may be irregular."
        )
    return f


def _choose_test_size(
    train: pd.DataFrame,
    input_size: int,
    h: int,
    step: int,
    test_size: int | None,
    *,
    val_size: int,
    start_padding_enabled: bool,
) -> int:
    # NeuralForecast CV uses a shared test tail for every series, so the shortest
    # series determines the largest safe test_size.
    min_series_len = int(train.groupby(ID_COL, observed=True).size().min())

    # Before the held-out tail starts, the earliest fold must still have enough prefix to build at least one training window for the auto model. When start padding is enabled, the model can bootstrap from horizon-sized prefixes;
    # otherwise it needs the full explicit input window.
    model_min_history = int(2 * h) if start_padding_enabled else int(input_size)

    # Reserve both the model history and any explicit validation tail requested by NeuralForecast CV before assigning the remainder to test_size.
    reserved_prefix = int(model_min_history + int(val_size))
    max_test_size = int(min_series_len - reserved_prefix)
    if max_test_size < h:
        raise RuntimeError(
            "Series too short for CV: "
            f"min_series_len={min_series_len}, input_size={input_size}, h={h}, "
            f"val_size={val_size}, start_padding_enabled={start_padding_enabled}, "
            f"model_min_history={model_min_history}, reserved_prefix={reserved_prefix}"
        )

    if test_size is None:
        requested_test_size = max_test_size
    else:
        requested_test_size = min(int(test_size), max_test_size)

    aligned_test_size = (int(requested_test_size) // int(step)) * int(step)
    aligned_test_size = max(aligned_test_size, int(step))
    return int(aligned_test_size)


class ForecasterAdapter:
    def __init__(self, nf: NeuralForecast, model_col: str | None = None):
        self.nf = nf
        self.model_col = model_col

    def _normalize_pred_frame(
        self, pred: pd.DataFrame, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        out = pred.copy()
        if ID_COL not in out.columns or DS_COL not in out.columns:
            out = out.reset_index()

        if (
            ID_COL not in out.columns
            and ID_COL in new_df.columns
            and new_df[ID_COL].nunique() == 1
        ):
            out[ID_COL] = new_df[ID_COL].iloc[0]

        missing = [c for c in (ID_COL, DS_COL) if c not in out.columns]
        if missing:
            raise RuntimeError(
                f"Prediction output missing required columns {missing}; got columns {list(out.columns)}"
            )
        return out

    def _infer_point_col(self, pred: pd.DataFrame) -> str:
        cols = [c for c in pred.columns if c not in (ID_COL, DS_COL)]
        if self.model_col is not None and self.model_col in cols:
            return self.model_col
        if len(cols) == 1:
            return cols[0]

        # If prediction intervals are present, prefer the lone non-interval column.
        interval_cols = [
            c
            for c in cols
            if isinstance(c, str)
            and ("-lo-" in c or "-hi-" in c or c.endswith("-lo") or c.endswith("-hi"))
        ]
        non_interval = [c for c in cols if c not in interval_cols]
        if len(non_interval) == 1:
            return non_interval[0]

        median_like = [
            c
            for c in non_interval
            if isinstance(c, str)
            and (
                c.endswith("-median") or c.endswith("_median") or "median" in c.lower()
            )
        ]
        if len(median_like) >= 1:
            for c in median_like:
                if isinstance(c, str) and c.endswith("-median"):
                    return c
            return median_like[0]

        # Fallback: first non-interval column
        if len(non_interval) >= 1:
            return non_interval[0]
        raise RuntimeError(f"Could not infer point prediction column from {cols}")

    def predict(self, h: int, new_df: pd.DataFrame) -> pd.DataFrame:
        pred = self.nf.predict(df=new_df).copy()
        pred = self._normalize_pred_frame(pred, new_df)
        col = self._infer_point_col(pred)
        self.model_col = col
        out = pred[[ID_COL, DS_COL, col]].copy()
        out = out.rename(columns={col: "y_hat"})
        return out

    def predict_with_bounds(
        self, h: int, new_df: pd.DataFrame, level: int
    ) -> pd.DataFrame:
        pred = self.nf.predict(df=new_df, level=[int(level)]).copy()
        pred = self._normalize_pred_frame(pred, new_df)

        lo_suffix = f"-lo-{int(level)}"
        hi_suffix = f"-hi-{int(level)}"
        lo_cols = [
            c for c in pred.columns if isinstance(c, str) and c.endswith(lo_suffix)
        ]
        hi_cols = [
            c for c in pred.columns if isinstance(c, str) and c.endswith(hi_suffix)
        ]

        if not lo_cols or not hi_cols:
            cols = [c for c in pred.columns if c not in (ID_COL, DS_COL)]
            raise RuntimeError(
                f"Missing interval columns for level={level}. Available={cols}"
            )

        lo_col = lo_cols[0]
        prefix = lo_col[: -len(lo_suffix)]
        hi_col = (
            prefix + hi_suffix if (prefix + hi_suffix) in pred.columns else hi_cols[0]
        )

        # Point forecast column varies across NF versions:
        # may be "<prefix>", "<prefix>-median", or "<prefix>-mean".
        point_candidates = [prefix, f"{prefix}-median", f"{prefix}-mean"]
        point_col = next((c for c in point_candidates if c in pred.columns), None)
        if point_col is None:
            # Fall back to generic inference (handles cases like only a single non-interval col)
            point_col = self._infer_point_col(pred)
        self.model_col = point_col

        out = pred[[ID_COL, DS_COL, point_col, lo_col, hi_col]].copy()
        out = out.rename(columns={point_col: "y_hat", lo_col: "y_lo", hi_col: "y_hi"})
        return out


def _fit_and_cv(
    train: pd.DataFrame,
    horizon: int,
    forecast_model: str,
    input_mult: int,
    start_padding_enabled: bool,
    refit: bool,
    val_size: int,
    test_size: int | None,
    *,
    loss=None,
):
    h = int(horizon)
    step = h
    input_size = int(input_mult) * h

    ModelCls = AUTO_MODELS[str(forecast_model)]

    model_kwargs = {
        "input_size": input_size,
        "logger": False,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
    }
    if start_padding_enabled:
        model_kwargs["start_padding_enabled"] = True

    ak_kwargs = {"h": h, "config": model_kwargs, "num_samples": 10}

    if loss is not None:
        try:
            import inspect as _inspect

            if "loss" in _inspect.signature(ModelCls.__init__).parameters:
                ak_kwargs["loss"] = loss
            else:
                model_kwargs["loss"] = loss
        except Exception:
            model_kwargs["loss"] = loss

    model = ModelCls(**ak_kwargs)

    nf = NeuralForecast(
        models=[model],
        freq=_infer_nf_freq_from_df(train),
    )

    ts = _choose_test_size(
        train,
        input_size=input_size,
        h=h,
        step=step,
        test_size=test_size,
        val_size=int(val_size),
        start_padding_enabled=bool(start_padding_enabled),
    )

    t0 = time.perf_counter()
    cv_df = nf.cross_validation(
        train,
        n_windows=None,
        val_size=int(val_size),
        test_size=int(ts),
        step_size=int(step),
        refit=bool(refit),
        verbose=0,
    ).copy()

    nf.fit(df=train)
    fit_time = time.perf_counter() - t0

    need_reset = False
    for col in (ID_COL, DS_COL, "cutoff"):
        if col not in cv_df.columns:
            need_reset = True
            break
    if need_reset:
        cv_df = cv_df.reset_index()

    pred_cols = [c for c in cv_df.columns if c not in (ID_COL, DS_COL, "cutoff", Y_COL)]
    if str(forecast_model) in pred_cols:
        point_col = str(forecast_model)
    elif len(pred_cols) == 1:
        point_col = pred_cols[0]
    else:
        raise RuntimeError(
            f"Could not infer CV prediction column for {forecast_model}. Available={pred_cols}"
        )
    cv_df = cv_df.rename(columns={point_col: "y_hat"})

    implied_windows = int(ts // step)
    return (
        nf,
        ForecasterAdapter(nf, model_col=point_col),
        cv_df,
        float(fit_time),
        int(input_size),
        int(implied_windows),
        int(step),
        int(ts),
    )


def _attach_pointwise_smape(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y_true = df[Y_COL].to_numpy(dtype=float)
    y_pred = df["y_hat"].to_numpy(dtype=float)
    eps = 1e-8
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    df["smape"] = 2.0 * np.abs(y_true - y_pred) / denom
    return df


def _windowize_pointwise(
    df: pd.DataFrame, horizon: int, require_complete: bool = True
) -> pd.DataFrame:
    df = df.copy().sort_values([ID_COL, "cutoff", DS_COL])
    df["step"] = df.groupby([ID_COL, "cutoff"], observed=True).cumcount() + 1
    if "smape" not in df.columns:
        df = _attach_pointwise_smape(df)
    agg = (
        df.groupby([ID_COL, "cutoff"], observed=True)
        .agg(
            window_smape=("smape", "mean"),
            n_steps=("smape", "size"),
            target_start=(DS_COL, "min"),
            target_end=(DS_COL, "max"),
        )
        .reset_index()
    )
    if require_complete:
        agg = agg[agg["n_steps"] == int(horizon)].copy()
    return agg


def _mid_cutoff_per_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[ID_COL, DS_COL]].copy().sort_values([ID_COL, DS_COL])

    def _mid_val(s: pd.Series):
        n = len(s)
        if n <= 1:
            return s.iloc[-1]
        idx = int(np.floor(0.5 * (n - 1)))
        return s.iloc[idx]

    out = (
        df.groupby(ID_COL, observed=True)[DS_COL]
        .apply(_mid_val)
        .reset_index()
        .rename(columns={DS_COL: "mid_cutoff"})
    )
    return out


def _cv_summary_from_pointwise(
    cv_df: pd.DataFrame, horizon: int, yhat_col: str = "y_hat"
):
    return _pointwise_summary_from_predictions(
        cv_df=cv_df,
        horizon=horizon,
        yhat_col=yhat_col,
    )


def _pointwise_summary_from_predictions(
    cv_df: pd.DataFrame, horizon: int, yhat_col: str = "y_hat"
):
    if ID_COL not in cv_df.columns:
        cv_df = cv_df.reset_index()

    required = {ID_COL, DS_COL, "cutoff", Y_COL, yhat_col}
    missing = required - set(cv_df.columns)
    if missing:
        raise KeyError(f"_cv_summary_from_pointwise missing columns: {sorted(missing)}")

    df = (
        cv_df[[ID_COL, DS_COL, "cutoff", Y_COL, yhat_col]]
        .copy()
        .rename(columns={yhat_col: "y_hat"})
    )
    df = _attach_pointwise_smape(df)
    win = _windowize_pointwise(df, horizon=int(horizon), require_complete=True)
    return {
        "mean_pointwise_smape": float(df["smape"].mean()) if len(df) else np.nan,
        "mean_window_smape": float(win["window_smape"].mean()) if len(win) else np.nan,
        "n_pointwise": int(len(df)),
        "n_windows": int(len(win)),
        "n_cutoffs": int(win["cutoff"].nunique()) if len(win) else 0,
    }


def _seasonal_naive_forecast_values(
    history: np.ndarray, *, horizon: int, season_length: int
) -> np.ndarray:
    hist = np.asarray(history, dtype=float)
    horizon = int(horizon)
    if hist.size == 0 or horizon <= 0:
        return np.empty(0, dtype=float)

    lag = max(1, min(int(season_length), hist.size))
    template = hist[-lag:]
    return np.resize(template, horizon).astype(float)


def _seasonal_naive_holdout_pointwise(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    horizon: int,
    season_length: int,
) -> pd.DataFrame:
    train = train[[ID_COL, DS_COL, Y_COL]].copy().sort_values([ID_COL, DS_COL])
    test = test[[ID_COL, DS_COL, Y_COL]].copy().sort_values([ID_COL, DS_COL])

    rows = []
    for uid, g_train in train.groupby(ID_COL, observed=True):
        g_test = test[test[ID_COL] == uid].sort_values(DS_COL)
        if len(g_train) == 0 or len(g_test) < int(horizon):
            continue

        future = g_test.head(int(horizon)).copy()
        preds = _seasonal_naive_forecast_values(
            g_train[Y_COL].to_numpy(dtype=float),
            horizon=int(horizon),
            season_length=int(season_length),
        )
        cutoff = g_train[DS_COL].max()

        out = future[[ID_COL, DS_COL, Y_COL]].copy()
        out["cutoff"] = cutoff
        out["y_hat"] = preds[: len(future)]
        rows.append(out[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]])

    if not rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"])

    return pd.concat(rows, axis=0, ignore_index=True)


def _rolling_seasonal_naive_pointwise_preds(
    df_full: pd.DataFrame,
    cutoffs: pd.DataFrame,
    *,
    horizon: int,
    season_length: int,
    max_series: int | None,
    max_windows_per_series: int | None,
) -> pd.DataFrame:
    df_full = df_full[[ID_COL, DS_COL, Y_COL]].copy().sort_values([ID_COL, DS_COL])
    cutoffs = cutoffs[[ID_COL, "cutoff"]].copy().sort_values([ID_COL, "cutoff"])

    out_rows = []
    uids = list(df_full[ID_COL].unique())
    if max_series is not None:
        uids = uids[: int(max_series)]

    for uid in uids:
        g = df_full[df_full[ID_COL] == uid].sort_values(DS_COL)
        cu = cutoffs[cutoffs[ID_COL] == uid].sort_values("cutoff")
        if max_windows_per_series is not None:
            cu = cu.head(int(max_windows_per_series))
        if cu.empty:
            continue

        for _, r in cu.iterrows():
            cutoff = r["cutoff"]
            prefix = g[g[DS_COL] <= cutoff][[ID_COL, DS_COL, Y_COL]].copy()
            future = g[g[DS_COL] > cutoff][[ID_COL, DS_COL, Y_COL]].head(int(horizon))
            if len(prefix) == 0 or len(future) < int(horizon):
                continue

            preds = _seasonal_naive_forecast_values(
                prefix[Y_COL].to_numpy(dtype=float),
                horizon=int(horizon),
                season_length=int(season_length),
            )
            out = future[[ID_COL, DS_COL, Y_COL]].copy()
            out["cutoff"] = cutoff
            out["y_hat"] = preds[: len(future)]
            out_rows.append(out[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]])

    if not out_rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"])

    return pd.concat(out_rows, axis=0, ignore_index=True)


def _summary_from_pointwise_subset(
    pointwise_df: pd.DataFrame | None,
    cutoffs_df: pd.DataFrame,
    *,
    horizon: int,
) -> dict | None:
    if pointwise_df is None or len(pointwise_df) == 0:
        return None
    if cutoffs_df is None or len(cutoffs_df) == 0:
        return None

    keep = cutoffs_df[[ID_COL, "cutoff"]].drop_duplicates()
    df = pointwise_df.merge(keep, on=[ID_COL, "cutoff"], how="inner")
    if len(df) == 0:
        return None
    return _pointwise_summary_from_predictions(
        df, horizon=int(horizon), yhat_col="y_hat"
    )


def _filter_cutoffs_by_history(
    df_long: pd.DataFrame, cutoffs_df: pd.DataFrame, l_meta: int
) -> pd.DataFrame:
    if cutoffs_df is None or len(cutoffs_df) == 0:
        return cutoffs_df

    df_long = df_long[[ID_COL, DS_COL]].copy().sort_values([ID_COL, DS_COL])
    cutoffs_df = cutoffs_df[[ID_COL, "cutoff"]].copy().sort_values([ID_COL, "cutoff"])

    ds_is_dt = np.issubdtype(df_long[DS_COL].dtype, np.datetime64) or isinstance(
        df_long[DS_COL].iloc[0], pd.Timestamp
    )
    if ds_is_dt:
        df_long[DS_COL] = pd.to_datetime(df_long[DS_COL]).to_numpy(
            dtype="datetime64[ns]"
        )
        cutoffs_df["cutoff"] = pd.to_datetime(cutoffs_df["cutoff"]).to_numpy(
            dtype="datetime64[ns]"
        )

    out_rows = []
    for uid, cu in cutoffs_df.groupby(ID_COL, observed=True):
        g = df_long[df_long[ID_COL] == uid][DS_COL].to_numpy()
        if g.size == 0:
            continue
        cvals = cu["cutoff"].to_numpy()
        counts = np.searchsorted(g, cvals, side="right")
        keep = counts >= int(l_meta)
        if np.any(keep):
            out_rows.append(cu.loc[keep, [ID_COL, "cutoff"]])

    if not out_rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff"])
    return pd.concat(out_rows, axis=0, ignore_index=True)


def _build_sliding_windows_long(
    df_long: pd.DataFrame,
    lags: int,
    horizon: int,
    id_col: str,
    ds_col: str,
    y_col: str,
    cutoffs_df: pd.DataFrame | None = None,
):
    df_sorted = df_long[[id_col, ds_col, y_col]].copy().sort_values([id_col, ds_col])
    windows = []
    meta_rows = []
    W = int(lags) + int(horizon)

    ds_is_dt = np.issubdtype(df_sorted[ds_col].dtype, np.datetime64) or isinstance(
        df_sorted[ds_col].iloc[0], pd.Timestamp
    )
    if ds_is_dt:
        df_sorted[ds_col] = pd.to_datetime(df_sorted[ds_col]).to_numpy(
            dtype="datetime64[ns]"
        )

    cut_map = None
    if cutoffs_df is not None and len(cutoffs_df) > 0:
        cutoffs_df = cutoffs_df[[id_col, "cutoff"]].copy()
        if ds_is_dt:
            cutoffs_df["cutoff"] = pd.to_datetime(cutoffs_df["cutoff"]).to_numpy(
                dtype="datetime64[ns]"
            )
        cut_map = (
            cutoffs_df.groupby(id_col, observed=True)["cutoff"]
            .apply(lambda s: set(s.to_numpy()))
            .to_dict()
        )

    for uid, g in df_sorted.groupby(id_col, observed=True):
        vals = g[y_col].to_numpy(dtype=float)
        ds_vals = g[ds_col].to_numpy()
        T = len(vals)
        first_end = W - 1
        if first_end >= T:
            continue

        allowed = None
        if cut_map is not None:
            allowed = cut_map.get(uid, None)
            if not allowed:
                continue

        w_idx = 0
        for end in range(first_end, T):
            if allowed is not None and ds_vals[end] not in allowed:
                continue
            start = end - W + 1
            if start < 0:
                continue
            window = vals[start : end + 1]
            if window.shape[0] != W:
                continue
            windows.append(window.astype(np.float32, copy=False))
            meta_rows.append({id_col: uid, "cutoff": ds_vals[end], "window_idx": w_idx})
            w_idx += 1

    meta_df = pd.DataFrame(meta_rows)
    return windows, meta_df


def _extract_tsfel_features_from_windows(
    windows,
    fs: int = 1,
    tsfel_wanted=None,
    drop_features=None,
    nan_col_thresh: float | None = 0.2,
    standardize: bool = False,
    eps: float = 1e-8,
):
    W = len(windows[0]) if windows else 0
    _dbg("TSFEL", f"Extracting TSFEL features from {len(windows)} windows of size {W}")

    cfg = {}
    cfg.update(tsfel.get_features_by_domain("statistical"))
    cfg.update(tsfel.get_features_by_domain("temporal"))
    if W >= 10:
        cfg.update(tsfel.get_features_by_domain("spectral"))

    wanted_set = set(tsfel_wanted) if tsfel_wanted is not None else None
    if wanted_set is not None:
        for dom in list(cfg.keys()):
            cfg[dom] = {
                name: params for name, params in cfg[dom].items() if name in wanted_set
            }

    rows = []
    for w in windows:
        x = np.asarray(w, dtype=float)
        if standardize:
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            x = (x - mu) / (sd + eps)
        fb = tsfel.time_series_features_extractor(cfg, x, fs=int(fs), verbose=0)
        rows.append(fb)

    if not rows:
        return pd.DataFrame()

    feats = pd.concat(rows, axis=0, ignore_index=True)
    feats.columns = [c.replace("0_", "") for c in feats.columns]

    if wanted_set is not None:
        feats = feats[[c for c in feats.columns if c in wanted_set]]

    if drop_features is not None:
        drop_set = set(drop_features)
        feats = feats[[c for c in feats.columns if c not in drop_set]]

    feats = feats.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    if nan_col_thresh is not None:
        nan_rate = feats.isna().mean()
        keep_cols = nan_rate[nan_rate <= float(nan_col_thresh)].index.tolist()
        feats = feats[keep_cols]

    return feats


def _tsfel_context_features(
    df_long: pd.DataFrame,
    l_meta: int,
    cutoffs_df: pd.DataFrame,
    nan_col_thresh: float | None,
    fs: int = DEFAULT_TSFEL_FS,
    standardize: bool = False,
) -> pd.DataFrame:
    cutoffs_df = _filter_cutoffs_by_history(df_long, cutoffs_df, l_meta=int(l_meta))
    if cutoffs_df is None or len(cutoffs_df) == 0:
        return pd.DataFrame(columns=[ID_COL, "cutoff"])

    windows, meta = _build_sliding_windows_long(
        df_long=df_long,
        lags=int(l_meta),
        horizon=0,
        id_col=ID_COL,
        ds_col=DS_COL,
        y_col=Y_COL,
        cutoffs_df=cutoffs_df,
    )
    feats = _extract_tsfel_features_from_windows(
        windows=windows,
        fs=int(fs),
        nan_col_thresh=nan_col_thresh,
        standardize=bool(standardize),
    )
    if len(feats) == 0 or len(meta) == 0:
        return pd.DataFrame(columns=[ID_COL, "cutoff"])

    meta = meta.reset_index(drop=True)
    feats = feats.reset_index(drop=True)
    feats.columns = [f"tsfel_{c}" for c in feats.columns]
    return pd.concat([meta[[ID_COL, "cutoff"]], feats], axis=1)


def _strip_tsfel_prefix(names: list[str]) -> list[str]:
    return [n.replace("tsfel_", "", 1) if n.startswith("tsfel_") else n for n in names]


def _unwrap_meta_pipeline(meta_model):
    if hasattr(meta_model, "named_steps"):
        steps = meta_model.named_steps
        imputer = steps.get("imputer", None)
        scaler = steps.get("scaler", None)
        reg = (
            steps.get("reg", None)
            or steps.get("model", None)
            or steps.get("clf", None)
            or meta_model
        )

        def transform_X(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            if imputer is not None:
                X = imputer.transform(X)
            if scaler is not None:
                X = scaler.transform(X)
            return X

        return reg, transform_X

    def transform_X(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)

    return meta_model, transform_X


def _tree_shap_values(meta_model, X_df: pd.DataFrame):
    model, transform_X = _unwrap_meta_pipeline(meta_model)
    X = X_df.to_numpy(dtype=float)
    Xp = transform_X(X)

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    sv = explainer.shap_values(Xp)
    base = explainer.expected_value

    if isinstance(sv, list):
        if len(sv) > 1:
            sv_use = np.asarray(sv[1], dtype=float)
            base_use = base[1] if isinstance(base, (list, np.ndarray)) else base
        else:
            sv_use = np.asarray(sv[0], dtype=float)
            base_use = base[0] if isinstance(base, (list, np.ndarray)) else base
    else:
        sv_use = np.asarray(sv, dtype=float)
        if isinstance(base, (list, np.ndarray)):
            base_use = float(np.asarray(base).ravel()[0])
        else:
            base_use = float(base)

    return explainer, Xp, sv_use, float(base_use)


def save_global_shap_plots(
    *,
    meta_clf,
    meta_df: pd.DataFrame,
    feature_cols: list[str],
    outdir: str | Path,
    tag: str,
    title: str | None = None,
    max_display: int = 10,
    max_rows: int = 5000,
    seed: int = 0,
):
    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = meta_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["window_smape"], how="any")
    if len(df) == 0:
        return None

    if max_rows is not None and len(df) > int(max_rows):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(len(df), size=int(max_rows), replace=False)
        df = df.iloc[idx].copy()

    X_df = df[feature_cols].copy()
    plot_names = _strip_tsfel_prefix(feature_cols)

    _, Xp, shap_vals, _ = _tree_shap_values(meta_clf, X_df)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    imp = (
        pd.DataFrame({"feature": plot_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(plots_dir / f"shap_global_importance_{tag}.csv", index=False)

    plt.figure(figsize=(10, 4.8))
    shap.summary_plot(
        shap_vals,
        Xp,
        feature_names=plot_names,
        max_display=int(max_display),
        show=False,
    )
    plt.title(
        title if title is not None else f"Global SHAP summary | {tag}",
        pad=10,
    )
    plt.tight_layout()
    _save_paper_figure(plt.gcf(), plots_dir / f"shap_summary_{tag}.pdf")
    plt.close()

    return imp


def _compute_spearman(p: np.ndarray, e: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    e = np.asarray(e, dtype=float)
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    if p.size < 3:
        return {"spearman_rho_p_vs_smape": np.nan, "spearman_pval": np.nan}
    try:
        rho, pval = spearmanr(p, e)
        return {"spearman_rho_p_vs_smape": float(rho), "spearman_pval": float(pval)}
    except Exception:
        return {"spearman_rho_p_vs_smape": np.nan, "spearman_pval": np.nan}


def _compute_kendall(p: np.ndarray, e: np.ndarray) -> dict:
    p = np.asarray(p, dtype=float)
    e = np.asarray(e, dtype=float)
    ok = np.isfinite(p) & np.isfinite(e)
    p, e = p[ok], e[ok]
    if p.size < 3:
        return {"kendall_tau_p_vs_smape": np.nan, "kendall_pval": np.nan}
    try:
        tau, pval = kendalltau(p, e)
        return {"kendall_tau_p_vs_smape": float(tau), "kendall_pval": float(pval)}
    except Exception:
        return {"kendall_tau_p_vs_smape": np.nan, "kendall_pval": np.nan}


class _FenwickTree:
    def __init__(self, n: int):
        self.n = int(max(1, n))
        self.bit = np.zeros(self.n + 1, dtype=float)

    def add(self, idx: int, value: float) -> None:
        i = int(idx)
        while i <= self.n:
            self.bit[i] += float(value)
            i += i & -i

    def prefix_sum(self, idx: int) -> float:
        s = 0.0
        i = int(idx)
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return float(s)


def _compute_c_index(score: np.ndarray, err: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(score) & np.isfinite(err)
    score, err = score[ok], err[ok]
    n = int(score.size)
    if n < 2:
        return float("nan")

    _, score_inv = np.unique(score, return_inverse=True)
    score_rank = score_inv.astype(int) + 1

    order = np.argsort(err, kind="mergesort")
    err_sorted = err[order]
    rank_sorted = score_rank[order]

    tree = _FenwickTree(int(score_rank.max()))
    seen = 0
    comparable = 0.0
    concordant = 0.0

    i = 0
    while i < n:
        j = i + 1
        while j < n and err_sorted[j] == err_sorted[i]:
            j += 1

        for k in range(i, j):
            r = int(rank_sorted[k])
            less = tree.prefix_sum(r - 1)
            leq = tree.prefix_sum(r)
            equal = leq - less
            concordant += less + 0.5 * equal
            comparable += float(seen)

        for k in range(i, j):
            tree.add(int(rank_sorted[k]), 1.0)
            seen += 1

        i = j

    if comparable <= 0:
        return float("nan")
    return float(concordant / comparable)


def _compute_topk_lift(
    score: np.ndarray,
    err: np.ndarray,
    fracs: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, dict]:
    score = np.asarray(score, dtype=float)
    err = np.asarray(err, dtype=float)
    ok = np.isfinite(score) & np.isfinite(err)
    score, err = score[ok], err[ok]
    n = int(score.size)
    if n == 0:
        return {
            f"{frac:.2f}": {
                "k": 0,
                "mean_window_smape": np.nan,
                "lift_vs_all": np.nan,
            }
            for frac in fracs
        }

    base = float(np.mean(err))
    order = np.argsort(-score, kind="mergesort")
    out: dict[str, dict] = {}
    for frac in fracs:
        k = int(max(1, np.ceil(float(frac) * n)))
        top_err = err[order[:k]]
        top_mean = float(np.mean(top_err)) if top_err.size else np.nan
        lift = top_mean / base if np.isfinite(base) and base > 0 else np.nan
        out[f"{frac:.2f}"] = {
            "k": int(k),
            "mean_window_smape": top_mean,
            "lift_vs_all": _safe_float(lift),
        }
    return out


def _summarize_metamodel_ranking(
    meta_df: pd.DataFrame,
    *,
    score_col: str = "u_hat",
    err_col: str = "window_smape",
) -> dict:
    if (
        meta_df is None
        or len(meta_df) == 0
        or score_col not in meta_df.columns
        or err_col not in meta_df.columns
    ):
        return {
            "n_rows": 0,
            "spearman": {"spearman_rho_p_vs_smape": np.nan, "spearman_pval": np.nan},
            "kendall": {"kendall_tau_p_vs_smape": np.nan, "kendall_pval": np.nan},
            "c_index": np.nan,
            "topk_lift": _compute_topk_lift(
                np.array([], dtype=float), np.array([], dtype=float)
            ),
        }

    score = pd.to_numeric(meta_df[score_col], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(meta_df[err_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(score) & np.isfinite(err)
    score = score[ok]
    err = err[ok]

    return {
        "n_rows": int(score.size),
        "spearman": _compute_spearman(score, err),
        "kendall": _compute_kendall(score, err),
        "c_index": _safe_float(_compute_c_index(score, err)),
        "topk_lift": _compute_topk_lift(score, err),
    }


def _random_reference_ranking_summary(
    err: np.ndarray,
    fracs: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    n = int(err.size)
    base = float(np.mean(err)) if n else np.nan
    topk = {}
    for frac in fracs:
        k = int(max(1, np.ceil(float(frac) * n))) if n else 0
        topk[f"{frac:.2f}"] = {
            "k": int(k),
            "mean_window_smape": _safe_float(base),
            "lift_vs_all": _safe_float(
                1.0 if np.isfinite(base) and base > 0 else np.nan
            ),
        }
    return {
        "n_rows": int(n),
        "spearman": {"spearman_rho_p_vs_smape": 0.0, "spearman_pval": np.nan},
        "kendall": {"kendall_tau_p_vs_smape": 0.0, "kendall_pval": np.nan},
        "c_index": 0.5 if n >= 2 else np.nan,
        "topk_lift": topk,
    }


def _summarize_available_score_rankings(
    meta_df: pd.DataFrame,
    *,
    err_col: str = "window_smape",
) -> dict[str, dict]:
    if meta_df is None or len(meta_df) == 0 or err_col not in meta_df.columns:
        return {}

    out: dict[str, dict] = {}
    for name, col in _available_reject_scores(meta_df):
        if col not in meta_df.columns:
            continue
        out[str(name)] = _summarize_metamodel_ranking(
            meta_df,
            score_col=col,
            err_col=err_col,
        )

    err = pd.to_numeric(meta_df[err_col], errors="coerce").to_numpy(dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return out

    oracle_df = pd.DataFrame(
        {
            "oracle_score": err,
            err_col: err,
        }
    )
    out["oracle"] = _summarize_metamodel_ranking(
        oracle_df,
        score_col="oracle_score",
        err_col=err_col,
    )
    out["random"] = _random_reference_ranking_summary(err)
    return out


def _make_reject_grid() -> np.ndarray:
    # Fraction of windows to reject (remove), in [0, 1).
    # Includes 0.0 so the first point corresponds to keeping all windows.
    return np.array(
        [
            0.00,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.99,
        ],
        dtype=float,
    )


def _compute_aurc_from_curve(coverage: np.ndarray, risk: np.ndarray) -> float:
    cov = np.asarray(coverage, dtype=float)
    rsk = np.asarray(risk, dtype=float)
    ok = np.isfinite(cov) & np.isfinite(rsk)
    cov, rsk = cov[ok], rsk[ok]
    if cov.size < 2:
        return float("nan")
    order = np.argsort(cov)
    cov, rsk = cov[order], rsk[order]
    cov = np.clip(cov, 0.0, 1.0)
    return float(np.trapezoid(rsk, cov))


def _serialize_risk_coverage_curve(rc: pd.DataFrame) -> list[dict]:
    if rc is None or len(rc) == 0:
        return []

    cols = [
        "p_reject",
        "coverage",
        "reject_rate",
        "base_risk_mean_smape",
        "risk_model",
        "risk_random",
        "risk_oracle",
        "co_error_model",
        "co_error_random",
        "aurc_model",
        "aurc_random",
        "aurc_oracle",
        "auco_model",
        "auco_random",
        "error_drop_model",
        "n_total",
        "n_accept_model",
        "n_reject_model",
    ]
    cols = [c for c in cols if c in rc.columns]
    payload = []
    for row in rc[cols].to_dict(orient="records"):
        payload.append({k: _safe_float(v) for k, v in row.items()})
    return payload


def _curve_plot_style(name: str) -> dict[str, object]:
    styles = {
        "random": {
            "color": "#4C78A8",
            "marker": "o",
            "linestyle": ":",
        },
        "oracle": {
            "color": "#F58518",
            "marker": "s",
            "linestyle": "--",
        },
        "u_hat": {
            "color": "#54A24B",
            "marker": "D",
            "linestyle": "-",
        },
        "pi_width": {
            "color": "#E45756",
            "marker": "^",
            "linestyle": "-.",
        },
        "resid_scale": {
            "color": "#B279A2",
            "marker": "P",
            "linestyle": ":",
        },
        "err_var": {
            "color": "#9D755D",
            "marker": "X",
            "linestyle": "-",
        },
    }
    return styles.get(
        str(name),
        {
            "color": None,
            "marker": "o",
            "linestyle": "-",
            "markersize": 8.0,
        },
    )


def _reject_score_display_name(name: str) -> str:
    return {
        "u_hat": "Metamodel",
        "pi_width": "PI width",
        "resid_scale": "Residual scale",
        "err_var": "Residual variance",
        "random": "Random",
        "oracle": "Oracle",
    }.get(str(name), str(name))


def _reject_score_tick_label(name: str) -> str:
    return {
        "u_hat": "Ours",
        "pi_width": "PI width",
        "resid_scale": "Resid.\nscale",
        "err_var": "Err.\nvar.",
        "random": "Random",
        "oracle": "Oracle",
    }.get(str(name), str(name).replace("_", "\n"))


def _online_reject_col(name: str) -> str:
    if str(name) == "u_hat":
        return "reject_meta_model"
    if str(name) == "oracle":
        return "reject_oracle_q"
    if str(name) == "random":
        return "reject_random_q"
    return f"reject_{name}"


def _online_accept_col(name: str) -> str:
    if str(name) == "u_hat":
        return "accept_meta_model"
    if str(name) == "oracle":
        return "accept_oracle_q"
    if str(name) == "random":
        return "accept_random_q"
    return f"accept_{name}"


def _online_score_order(names: Iterable[str]) -> list[str]:
    order = {
        "u_hat": 0,
        "pi_width": 1,
        "resid_scale": 2,
        "err_var": 3,
        "random": 4,
        "oracle": 5,
    }
    seen = set()
    out = []
    for name in sorted((str(n) for n in names), key=lambda n: (order.get(n, 999), n)):
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _bool_array(values) -> np.ndarray:
    if isinstance(values, pd.Series):
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).to_numpy(dtype=bool)
        normalized = values.astype(str).str.strip().str.lower()
        return normalized.isin({"true", "1", "t", "yes", "y"}).to_numpy(dtype=bool)
    arr = np.asarray(values)
    if arr.dtype == bool:
        return arr.astype(bool)
    normalized = pd.Series(arr).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "t", "yes", "y"}).to_numpy(dtype=bool)


def _risk_coverage_curve(
    meta_all: pd.DataFrame,
    *,
    score_col: str = "score",
    err_col: str = "window_smape",
    grid: np.ndarray | None = None,
    random_reps: int = 50,
    random_seed: int = 0,
) -> pd.DataFrame:
    df = meta_all[[score_col, err_col]].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df[err_col] = pd.to_numeric(df[err_col], errors="coerce")
    df = df.dropna(subset=[score_col, err_col]).copy()
    if len(df) == 0:
        return pd.DataFrame()

    if grid is None:
        grid = _make_reject_grid()

    score = df[score_col].to_numpy(dtype=float)
    e = df[err_col].to_numpy(dtype=float)

    order_p_desc = np.argsort(-score)
    order_e_desc = np.argsort(-e)

    n_total = int(len(df))
    base_risk = float(np.mean(e)) if n_total else np.nan

    rng = np.random.default_rng(int(random_seed))
    rows = []

    for p_reject in grid:
        # p_reject denotes the fraction of windows to reject (remove), not a score-quantile.
        p_reject = float(np.clip(p_reject, 0.0, 1.0))
        if n_total <= 1:
            continue

        n_rej_model = int(np.floor(p_reject * n_total + 1e-12))
        n_rej_model = int(np.clip(n_rej_model, 0, n_total - 1))
        n_acc_model = int(n_total - n_rej_model)

        reject_rate = n_rej_model / n_total
        coverage = n_acc_model / n_total

        if n_rej_model == 0:
            reject_mask_model = np.zeros(n_total, dtype=bool)
            thr_model = float("nan")
        else:
            idx_rej = order_p_desc[:n_rej_model]
            reject_mask_model = np.zeros(n_total, dtype=bool)
            reject_mask_model[idx_rej] = True
            thr_model = float(np.min(score[idx_rej])) if idx_rej.size else float("nan")
        accept_mask_model = ~reject_mask_model

        risk_model = float(np.mean(e[accept_mask_model])) if n_acc_model else np.nan

        if n_rej_model == 0:
            thr_oracle = float("nan")
            risk_oracle = base_risk
        else:
            idx_oracle_rej = order_e_desc[:n_rej_model]
            accept_mask_oracle = np.ones(n_total, dtype=bool)
            accept_mask_oracle[idx_oracle_rej] = False
            thr_oracle = (
                float(np.min(e[idx_oracle_rej]))
                if idx_oracle_rej.size
                else float("nan")
            )
            risk_oracle = (
                float(np.mean(e[accept_mask_oracle]))
                if np.any(accept_mask_oracle)
                else np.nan
            )

        if n_rej_model == 0:
            risk_random = base_risk
        elif 0 < n_rej_model < n_total:
            risks = []
            for _ in range(int(random_reps)):
                idx_rej = rng.choice(n_total, size=n_rej_model, replace=False)
                mask_rej = np.zeros(n_total, dtype=bool)
                mask_rej[idx_rej] = True
                mask_acc = ~mask_rej
                risks.append(float(np.mean(e[mask_acc])))
            risk_random = float(np.mean(risks)) if risks else np.nan
        else:
            risk_random = np.nan

        rows.append(
            {
                "p_reject": p_reject,
                "thr_model": thr_model,
                "thr_oracle": thr_oracle,
                "coverage": coverage,
                "reject_rate": reject_rate,
                "base_risk_mean_smape": base_risk,
                "risk_model": risk_model,
                "risk_random": risk_random,
                "risk_oracle": risk_oracle,
                "n_total": n_total,
                "n_accept_model": n_acc_model,
                "n_reject_model": n_rej_model,
            }
        )

    rc = pd.DataFrame(rows)

    aurc_model = _compute_aurc_from_curve(
        rc["coverage"].to_numpy(), rc["risk_model"].to_numpy()
    )
    aurc_random = _compute_aurc_from_curve(
        rc["coverage"].to_numpy(), rc["risk_random"].to_numpy()
    )
    aurc_oracle = _compute_aurc_from_curve(
        rc["coverage"].to_numpy(), rc["risk_oracle"].to_numpy()
    )

    rc["aurc_model"] = aurc_model
    rc["aurc_random"] = aurc_random
    rc["aurc_oracle"] = aurc_oracle

    rc["co_error_model"] = rc["risk_model"].to_numpy(dtype=float) - rc[
        "risk_oracle"
    ].to_numpy(dtype=float)
    rc["co_error_random"] = rc["risk_random"].to_numpy(dtype=float) - rc[
        "risk_oracle"
    ].to_numpy(dtype=float)

    auco_model = _compute_aurc_from_curve(
        rc["coverage"].to_numpy(), rc["co_error_model"].to_numpy()
    )
    auco_random = _compute_aurc_from_curve(
        rc["coverage"].to_numpy(), rc["co_error_random"].to_numpy()
    )

    rc["auco_model"] = auco_model
    rc["auco_random"] = auco_random

    # Error Drop: ratio between the risk at maximum coverage (keep-all) and minimum coverage (most selective).
    try:
        cov = pd.to_numeric(rc["coverage"], errors="coerce")
        idx_hi = cov.idxmax()
        idx_lo = cov.idxmin()
        risk_hi = float(rc.loc[idx_hi, "risk_model"])
        risk_lo = float(rc.loc[idx_lo, "risk_model"])
        if np.isfinite(risk_hi) and np.isfinite(risk_lo) and risk_lo > 0:
            err_drop = risk_hi / risk_lo
        else:
            err_drop = float("nan")
    except Exception:
        err_drop = float("nan")

    rc["error_drop_model"] = err_drop

    return rc


def _plot_risk_coverage_curves_multi(
    curves: dict[str, pd.DataFrame],
    *,
    outpath: Path | None = None,
    title: str = "Risk-Coverage (window-level sMAPE)",
    normalize: bool = False,
    include_oracle_random: bool = True,
    oracle_from: str | None = "u_hat",
    show_legend: bool = True,
) -> None:
    if not curves:
        return

    def _pretty(n: str) -> str:
        return {
            "u_hat": "Metamodel",
            "pi_width": "PI width",
            "resid_scale": "Residual scale",
            "err_var": "Residual variance",
        }.get(n, n)

    # Pick a reference curve for oracle/random (they should be identical if same eval set/grid)
    ref = None
    if (
        oracle_from is not None
        and oracle_from in curves
        and len(curves[oracle_from]) > 0
    ):
        ref = curves[oracle_from]
    else:
        for _, rc in curves.items():
            if rc is not None and len(rc) > 0:
                ref = rc
                break

    title_main, title_note = _split_plot_title(title)
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111)

    # Plot oracle/random once
    if include_oracle_random and ref is not None and len(ref) > 0:
        cov = ref["coverage"].to_numpy(dtype=float)
        reject_rate = (
            ref["reject_rate"].to_numpy(dtype=float)
            if "reject_rate" in ref.columns
            else 1.0 - cov
        )
        y_rand = ref["risk_random"].to_numpy(dtype=float)
        y_orac = ref["risk_oracle"].to_numpy(dtype=float)

        if normalize:
            try:
                idx_hi = np.nanargmax(cov)
                base = float(ref["risk_model"].iloc[idx_hi])
                if np.isfinite(base) and base > 0:
                    y_rand = y_rand / base
                    y_orac = y_orac / base
            except Exception:
                pass

        rand_style = _curve_plot_style("random")
        oracle_style = _curve_plot_style("oracle")
        ax.plot(
            reject_rate,
            y_rand,
            label="Random",
            markerfacecolor="white",
            markeredgewidth=1.7,
            **rand_style,
        )
        ax.plot(
            reject_rate,
            y_orac,
            label="Oracle",
            markerfacecolor="white",
            markeredgewidth=1.7,
            **oracle_style,
        )

    # Plot each reject score curve
    for name, rc in curves.items():
        if rc is None or len(rc) == 0:
            continue

        cov = rc["coverage"].to_numpy(dtype=float)
        reject_rate = (
            rc["reject_rate"].to_numpy(dtype=float)
            if "reject_rate" in rc.columns
            else 1.0 - cov
        )
        y = rc["risk_model"].to_numpy(dtype=float)

        if normalize:
            try:
                idx_hi = np.nanargmax(cov)
                base = float(y[idx_hi])
                if np.isfinite(base) and base > 0:
                    y = y / base
            except Exception:
                pass

        disp = _pretty(name)
        curve_style = _curve_plot_style(name)
        ax.plot(
            reject_rate,
            y,
            label=disp,
            markerfacecolor="white",
            markeredgewidth=1.7,
            **curve_style,
        )

    ax.set_xlabel("Rejected fraction")
    ax.set_ylabel(
        "Normalized selective risk (risk / risk@coverage=1)"
        if normalize
        else "Accepted-window sMAPE"
    )
    ax.set_title(title_main or title)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.55)
    if show_legend:
        ax.legend(loc="best", frameon=True)
    if title_note:
        ax.text(
            0.01,
            0.99,
            title_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    plt.tight_layout()

    if outpath is not None:
        _save_paper_figure(fig, outpath)
    plt.close(fig)


def _generate_cutoffs_from_start(
    df_full: pd.DataFrame,
    *,
    l_fcst: int,
    l_meta: int,
    horizon: int,
    step: int,
    diff_warmup: int,
) -> pd.DataFrame:
    df_full = df_full[[ID_COL, DS_COL, Y_COL]].copy().sort_values([ID_COL, DS_COL])
    rows = []
    for uid, g in df_full.groupby(ID_COL, observed=True):
        ds_vals = g[DS_COL].to_numpy()
        n = len(ds_vals)

        first_idx = max(int(l_fcst) + int(diff_warmup), int(l_meta)) - 1
        last_idx = int(n) - int(horizon) - 1
        if last_idx < first_idx:
            continue

        for idx in range(first_idx, last_idx + 1, int(step)):
            rows.append({ID_COL: uid, "cutoff": ds_vals[idx]})

    return pd.DataFrame(rows)


def _mad_np(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _add_past_residual_variance_score(
    pointwise: pd.DataFrame,
    *,
    horizon: int,
    k_past_cutoffs: int,
    use_t_width: bool = False,
    alpha: float = 0.1,
) -> pd.DataFrame:
    pw = pointwise[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]].copy()
    pw = pw.dropna(subset=[Y_COL, "y_hat"]).sort_values([ID_COL, "cutoff", DS_COL])

    pw["h_step"] = pw.groupby([ID_COL, "cutoff"], sort=False).cumcount() + 1
    pw = pw[pw["h_step"] <= int(horizon)]
    pw["err"] = pw[Y_COL].astype(float) - pw["y_hat"].astype(float)

    # keep arrays per (uid, cutoff): residuals pooled across horizon
    grp = (
        pw.groupby([ID_COL, "cutoff"], sort=False)["err"]
        .apply(lambda s: s.to_numpy(dtype=float))
        .reset_index()
        .rename(columns={"err": "_err_arr"})
        .sort_values([ID_COL, "cutoff"])
    )

    out_rows = []
    for uid, g in grp.groupby(ID_COL, sort=False):
        cutoffs = g["cutoff"].to_list()
        arrs = g["_err_arr"].to_list()

        for i in range(len(cutoffs)):
            if i == 0:
                continue
            j0 = max(0, i - int(k_past_cutoffs))
            hist_arrs = arrs[j0:i]
            if not hist_arrs:
                continue
            hist = np.concatenate(hist_arrs, axis=0)
            hist = hist[np.isfinite(hist)]
            if hist.size == 0:
                continue

            var_hat = float(np.mean(hist**2))

            if not use_t_width:
                score = var_hat
            else:
                import scipy.stats as st

                N = int(hist.size)
                s = float(np.sqrt(max(var_hat, 0.0)))
                tcrit = float(st.t.ppf(1.0 - float(alpha) / 2.0, df=max(N - 1, 1)))
                W = 2.0 * tcrit * (s / np.sqrt(max(N, 1)))
                score = W

            out_rows.append({ID_COL: uid, "cutoff": cutoffs[i], "score_err_var": score})

    if not out_rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff", "score_err_var"])
    return pd.DataFrame(out_rows)


def _rolling_ds2_conformal_pi_width_scores_per_uid(
    pointwise: pd.DataFrame,
    df_full: pd.DataFrame,
    cutoffs: Sequence[pd.Timestamp],
    *,
    horizon: int,
    level: int,
    m: int,
    stat: str,
    pi_agg: str = "mean",
    eps: float = 1e-8,
    calib_windows: int | None = 50,
) -> pd.DataFrame:
    q = float(level) / 100.0
    horizon = int(horizon)
    m = int(m)
    stat = str(stat)
    pi_agg = str(pi_agg)

    cutoffs_sorted = list(sorted(pd.to_datetime(list(cutoffs))))

    df_full = df_full[[ID_COL, DS_COL, Y_COL]].sort_values([ID_COL, DS_COL]).copy()

    # scale(uid, cutoff)
    scale_rows = []
    for cutoff in cutoffs_sorted:
        prefix = df_full[df_full[DS_COL] <= cutoff]
        tail = prefix.groupby(ID_COL, sort=False).tail(m)
        if tail.empty:
            continue
        if stat == "mad":
            s = tail.groupby(ID_COL, sort=False)[Y_COL].apply(
                lambda x: _mad_np(x.to_numpy(dtype=float))
            )
        elif stat == "std":
            s = tail.groupby(ID_COL, sort=False)[Y_COL].apply(
                lambda x: float(np.nanstd(x.to_numpy(dtype=float), ddof=0))
            )
        else:
            raise ValueError(f"Unsupported stat={stat}")
        s = s.fillna(1.0).astype(float)
        scale_rows.append(
            pd.DataFrame(
                {ID_COL: s.index.to_numpy(), "cutoff": cutoff, "scale": s.to_numpy()}
            )
        )
    if not scale_rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff", "uq_score"])
    scale_df = pd.concat(scale_rows, ignore_index=True)

    # nc(uid, cutoff, h_step)
    pw = pointwise[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]].copy()
    pw = pw.dropna(subset=[Y_COL, "y_hat"])
    pw = pw.sort_values([ID_COL, "cutoff", DS_COL])
    pw["h_step"] = pw.groupby([ID_COL, "cutoff"], sort=False).cumcount() + 1
    pw = pw[pw["h_step"] <= horizon]
    pw["abs_err"] = (pw[Y_COL].astype(float) - pw["y_hat"].astype(float)).abs()

    pw = pw.merge(scale_df, on=[ID_COL, "cutoff"], how="left")
    pw["scale"] = pw["scale"].fillna(1.0).astype(float)
    pw["nc"] = pw["abs_err"].astype(float) / (pw["scale"].astype(float) + eps)

    # per-uid cutoff order
    uid_cutoffs = (
        pw[[ID_COL, "cutoff"]]
        .drop_duplicates()
        .sort_values([ID_COL, "cutoff"])
        .groupby(ID_COL, sort=False)["cutoff"]
        .apply(list)
        .to_dict()
    )

    pw_by_uid = {uid: g for uid, g in pw.groupby(ID_COL, sort=False)}

    out = []
    for uid, uid_cuts in uid_cutoffs.items():
        g = pw_by_uid[uid]
        # ensure unique sorted cutoffs for this uid
        uid_cuts = list(sorted(pd.to_datetime(uid_cuts)))
        for i, cutoff in enumerate(uid_cuts):
            # calibration cutoffs for this uid
            if i == 0:
                continue
            if calib_windows is None:
                past_cuts = uid_cuts[:i]
            else:
                past_cuts = uid_cuts[max(0, i - int(calib_windows)) : i]

            cal = g[g["cutoff"].isin(past_cuts)]
            if cal.empty:
                continue

            # current scale
            s_now = scale_df[
                (scale_df[ID_COL] == uid) & (scale_df["cutoff"] == cutoff)
            ]["scale"]
            if s_now.empty:
                continue
            s_now = float(s_now.iloc[0])

            # quantiles per step (fallback to uid-global)
            v_all = cal["nc"].to_numpy(dtype=float)
            v_all = v_all[np.isfinite(v_all)]
            q_global = float(np.nanquantile(v_all, q)) if v_all.size else 1.0

            widths = []
            for h in range(1, horizon + 1):
                v = cal.loc[cal["h_step"] == h, "nc"].to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                qh = float(np.nanquantile(v, q)) if v.size else q_global
                widths.append(2.0 * qh * s_now)

            if pi_agg == "mean":
                score = float(np.nanmean(np.array(widths, dtype=float)))
            elif pi_agg == "max":
                score = float(np.nanmax(np.array(widths, dtype=float)))
            else:
                raise ValueError(f"Unsupported pi_agg={pi_agg}")

            out.append({ID_COL: uid, "cutoff": cutoff, "uq_score": score})

    if not out:
        return pd.DataFrame(columns=[ID_COL, "cutoff", "uq_score"])
    return pd.DataFrame(out)


def _add_residual_scale_score(
    win_df: pd.DataFrame,
    *,
    m: int,
    stat: str = "mad",
) -> pd.DataFrame:
    win_df = win_df.copy().sort_values([ID_COL, "cutoff"])
    m = int(m)

    def _mad(x: np.ndarray) -> float:
        med = np.median(x)
        return float(np.median(np.abs(x - med)))

    outs = []
    for uid, g in win_df.groupby(ID_COL, observed=True):
        g = g.sort_values("cutoff").copy()
        e = g["window_smape"].to_numpy()
        s = np.full(len(g), np.nan, dtype=float)
        for i in range(len(g)):
            if i == 0:
                continue
            start = max(0, i - m)
            hist = e[start:i]
            if len(hist) == 0:
                continue
            if stat == "std":
                s[i] = float(np.std(hist, ddof=0))
            else:
                s[i] = _mad(hist)
        g["score_resid_scale"] = s
        outs.append(g[[ID_COL, "cutoff", "score_resid_scale"]])
    out = (
        pd.concat(outs, ignore_index=True)
        if outs
        else win_df[[ID_COL, "cutoff"]].assign(score_resid_scale=np.nan)
    )
    return win_df.merge(out, on=[ID_COL, "cutoff"], how="left")


def _add_reject_baseline_scores(
    win_df: pd.DataFrame,
    *,
    pointwise: pd.DataFrame,
    df_full: pd.DataFrame,
    horizon: int,
    resid_m: int,
) -> tuple[pd.DataFrame, list[str]]:
    scored = win_df.copy()
    baseline_score_cols: list[str] = []

    try:
        scored = _add_residual_scale_score(
            scored,
            m=int(resid_m),
            stat=str(DEFAULT_UQ_RESID_STAT),
        )
        baseline_score_cols.append("score_resid_scale")
    except Exception as e:
        _dbg("UQ_RESID", f"Residual-scale baseline failed: {type(e).__name__}: {e}")

    try:
        err_var_scores = _add_past_residual_variance_score(
            pointwise=pointwise,
            horizon=int(horizon),
            k_past_cutoffs=int(resid_m),
            use_t_width=True,
            alpha=1.0 - (float(DEFAULT_UQ_PI_LEVEL) / 100.0),
        )
        if len(err_var_scores) > 0:
            scored = scored.drop(columns=["score_err_var"], errors="ignore").merge(
                err_var_scores, on=[ID_COL, "cutoff"], how="left"
            )
            baseline_score_cols.append("score_err_var")
    except Exception as e:
        _dbg("UQ_ERR_VAR", f"Err-variance baseline failed: {type(e).__name__}: {e}")

    try:
        pi_scores = _rolling_ds2_conformal_pi_width_scores_per_uid(
            pointwise=pointwise,
            df_full=df_full[df_full[ID_COL].isin(scored[ID_COL].unique())],
            cutoffs=sorted(scored["cutoff"].dropna().unique()),
            horizon=int(horizon),
            level=int(DEFAULT_UQ_PI_LEVEL),
            m=int(resid_m),
            stat=str(DEFAULT_UQ_RESID_STAT),
            pi_agg=str(DEFAULT_UQ_PI_AGG),
            calib_windows=50,
        ).rename(columns={"uq_score": "score_pi_width"})
        if len(pi_scores) > 0:
            scored = scored.drop(columns=["score_pi_width"], errors="ignore").merge(
                pi_scores, on=[ID_COL, "cutoff"], how="left"
            )
            baseline_score_cols.append("score_pi_width")
    except Exception as e:
        _dbg("UQ_PI", f"PI width scoring failed: {type(e).__name__}: {e}")

    for c in baseline_score_cols:
        if c in scored.columns:
            scored[c] = scored[c].fillna(0)

    return scored, baseline_score_cols


def _rolling_transfer_pointwise_preds(
    fcst: ForecasterAdapter,
    df_full: pd.DataFrame,
    cutoffs: pd.DataFrame,
    horizon: int,
    max_series: int | None,
    max_windows_per_series: int | None,
) -> pd.DataFrame:
    df_full = df_full[[ID_COL, DS_COL, Y_COL]].copy().sort_values([ID_COL, DS_COL])
    cutoffs = cutoffs[[ID_COL, "cutoff"]].copy().sort_values([ID_COL, "cutoff"])

    out_rows = []

    uids = list(df_full[ID_COL].unique())
    if max_series is not None:
        uids = uids[: int(max_series)]

    for i, uid in enumerate(uids, start=1):
        g = df_full[df_full[ID_COL] == uid].sort_values(DS_COL)
        cu = cutoffs[cutoffs[ID_COL] == uid].sort_values("cutoff")
        if max_windows_per_series is not None:
            cu = cu.head(int(max_windows_per_series))
        if cu.empty:
            continue

        _dbg("XFER", f"uid={uid} ({i}/{len(uids)}) windows={len(cu)}")
        for _, r in cu.iterrows():
            cutoff = r["cutoff"]
            prefix = g[g[DS_COL] <= cutoff][[ID_COL, DS_COL, Y_COL]].copy()
            if len(prefix) < 2:
                continue

            pred = fcst.predict(h=int(horizon), new_df=prefix).copy()
            pred["cutoff"] = cutoff

            actual = g[[DS_COL, Y_COL]].copy()
            m = pred.merge(actual, on=DS_COL, how="left")
            m[ID_COL] = uid
            m = m[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]]
            out_rows.append(m)

    if not out_rows:
        return pd.DataFrame(columns=[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"])

    return pd.concat(out_rows, axis=0, ignore_index=True)


def _make_meta_regressor(meta_model: str, random_state: int, n_jobs: int):
    if meta_model == "lgbm":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )
    if meta_model == "catboost":
        return CatBoostRegressor(
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=int(random_state),
            thread_count=int(n_jobs) if int(n_jobs) > 0 else -1,
            iterations=1000,
            learning_rate=0.05,
            depth=6,
        )
    raise ValueError(f"Unknown meta_model: {meta_model}")


def _make_meta_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="constant", fill_value=0.0)


def _tune_meta_regressor_random_search(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta_model: str,
    random_state: int,
    n_iter: int,
    cv_folds: int,
    n_jobs: int,
    sample_weight: np.ndarray | None,
):
    if meta_model == "lgbm":
        base = _make_meta_regressor("lgbm", random_state=random_state, n_jobs=n_jobs)
        param_distributions = {
            "reg__n_estimators": randint(300, 2000),
            "reg__learning_rate": loguniform(1e-3, 2e-1),
            "reg__num_leaves": randint(8, 256),
            "reg__min_child_samples": randint(5, 200),
            "reg__subsample": uniform(0.6, 0.4),
            "reg__colsample_bytree": uniform(0.6, 0.4),
            "reg__reg_alpha": loguniform(1e-8, 10.0),
            "reg__reg_lambda": loguniform(1e-8, 10.0),
        }
    elif meta_model == "catboost":
        base = _make_meta_regressor(
            "catboost", random_state=random_state, n_jobs=n_jobs
        )
        param_distributions = {
            "reg__depth": randint(3, 11),
            "reg__learning_rate": uniform(0.01, 0.19),
            "reg__l2_leaf_reg": loguniform(1e-2, 10.0),
            "reg__bagging_temperature": uniform(0.0, 1.0),
            "reg__border_count": randint(32, 256),
            "reg__iterations": randint(300, 2000),
        }
    else:
        raise ValueError(f"Unknown meta_model: {meta_model}")

    pipe = Pipeline(steps=[("imputer", _make_meta_imputer()), ("reg", base)])

    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=int(cv_folds))

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=int(n_iter),
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=int(n_jobs),
        refit=True,
        verbose=0,
        random_state=int(random_state),
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["reg__sample_weight"] = np.asarray(sample_weight, dtype=float)

    search.fit(X, y, groups=groups, **fit_kwargs)
    _dbg(
        "META_TUNE",
        f"best neg_mse={search.best_score_:.6f} params={search.best_params_}",
    )
    return search.best_estimator_


def _train_meta_regressor(
    meta_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    meta_model: str,
    random_state: int,
    n_jobs: int,
    tune: bool,
    tune_iter: int,
    tune_cv_folds: int,
    sample_weight: np.ndarray | None,
):
    df = meta_train.copy().replace([np.inf, -np.inf], np.nan)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    groups = df[ID_COL].to_numpy()

    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]

    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        if w.shape[0] != df.shape[0]:
            raise ValueError(
                f"sample_weight length {w.shape[0]} does not match meta_train rows {df.shape[0]}"
            )
        w = w[ok]
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    if tune:
        return _tune_meta_regressor_random_search(
            X=X,
            y=y,
            groups=groups,
            meta_model=meta_model,
            random_state=random_state,
            n_iter=tune_iter,
            cv_folds=tune_cv_folds,
            n_jobs=n_jobs,
            sample_weight=w,
        )

    base = _make_meta_regressor(meta_model, random_state=random_state, n_jobs=n_jobs)
    pipe = Pipeline(steps=[("imputer", _make_meta_imputer()), ("reg", base)])

    fit_kwargs = {}
    if w is not None:
        fit_kwargs["reg__sample_weight"] = w

    pipe.fit(X, y, **fit_kwargs)
    return pipe


def _predict_u_hat(
    meta_reg, meta_df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    X = meta_df[feature_cols].to_numpy(dtype=float)
    u = np.asarray(meta_reg.predict(X), dtype=float)
    u = np.where(np.isfinite(u), u, np.nan)
    return np.clip(u, 0.0, 1.0)


def _build_ds1_meta_bases(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_df: pd.DataFrame,
    fcst: ForecasterAdapter,
    *,
    horizon: int,
    season_length: int,
    l_meta: int,
    tsfel_nan_col_thresh: float,
    uq_resid_m: int,
    tsfel_fs: int = DEFAULT_TSFEL_FS,
    standardize: bool = False,
):
    cv_df = cv_df.copy()
    cv_df = _attach_pointwise_smape(cv_df)
    win_cv_all = _windowize_pointwise(
        cv_df, horizon=int(horizon), require_complete=True
    )

    mid = _mid_cutoff_per_series(train)
    win_cv_with_mid = win_cv_all.merge(mid, on=ID_COL, how="left")
    before = len(win_cv_with_mid)
    win_cv = win_cv_with_mid[
        win_cv_with_mid["cutoff"] >= win_cv_with_mid["mid_cutoff"]
    ].copy()
    _dbg("WARMUP", f"DS1 kept windows cutoff>=midpoint: {len(win_cv)}/{before}")

    cutoffs_cv = win_cv_all[[ID_COL, "cutoff"]].drop_duplicates()
    last_train = (
        train.groupby(ID_COL, observed=True)[DS_COL]
        .max()
        .reset_index()
        .rename(columns={DS_COL: "cutoff"})
    )
    cutoffs_all = pd.concat(
        [cutoffs_cv, last_train], axis=0, ignore_index=True
    ).drop_duplicates()
    cutoffs_all = _filter_cutoffs_by_history(train, cutoffs_all, l_meta=int(l_meta))
    _dbg("TSFEL", f"DS1 cutoffs after history filter: {len(cutoffs_all)}")

    ctx = _tsfel_context_features(
        df_long=train,
        l_meta=int(l_meta),
        fs=int(tsfel_fs),
        cutoffs_df=cutoffs_all,
        nan_col_thresh=float(tsfel_nan_col_thresh),
        standardize=standardize,
    )
    feature_cols = [c for c in ctx.columns if c.startswith("tsfel_")]
    _dbg("TSFEL", f"DS1 extracted {len(feature_cols)} TSFEL features")

    meta_train_all_base = win_cv_all.merge(
        ctx, on=[ID_COL, "cutoff"], how="left"
    ).replace([np.inf, -np.inf], np.nan)

    preds_test = fcst.predict(h=int(horizon), new_df=train).copy()
    tmp = test.merge(preds_test, on=[ID_COL, DS_COL], how="left")
    matched_rows = int(tmp["y_hat"].notna().sum())
    _dbg(
        "MERGE",
        f"DS1 test matched_rows={matched_rows} matched_rate={(tmp['y_hat'].notna().mean() if len(tmp) else 0):.3f}",
    )
    if matched_rows == 0:
        raise RuntimeError(
            "No DS1 test points matched predictions. Check freq/ds alignment."
        )

    joined = (
        test.merge(preds_test, on=[ID_COL, DS_COL], how="left")
        .dropna(subset=["y_hat"])
        .copy()
    )
    sn_test_pointwise = _seasonal_naive_holdout_pointwise(
        train,
        test,
        horizon=int(horizon),
        season_length=int(season_length),
    )
    ds1_holdout_seasonal_naive_summary = (
        None
        if len(sn_test_pointwise) == 0
        else _pointwise_summary_from_predictions(
            sn_test_pointwise,
            horizon=int(horizon),
            yhat_col="y_hat",
        )
    )
    joined = joined.merge(last_train, on=ID_COL, how="left")
    ds1_holdout_forecast_summary = _pointwise_summary_from_predictions(
        joined,
        horizon=int(horizon),
        yhat_col="y_hat",
    )
    joined = _attach_pointwise_smape(joined)
    win_test = (
        joined.groupby([ID_COL, "cutoff"], observed=True)
        .agg(
            window_smape=("smape", "mean"),
            n_steps=("smape", "size"),
            target_start=(DS_COL, "min"),
            target_end=(DS_COL, "max"),
        )
        .reset_index()
    )
    win_test = win_test[win_test["n_steps"] == int(horizon)].copy()

    df1_full = (
        pd.concat(
            [
                train[[ID_COL, DS_COL, Y_COL]].copy(),
                test[[ID_COL, DS_COL, Y_COL]].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        .sort_values([ID_COL, DS_COL])
        .reset_index(drop=True)
    )
    pointwise_all = (
        pd.concat(
            [
                cv_df[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]].copy(),
                joined[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]].copy(),
            ],
            axis=0,
            ignore_index=True,
        )
        .sort_values([ID_COL, "cutoff", DS_COL])
        .reset_index(drop=True)
    )
    win_all = (
        pd.concat([win_cv_all, win_test], axis=0, ignore_index=True)
        .sort_values([ID_COL, "cutoff"])
        .drop_duplicates(subset=[ID_COL, "cutoff"], keep="last")
        .reset_index(drop=True)
    )
    win_all_scored, ds1_baseline_score_cols = _add_reject_baseline_scores(
        win_all,
        pointwise=pointwise_all,
        df_full=df1_full,
        horizon=int(horizon),
        resid_m=int(uq_resid_m),
    )
    ds1_baseline_tbl = win_all_scored[
        [ID_COL, "cutoff", *ds1_baseline_score_cols]
    ].copy()

    meta_test_base = win_test.merge(ctx, on=[ID_COL, "cutoff"], how="left").replace(
        [np.inf, -np.inf], np.nan
    )
    if ds1_baseline_score_cols:
        meta_train_all_base = meta_train_all_base.merge(
            ds1_baseline_tbl, on=[ID_COL, "cutoff"], how="left"
        )
        meta_test_base = meta_test_base.merge(
            ds1_baseline_tbl, on=[ID_COL, "cutoff"], how="left"
        )

    meta_train_base = meta_train_all_base.merge(mid, on=ID_COL, how="left")
    meta_train_base = meta_train_base[
        meta_train_base["cutoff"] >= meta_train_base["mid_cutoff"]
    ].copy()
    meta_train_base = meta_train_base.drop(columns=["mid_cutoff"])

    return (
        meta_train_all_base,
        meta_train_base,
        meta_test_base,
        ctx,
        feature_cols,
        ds1_baseline_score_cols,
        ds1_holdout_forecast_summary,
        ds1_holdout_seasonal_naive_summary,
    )


def _build_ds2_meta_base(
    fcst_ds1: ForecasterAdapter,
    df2_full: pd.DataFrame,
    *,
    horizon: int,
    season_length: int,
    l_fcst: int,
    l_meta: int,
    diff_warmup: int,
    step: int,
    max_series: int | None,
    max_windows_per_series: int | None,
    tsfel_fs: int = DEFAULT_TSFEL_FS,
    standardize: bool = False,
):
    cutoffs2 = _generate_cutoffs_from_start(
        df2_full,
        l_fcst=int(l_fcst),
        l_meta=int(l_meta),
        horizon=int(horizon),
        step=int(step),
        diff_warmup=int(diff_warmup),
    )
    _dbg(
        "CUTOFFS",
        f"DS2 generated cutoffs rows={len(cutoffs2)} series={cutoffs2[ID_COL].nunique()} step={step}",
    )
    if len(cutoffs2) == 0:
        raise RuntimeError("No valid DS2 cutoffs (likely too-short series).")

    cutoffs2 = _filter_cutoffs_by_history(df2_full, cutoffs2, l_meta=int(l_meta))
    _dbg(
        "CUTOFFS",
        f"DS2 cutoffs after history filter: rows={len(cutoffs2)} series={cutoffs2[ID_COL].nunique()}",
    )
    if len(cutoffs2) == 0:
        raise RuntimeError("No DS2 cutoffs remain after history filter.")

    _dbg(
        "TSFEL",
        f"DS2 extracting TSFEL context features (fs={tsfel_fs}, L_meta={l_meta})",
    )
    ctx2 = _tsfel_context_features(
        df_long=df2_full,
        l_meta=int(l_meta),
        fs=int(tsfel_fs),
        cutoffs_df=cutoffs2,
        nan_col_thresh=None,
        standardize=standardize,
    )
    _dbg("TSFEL", f"DS2 ctx2 rows={len(ctx2)}")

    t0 = time.perf_counter()
    pointwise2 = _rolling_transfer_pointwise_preds(
        fcst=fcst_ds1,
        df_full=df2_full,
        cutoffs=cutoffs2,
        horizon=int(horizon),
        max_series=max_series,
        max_windows_per_series=max_windows_per_series,
    )
    _dbg(
        "XFER",
        f"DS2 transfer pointwise rows={len(pointwise2)} elapsed={time.perf_counter()-t0:.1f}s",
    )
    if len(pointwise2) == 0:
        raise RuntimeError("No DS2 transfer predictions produced.")

    pointwise2 = pointwise2.dropna(subset=[Y_COL, "y_hat"]).copy()
    pointwise2 = _attach_pointwise_smape(pointwise2)

    pointwise2_sn = _rolling_seasonal_naive_pointwise_preds(
        df_full=df2_full,
        cutoffs=cutoffs2,
        horizon=int(horizon),
        season_length=int(season_length),
        max_series=max_series,
        max_windows_per_series=max_windows_per_series,
    )
    if len(pointwise2_sn) > 0:
        pointwise2_sn = pointwise2_sn.dropna(subset=[Y_COL, "y_hat"]).copy()
        pointwise2_sn = _attach_pointwise_smape(pointwise2_sn)
    else:
        pointwise2_sn = None

    win2 = (
        pointwise2.groupby([ID_COL, "cutoff"], observed=True)
        .agg(
            window_smape=("smape", "mean"),
            n_steps=("smape", "size"),
            target_start=(DS_COL, "min"),
            target_end=(DS_COL, "max"),
        )
        .reset_index()
    )

    win2 = win2[win2["n_steps"] == int(horizon)].copy()
    _dbg("DS2_WIN", f"DS2 windows with complete horizons={len(win2)}")
    if len(win2) == 0:
        raise RuntimeError("No complete DS2 windows; cannot evaluate.")

    meta2_base = win2.merge(ctx2, on=[ID_COL, "cutoff"], how="left").replace(
        [np.inf, -np.inf], np.nan
    )
    return meta2_base, ctx2, pointwise2, pointwise2_sn


@dataclass
class RunArtifacts:
    outdir: Path
    plots_dir: Path
    models_dir: Path | None = None


def _prepare_outdirs(
    outdir: str | Path, *, save_model_artifacts: bool = False
) -> RunArtifacts:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if save_model_artifacts:
        models_dir = outdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(outdir=outdir, plots_dir=plots_dir, models_dir=models_dir)


def _safe_group_tag(value: str) -> str:
    text = str(value).strip().replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    text = text.strip("_")
    return text or "dataset"


def _dataset_tag(dataset: str, group: str, csv_path: str | None = None) -> str:
    source = csv_path if csv_path else dataset
    source_str = str(source)
    if source_str.lower().endswith(".csv"):
        return _safe_group_tag(Path(source_str).stem)
    return f"{_safe_group_tag(dataset)}_{_safe_group_tag(group)}"


def _dataset_label(dataset: str, group: str, csv_path: str | None = None) -> str:
    source = csv_path if csv_path else dataset
    source_str = str(source)
    if source_str.lower().endswith(".csv"):
        return Path(source_str).stem
    if group:
        return f"{dataset} {group}"
    return str(dataset)


def _make_prepare_dataset(args, prefix: str) -> PrepareDataset:
    return PrepareDataset(
        dataset=getattr(args, f"{prefix}_data"),
        group=getattr(args, f"{prefix}_group"),
        csv_path=getattr(args, f"{prefix}_csv_path"),
        id_col=getattr(args, f"{prefix}_id_col"),
        ds_col=getattr(args, f"{prefix}_ds_col"),
        value_col=getattr(args, f"{prefix}_value_col"),
        freq=getattr(args, f"{prefix}_freq"),
        seasonality=getattr(args, f"{prefix}_seasonality"),
        long_horizon_max_points=getattr(args, f"{prefix}_long_horizon_max_points"),
        long_horizon_max_points_factor=getattr(
            args, f"{prefix}_long_horizon_max_points_factor"
        ),
    )


def _empirical_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise RuntimeError("Reference error distribution is empty.")

    ref_sorted = np.sort(ref)
    n_ref = float(ref_sorted.size)

    v = np.asarray(values, dtype=float)
    out = np.full(v.shape[0], np.nan, dtype=float)
    ok = np.isfinite(v)
    ranks = np.searchsorted(ref_sorted, v[ok], side="right").astype(float)
    out[ok] = ranks / (n_ref + 1.0)
    return out


def _available_reject_scores(meta_df: pd.DataFrame) -> list[tuple[str, str]]:
    scores = [("u_hat", "u_hat")]
    for name, col in [
        ("pi_width", "score_pi_width"),
        ("resid_scale", "score_resid_scale"),
        ("err_var", "score_err_var"),
    ]:
        if col in meta_df.columns:
            scores.append((name, col))
    return scores


@dataclass
class ForecastStageArtifacts:
    safe_ds1: str
    train1: pd.DataFrame
    test1: pd.DataFrame
    fcst_ds1: ForecasterAdapter
    cv_df: pd.DataFrame
    input_size: int
    fit_time: float
    cv_windows: int | None
    freq1: str | int
    seasonality1: int
    ds1_forecast_cv_summary: dict
    ds1_seasonal_naive_cv_summary: dict | None
    nf_cache_path: str | None
    nf_alias_path: str | None


@dataclass
class MetaStageArtifacts:
    safe_ds1: str
    safe_ds2: str
    feature_cols: list[str]
    baseline_score_cols: list[str]
    meta_train_all_base: pd.DataFrame
    meta_train_base: pd.DataFrame
    meta_test_base: pd.DataFrame
    meta2_base: pd.DataFrame
    meta2_eval_base: pd.DataFrame
    ds2_baseline_mean: float
    ds1_holdout_forecast_summary: dict
    ds1_holdout_seasonal_naive_summary: dict | None
    ds2_transfer_pointwise: pd.DataFrame
    ds2_seasonal_naive_pointwise: pd.DataFrame | None


def _run_forecast_stage(args, art: RunArtifacts) -> ForecastStageArtifacts:
    h = int(args.horizon)
    lags = int(args.meta_lags)
    save_model_artifacts = bool(getattr(args, "save_model_artifacts", False))
    safe_ds1 = _dataset_tag(args.ds1_data, args.ds1_group, args.ds1_csv_path)
    ds1_label = _dataset_label(args.ds1_data, args.ds1_group, args.ds1_csv_path)

    ds1 = _make_prepare_dataset(args, "ds1")
    _dbg("DATA", f"Loading DS1: {ds1_label}")
    ds1.load_dataset(horizon=h, lags=lags, drop_short_series_factor=1)
    ds1.train_test_split(horizon=h)

    train1 = _normalize_ds_dtype(ds1.train[[ID_COL, DS_COL, Y_COL]].copy())
    test1 = _normalize_ds_dtype(ds1.test[[ID_COL, DS_COL, Y_COL]].copy())

    freq1, seasonality1 = _infer_freq_and_seasonality(args.ds1_data, ds1)
    _dbg("DS1_META", f"freq={freq1} seasonality={seasonality1}")

    input_mult = int(args.input_mult)
    start_padding_enabled = bool(args.start_padding_enabled)
    input_size = input_mult * h
    cache_cfg = None
    nf_cache_path = None
    cv_path = None
    meta_path = None
    if save_model_artifacts:
        if art.models_dir is None:
            raise RuntimeError("models_dir is required when saving model artifacts.")
        cache_cfg = {
            "ds1_data": args.ds1_data,
            "ds1_group": str(args.ds1_group),
            "ds1_csv_path": args.ds1_csv_path,
            "ds1_id_col": args.ds1_id_col,
            "ds1_ds_col": args.ds1_ds_col,
            "ds1_value_col": args.ds1_value_col,
            "ds1_freq": args.ds1_freq,
            "ds1_seasonality": args.ds1_seasonality,
            "forecast_model": str(args.forecast_model),
            "horizon": h,
            "input_mult": input_mult,
            "start_padding_enabled": start_padding_enabled,
            "input_size": input_size,
            "batch_size": DEFAULT_NF_BATCH_SIZE,
            "cv_refit": DEFAULT_CV_REFIT,
            "cv_val_size": DEFAULT_CV_VAL_SIZE,
            "cv_test_size": DEFAULT_CV_TEST_SIZE,
            "seed": int(args.seed),
            "train_rows": int(len(train1)),
            "train_series": int(train1[ID_COL].nunique()),
            "train_ds_min": str(train1[DS_COL].min()),
            "train_ds_max": str(train1[DS_COL].max()),
        }
        try:
            import neuralforecast

            cache_cfg["neuralforecast_version"] = getattr(
                neuralforecast, "__version__", "unknown"
            )
        except Exception:
            cache_cfg["neuralforecast_version"] = "unknown"

        cache_id = _hash_config(cache_cfg)
        nf_cache_path = art.models_dir / f"nf_{safe_ds1}_{cache_id}"
        cv_path = art.models_dir / f"cv_{safe_ds1}_{cache_id}.joblib"
        meta_path = art.models_dir / f"nf_{safe_ds1}_{cache_id}.json"

    nf = None
    cv_df = None
    fit_time = 0.0
    step_cv = h
    test_size_cv = None
    n_windows = None

    if (
        save_model_artifacts
        and nf_cache_path is not None
        and cv_path is not None
        and meta_path is not None
        and nf_cache_path.exists()
        and cv_path.exists()
        and meta_path.exists()
    ):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("cache_cfg") == cache_cfg:
                nf = _try_load_nf(nf_cache_path)
                if nf is not None:
                    cv_df = joblib.load(cv_path)
                    step_cv = int(meta["step_cv"])
                    test_size_cv = int(meta["test_size_cv"])
                    input_size = int(meta["input_size"])
        except Exception:
            nf, cv_df = None, None

    if nf is None or cv_df is None:
        nf, fcst_ds1, cv_df, fit_time, input_size, n_windows, step_cv, test_size_cv = (
            _fit_and_cv(
                train=train1,
                horizon=h,
                forecast_model=str(args.forecast_model),
                seed=int(args.seed),
                input_mult=input_mult,
                start_padding_enabled=start_padding_enabled,
                batch_size=DEFAULT_NF_BATCH_SIZE,
                refit=DEFAULT_CV_REFIT,
                val_size=DEFAULT_CV_VAL_SIZE,
                test_size=DEFAULT_CV_TEST_SIZE,
                loss=None,
            )
        )

        if (
            save_model_artifacts
            and nf_cache_path is not None
            and cv_path is not None
            and meta_path is not None
        ):
            nf.save(path=str(nf_cache_path), overwrite=True, save_dataset=False)
            joblib.dump(cv_df, cv_path)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "cache_cfg": cache_cfg,
                        "input_size": int(input_size),
                        "step_cv": int(step_cv),
                        "test_size_cv": int(test_size_cv),
                    },
                    f,
                    indent=2,
                )
    else:
        fcst_ds1 = ForecasterAdapter(nf, model_col=None)
        n_windows = int(test_size_cv // step_cv) if test_size_cv is not None else None

    s_sn = None
    if _HAS_STATSFORECAST:
        try:
            sf = StatsForecast(
                models=[SeasonalNaive(season_length=int(seasonality1))],
                freq=freq1,
                n_jobs=-1,
            )
            cv_sn = sf.cross_validation(
                df=train1,
                h=h,
                n_windows=int(max(1, n_windows or 1)),
                step_size=h,
            )
            if ID_COL not in cv_sn.columns:
                cv_sn = cv_sn.reset_index()
            if "SeasonalNaive" in cv_sn.columns:
                s_sn = _cv_summary_from_pointwise(
                    cv_df=cv_sn,
                    horizon=h,
                    yhat_col="SeasonalNaive",
                )
        except Exception as e:
            _dbg("DS1_BASE_CV", f"SeasonalNaive CV failed: {type(e).__name__}: {e}")

    s_base_nf = _cv_summary_from_pointwise(cv_df=cv_df, horizon=h, yhat_col="y_hat")
    _dbg(
        "DS1_FCST",
        f"fit_time={fit_time:.1f}s input_size={input_size} input_mult={input_mult} "
        f"start_padding_enabled={start_padding_enabled} CV_windows={n_windows} CV_test_size={test_size_cv}",
    )
    _dbg(
        "DS1_BASE_CV",
        f"mean_window_sMAPE={s_base_nf['mean_window_smape']:.4f} "
        f"mean_point_sMAPE={s_base_nf['mean_pointwise_smape']:.4f} "
        f"windows={s_base_nf['n_windows']} cutoffs={s_base_nf['n_cutoffs']}",
    )

    nf_cache_path_str = str(nf_cache_path) if nf_cache_path is not None else None
    nf_alias_path_str = None
    if save_model_artifacts:
        if art.models_dir is None:
            raise RuntimeError("models_dir is required when saving model artifacts.")
        nf_alias_path = art.models_dir / f"nf_{safe_ds1}"
        nf.save(path=str(nf_alias_path), overwrite=True, save_dataset=False)
        _dbg("SAVE", f"Saved DS1 forecaster to {nf_alias_path}")
        nf_alias_path_str = str(nf_alias_path)
    else:
        _dbg("SAVE", "Model artifacts disabled; skipping forecaster checkpoint export.")

    return ForecastStageArtifacts(
        safe_ds1=safe_ds1,
        train1=train1,
        test1=test1,
        fcst_ds1=fcst_ds1,
        cv_df=cv_df,
        input_size=int(input_size),
        fit_time=float(fit_time),
        cv_windows=n_windows,
        freq1=freq1,
        seasonality1=int(seasonality1),
        ds1_forecast_cv_summary=s_base_nf,
        ds1_seasonal_naive_cv_summary=s_sn,
        nf_cache_path=nf_cache_path_str,
        nf_alias_path=nf_alias_path_str,
    )


def _run_meta_data_stage(
    args, art: RunArtifacts, forecast_art: ForecastStageArtifacts
) -> MetaStageArtifacts:
    ds1_label = _dataset_label(args.ds1_data, args.ds1_group, args.ds1_csv_path)
    safe_ds2 = _dataset_tag(args.ds2_data, args.ds2_group, args.ds2_csv_path)
    ds2_label = _dataset_label(args.ds2_data, args.ds2_group, args.ds2_csv_path)
    h = int(args.horizon)
    lags = int(args.meta_lags)
    uq_resid_m = int(max(5, 2 * h))

    (
        meta_train_all_base,
        meta_train_base,
        meta_test_base,
        _,
        feature_cols,
        ds1_baseline_score_cols,
        ds1_holdout_forecast_summary,
        ds1_holdout_seasonal_naive_summary,
    ) = _build_ds1_meta_bases(
        train=forecast_art.train1,
        test=forecast_art.test1,
        cv_df=forecast_art.cv_df,
        fcst=forecast_art.fcst_ds1,
        horizon=h,
        season_length=int(forecast_art.seasonality1),
        l_meta=lags,
        tsfel_nan_col_thresh=float(DEFAULT_TSFEL_NAN_COL_THRESH),
        uq_resid_m=int(uq_resid_m),
        tsfel_fs=int(args.fs),
        standardize=bool(args.standardize_features),
    )
    if len(feature_cols) == 0:
        raise RuntimeError(
            "No TSFEL features extracted. Adjust --meta_lags or TSFEL settings."
        )
    _dbg(
        "DS1_META_BASE",
        f"meta_train_base rows={len(meta_train_base)} meta_test_base rows={len(meta_test_base)} features={len(feature_cols)}",
    )

    ds2 = _make_prepare_dataset(args, "ds2")
    _dbg("DATA", f"Loading DS2: {ds2_label}")
    ds2.load_dataset(horizon=h, lags=lags, drop_short_series_factor=1)
    ds2.train_test_split(horizon=h)
    freq2, seasonality2 = _infer_freq_and_seasonality(args.ds2_data, ds2)
    _dbg("DS2_META", f"freq={freq2} seasonality={seasonality2}")
    _validate_transfer_frequency_match(
        ds1_label=ds1_label,
        ds2_label=ds2_label,
        freq1=forecast_art.freq1,
        freq2=freq2,
    )

    df2_full = pd.concat(
        [
            ds2.train[[ID_COL, DS_COL, Y_COL]].copy(),
            ds2.test[[ID_COL, DS_COL, Y_COL]].copy(),
        ],
        axis=0,
        ignore_index=True,
    )
    df2_full = (
        _normalize_ds_dtype(df2_full)
        .sort_values([ID_COL, DS_COL])
        .reset_index(drop=True)
    )

    step_ds2 = h if DEFAULT_DS2_STEP is None else int(DEFAULT_DS2_STEP)

    meta2_base, _, pointwise2, pointwise2_sn = _build_ds2_meta_base(
        fcst_ds1=forecast_art.fcst_ds1,
        df2_full=df2_full,
        horizon=h,
        season_length=int(seasonality2),
        l_fcst=int(forecast_art.input_size),
        l_meta=int(args.meta_lags),
        diff_warmup=int(DEFAULT_DIFF_WARMUP),
        step=int(step_ds2),
        max_series=None,
        max_windows_per_series=None,
        tsfel_fs=int(args.fs),
        standardize=bool(args.standardize_features),
    )

    for c in feature_cols:
        if c not in meta2_base.columns:
            meta2_base[c] = np.nan

    feature_values = meta2_base[feature_cols].apply(pd.to_numeric, errors="coerce")
    finite_features = np.isfinite(feature_values.to_numpy(dtype=float))
    finite_feature_cols = int(np.any(finite_features, axis=0).sum())
    finite_feature_values = int(finite_features.sum())
    _dbg(
        "DS2_META_BASE",
        f"meta2_base rows={len(meta2_base)} "
        f"features_present={sum(c in meta2_base.columns for c in feature_cols)}/{len(feature_cols)} "
        f"finite_feature_cols={finite_feature_cols}/{len(feature_cols)} "
        f"finite_feature_values={finite_feature_values}",
    )

    meta2_base, ds2_baseline_score_cols = _add_reject_baseline_scores(
        meta2_base,
        pointwise=pointwise2,
        df_full=df2_full,
        horizon=h,
        resid_m=int(uq_resid_m),
    )
    baseline_score_cols = sorted(
        set(ds1_baseline_score_cols).union(ds2_baseline_score_cols)
    )

    _dbg("UQ_SCORES", f"baseline_score_cols={baseline_score_cols}")

    meta2_eval_base = meta2_base.copy().replace([np.inf, -np.inf], np.nan)
    meta2_eval_base = meta2_eval_base.dropna(subset=["window_smape"]).copy()
    ds2_baseline_mean = (
        float(meta2_eval_base["window_smape"].mean())
        if len(meta2_eval_base)
        else np.nan
    )

    return MetaStageArtifacts(
        safe_ds1=forecast_art.safe_ds1,
        safe_ds2=safe_ds2,
        feature_cols=feature_cols,
        baseline_score_cols=baseline_score_cols,
        meta_train_all_base=meta_train_all_base,
        meta_train_base=meta_train_base,
        meta_test_base=meta_test_base,
        meta2_base=meta2_base,
        meta2_eval_base=meta2_eval_base,
        ds2_baseline_mean=ds2_baseline_mean,
        ds1_holdout_forecast_summary=ds1_holdout_forecast_summary,
        ds1_holdout_seasonal_naive_summary=ds1_holdout_seasonal_naive_summary,
        ds2_transfer_pointwise=pointwise2,
        ds2_seasonal_naive_pointwise=pointwise2_sn,
    )


def _fit_reject_meta_regressor(
    meta_train_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    meta_model: str,
    seed: int,
    tune_meta: bool,
) -> tuple[Pipeline, pd.DataFrame]:
    meta_train = meta_train_df.copy().replace([np.inf, -np.inf], np.nan)
    meta_train = meta_train.dropna(subset=["window_smape"]).copy()
    if len(meta_train) == 0:
        raise RuntimeError(
            "Reject meta-train is empty after dropping invalid window_smape."
        )

    train_smape = meta_train["window_smape"].to_numpy(dtype=float)
    meta_train["u_target"] = _empirical_percentiles(train_smape, train_smape)

    meta_reg = _train_meta_regressor(
        meta_train=meta_train,
        feature_cols=feature_cols,
        target_col="u_target",
        meta_model=str(meta_model),
        random_state=int(seed),
        n_jobs=int(DEFAULT_META_N_JOBS),
        tune=bool(tune_meta),
        tune_iter=int(DEFAULT_TUNE_ITER),
        tune_cv_folds=int(DEFAULT_TUNE_CV_FOLDS),
        sample_weight=None,
    )
    return meta_reg, meta_train


def _score_meta_frame(
    meta_df: pd.DataFrame, meta_reg, feature_cols: list[str]
) -> pd.DataFrame:
    scored = meta_df.copy().replace([np.inf, -np.inf], np.nan)
    scored = scored.dropna(subset=["window_smape"]).copy()
    if len(scored) == 0:
        scored["u_hat"] = pd.Series(dtype=float)
        return scored

    scored["u_hat"] = _predict_u_hat(meta_reg, scored, feature_cols=feature_cols)
    return scored


def _split_meta_holdout_windows(
    meta_eval: pd.DataFrame,
    holdout_frac: float = DEFAULT_DS2_HOLDOUT_FRAC,
    holdout_min_windows: int = DEFAULT_DS2_HOLDOUT_MIN_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    if len(meta_eval) == 0:
        empty = meta_eval.copy()
        return empty, empty, empty, 0

    def _choose_split_time(sdf: pd.DataFrame):
        starts = np.sort(pd.unique(sdf["target_start"]))
        n = int(starts.size)
        if n <= int(holdout_min_windows):
            return pd.NaT
        n_hold = max(int(holdout_min_windows), int(np.ceil(float(holdout_frac) * n)))
        n_hold = min(n_hold, n - 1)
        k = n - n_hold - 1
        k = max(0, min(k, n - 2))
        return starts[k]

    split_tbl = (
        meta_eval.groupby(ID_COL, observed=True)
        .apply(_choose_split_time)
        .reset_index(name="holdout_split_time")
    )

    eval_df = meta_eval.merge(split_tbl, on=ID_COL, how="left")
    eval_df = eval_df.dropna(subset=["holdout_split_time"]).copy()

    is_train_window = eval_df["target_end"] <= eval_df["holdout_split_time"]
    is_holdout_window = eval_df["target_start"] > eval_df["holdout_split_time"]
    is_boundary_window = ~(is_train_window | is_holdout_window)

    return (
        eval_df,
        eval_df[is_train_window].copy(),
        eval_df[is_holdout_window].copy(),
        int(is_boundary_window.sum()),
    )


def _summarize_rc_artifact(
    rc: pd.DataFrame,
    *,
    overlay_path: Path | None,
    score_col: str,
) -> dict:
    return {
        "aurc_model": _safe_float(rc["aurc_model"].iloc[0]),
        "aurc_random": _safe_float(rc["aurc_random"].iloc[0]),
        "aurc_oracle": _safe_float(rc["aurc_oracle"].iloc[0]),
        "auco_model": _safe_float(rc["auco_model"].iloc[0]),
        "error_drop_model": _safe_float(rc["error_drop_model"].iloc[0]),
        "rc_table_path": None,
        "rc_plot_path": None if overlay_path is None else str(overlay_path),
        "rc_plot_path_no_legend": None,
        "rc_plot_path_normalized": None,
        "reject_score": str(score_col),
        "curve_points": _serialize_risk_coverage_curve(rc),
    }


def _save_meta_bundle(
    path: Path,
    *,
    meta_reg,
    feature_cols: list[str],
    train_meta: pd.DataFrame,
    train_domain: str,
    eval_domain: str,
) -> str:
    joblib.dump(
        {
            "meta_reg": meta_reg,
            "feature_cols": feature_cols,
            "u_target_definition": "empirical_percentile_rank_of_training_window_smape",
            "train_domain": str(train_domain),
            "eval_domain": str(eval_domain),
            "train_rows": int(len(train_meta)),
        },
        path,
    )
    return str(path)


def _evaluate_reject_scores(
    meta_scored: pd.DataFrame,
    *,
    art: RunArtifacts,
    tag: str,
    title: str,
    seed: int,
    overlay_filename: str | None = None,
    save_plot: bool = True,
) -> tuple[dict, dict]:
    if meta_scored is None or len(meta_scored) == 0:
        return {}, {}

    rc_summaries: dict[str, dict] = {}
    rc_curves: dict[str, pd.DataFrame] = {}
    grid = _make_reject_grid()

    for name, col in _available_reject_scores(meta_scored):
        if col not in meta_scored.columns:
            continue
        rc_input = meta_scored.dropna(subset=[col, "window_smape"]).copy()
        rc = _risk_coverage_curve(
            meta_all=rc_input,
            score_col=col,
            err_col="window_smape",
            grid=np.array(grid, dtype=float),
            random_reps=int(DEFAULT_RANDOM_REPS),
            random_seed=int(seed),
        )
        if len(rc) == 0:
            continue

        rc_curves[name] = rc
        rc_summaries[name] = _summarize_rc_artifact(
            rc,
            overlay_path=None,
            score_col=col,
        )

    overlay_paths: dict[str, str | None] = {}
    if save_plot and len(rc_curves) >= 1:
        filename = (
            str(Path(str(overlay_filename)).with_suffix(".pdf"))
            if overlay_filename is not None
            else f"risk_coverage_{tag}.pdf"
        )
        overlay_path = art.plots_dir / filename
        _plot_risk_coverage_curves_multi(
            rc_curves,
            outpath=overlay_path,
            title=title,
            normalize=False,
            include_oracle_random=True,
            oracle_from="u_hat",
            show_legend=True,
        )
        overlay_paths = {
            "overlay_plot_path": str(overlay_path),
            "overlay_plot_path_no_legend": None,
        }
        for summary in rc_summaries.values():
            summary["rc_plot_path"] = str(overlay_path)
            summary["rc_plot_path_no_legend"] = None

    return rc_summaries, overlay_paths


def _summarize_reject_regime(
    meta_all_scored: pd.DataFrame,
    *,
    meta_holdout_scored: pd.DataFrame | None,
    rc_summaries: dict,
    overlay_paths: dict,
    train_rows: int | None = None,
    boundary_dropped: int | None = None,
) -> dict:
    all_df = meta_all_scored if meta_all_scored is not None else pd.DataFrame()
    holdout_df = meta_holdout_scored if meta_holdout_scored is not None else all_df
    primary_df = holdout_df if meta_holdout_scored is not None else all_df

    metamodel_all = _summarize_metamodel_ranking(all_df)
    metamodel_primary = _summarize_metamodel_ranking(primary_df)

    summary = {
        "evaluation_scope": (
            "holdout_windows" if meta_holdout_scored is not None else "all_windows"
        ),
        "eval_rows": int(len(primary_df)),
        "holdout_window_rows": int(len(holdout_df)),
        "baseline_mean_window_smape": _safe_float(
            primary_df["window_smape"].mean() if len(primary_df) else np.nan
        ),
        "u_hat_vs_window_smape_spearman": metamodel_primary["spearman"],
        "metamodel_evaluation": metamodel_primary,
        "score_evaluations": _summarize_available_score_rankings(primary_df),
        "risk_coverage_summary": rc_summaries,
        "score_columns": [name for name, _ in _available_reject_scores(primary_df)],
        **overlay_paths,
    }
    if meta_holdout_scored is not None:
        summary["full_eval_diagnostics"] = {
            "eval_rows": int(len(all_df)),
            "baseline_mean_window_smape": _safe_float(
                all_df["window_smape"].mean() if len(all_df) else np.nan
            ),
            "u_hat_vs_window_smape_spearman": metamodel_all["spearman"],
            "metamodel_evaluation": metamodel_all,
            "score_evaluations": _summarize_available_score_rankings(all_df),
        }
    if train_rows is not None:
        summary["train_window_rows"] = int(train_rows)
    if boundary_dropped is not None:
        summary["boundary_window_rows_dropped"] = int(boundary_dropped)
    return summary


def _format_float_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _normalize_q_values(values: Sequence[float] | None) -> list[float]:
    if values is None:
        return []

    out = []
    seen = set()
    for value in values:
        q = float(value)
        if not (0.0 <= q < 1.0):
            raise ValueError(f"Meta-model reject q must satisfy 0 <= q < 1, got {q}.")
        key = round(q, 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _normalize_forecast_models(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return ["AutoKAN"]

    out = []
    seen = set()
    for value in values:
        model = str(value)
        if model not in AUTO_MODELS:
            raise ValueError(
                f"Unknown forecast model {model!r}. Choose from {sorted(AUTO_MODELS)}."
            )
        if model in seen:
            continue
        seen.add(model)
        out.append(model)

    if not out:
        raise ValueError("At least one forecast model must be provided.")
    return out


def _clone_args(args, **updates):
    cloned = argparse.Namespace(**vars(args))
    for key, value in updates.items():
        setattr(cloned, key, value)
    return cloned


def _fixed_reject_threshold_from_scores(
    scores: np.ndarray, q: float
) -> tuple[float, int, float]:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise RuntimeError(
            "Cannot calibrate Meta-model reject threshold: empty score set."
        )

    q = float(q)
    if not (0.0 <= q < 1.0):
        raise ValueError(f"Meta-model reject q must satisfy 0 <= q < 1, got {q}.")

    n_target_reject = int(np.floor(q * s.size + 1e-12))
    if n_target_reject <= 0:
        return float("inf"), 0, 0.0

    order = np.argsort(-s, kind="mergesort")
    threshold = float(np.min(s[order[:n_target_reject]]))
    realized_rate = float(np.mean(s >= threshold))
    return threshold, int(n_target_reject), realized_rate


def _accept_reject_stats(err: np.ndarray, reject_mask: np.ndarray) -> dict:
    e = np.asarray(err, dtype=float)
    reject_mask = np.asarray(reject_mask, dtype=bool)
    ok = np.isfinite(e)
    e = e[ok]
    reject_mask = reject_mask[ok]

    n_total = int(e.size)
    n_reject = int(np.sum(reject_mask))
    n_accept = int(n_total - n_reject)
    accept_mask = ~reject_mask

    accepted_mean = float(np.mean(e[accept_mask])) if n_accept else np.nan
    rejected_mean = float(np.mean(e[reject_mask])) if n_reject else np.nan

    return {
        "n_total": n_total,
        "n_accept": n_accept,
        "n_reject": n_reject,
        "coverage": (n_accept / n_total) if n_total else np.nan,
        "reject_rate": (n_reject / n_total) if n_total else np.nan,
        "accepted_mean_window_smape": accepted_mean,
        "rejected_mean_window_smape": rejected_mean,
    }


def _oracle_reject_mask_from_n(err: np.ndarray, n_reject: int) -> np.ndarray:
    e = np.asarray(err, dtype=float)
    ok = np.isfinite(e)
    valid_idx = np.flatnonzero(ok)
    n_total = int(valid_idx.size)
    n_reject = int(np.clip(int(n_reject), 0, n_total))
    mask = np.zeros(e.shape[0], dtype=bool)
    if n_reject <= 0 or n_total == 0:
        return mask
    order = np.argsort(-e[ok], kind="mergesort")
    mask[valid_idx[order[:n_reject]]] = True
    return mask


def _oracle_reject_mask_from_q(err: np.ndarray, q: float) -> np.ndarray:
    e = np.asarray(err, dtype=float)
    n_valid = int(np.isfinite(e).sum())
    n_reject = int(np.floor(float(q) * n_valid + 1e-12))
    return _oracle_reject_mask_from_n(e, n_reject)


def _random_reject_mask_from_q(err: np.ndarray, q: float, *, seed: int) -> np.ndarray:
    e = np.asarray(err, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(e))
    n_valid = int(valid_idx.size)
    n_reject = int(np.floor(float(q) * n_valid + 1e-12))
    n_reject = int(np.clip(n_reject, 0, n_valid))
    mask = np.zeros(e.shape[0], dtype=bool)
    if n_reject <= 0 or n_valid == 0:
        return mask
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(valid_idx, size=n_reject, replace=False)
    mask[chosen] = True
    return mask


def _stats_payload(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        if k in {"n_total", "n_accept", "n_reject"}:
            out[k] = int(v)
        else:
            out[k] = _safe_float(v)
    return out


def _sort_online_windows(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ("target_start", "cutoff", ID_COL) if c in df.columns]
    if not sort_cols:
        sort_cols = [ID_COL]
    return df.sort_values(sort_cols).reset_index(drop=True)


def _cumulative_incurred_error(err: np.ndarray, accept_mask: np.ndarray) -> np.ndarray:
    e = np.asarray(err, dtype=float)
    accept = np.asarray(accept_mask, dtype=bool)
    ok = np.isfinite(e) & accept
    return np.cumsum(np.where(ok, e, 0.0))


def _cumulative_accepted_mean(err: np.ndarray, accept_mask: np.ndarray) -> np.ndarray:
    e = np.asarray(err, dtype=float)
    accept = np.asarray(accept_mask, dtype=bool)
    ok = np.isfinite(e) & accept
    counts = np.cumsum(ok.astype(int))
    sums = np.cumsum(np.where(ok, e, 0.0))
    out = np.full(e.shape[0], np.nan, dtype=float)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    return out


def _last_finite(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else np.nan


def _format_plot_metric(value: float) -> str:
    return f"{float(value):.3f}" if np.isfinite(value) else "NA"


def _split_online_fit_calibration_windows(
    train_windows: pd.DataFrame,
    eval_windows: pd.DataFrame,
    *,
    calibration_frac: float = 0.2,
    min_fit_windows: int = 1,
    min_calibration_windows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_df = (
        train_windows.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["window_smape"])
    )
    eval_df = (
        eval_windows.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["window_smape"])
    )

    calibration_frac = float(np.clip(calibration_frac, 0.05, 0.95))
    min_fit_windows = max(1, int(min_fit_windows))
    min_calibration_windows = max(1, int(min_calibration_windows))

    fit_parts = []
    calibration_parts = []
    train_sizes = []
    dropped = []

    for uid, g in train_df.groupby(ID_COL, observed=True):
        g = _sort_online_windows(g)
        n = int(len(g))
        train_sizes.append(n)
        if n < (min_fit_windows + min_calibration_windows):
            dropped.append((str(uid), n))
            continue

        n_calibration = max(
            int(min_calibration_windows),
            int(np.ceil(calibration_frac * n)),
        )
        n_calibration = min(n_calibration, n - int(min_fit_windows))
        n_fit = int(n - n_calibration)
        if n_fit < min_fit_windows or n_calibration < min_calibration_windows:
            dropped.append((str(uid), n))
            continue

        fit_df = g.iloc[:n_fit].copy()
        calibration_df = g.iloc[n_fit:].copy()
        fit_df["online_source"] = "fit"
        calibration_df["online_source"] = "calibration"
        fit_parts.append(fit_df)
        calibration_parts.append(calibration_df)

    if not fit_parts or not calibration_parts:
        preview = ", ".join(f"{uid}:{n}" for uid, n in dropped[:10])
        more = "" if len(dropped) <= 10 else f", ... (+{len(dropped) - 10} more)"
        raise RuntimeError(
            "Target-domain online split is infeasible because no series has enough earlier target windows "
            "to reserve both fit and calibration subsets. "
            f"Need at least {min_fit_windows + min_calibration_windows} target-train windows per series, "
            f"but {len(dropped)} series failed: {preview}{more}."
        )

    fit_out = pd.concat(fit_parts, axis=0, ignore_index=True)
    calibration_out = pd.concat(calibration_parts, axis=0, ignore_index=True)
    eval_out = _sort_online_windows(eval_df.copy())
    if len(eval_out):
        eval_out["online_series_step"] = (
            eval_out.groupby(ID_COL, observed=True).cumcount() + 1
        ).astype(int)
        eval_out["online_source"] = "holdout_evaluation"

    split_summary = {
        "series_total_train": int(train_df[ID_COL].nunique()),
        "series_used_train": int(fit_out[ID_COL].nunique()) if len(fit_out) else 0,
        "series_used_calibration": (
            int(calibration_out[ID_COL].nunique()) if len(calibration_out) else 0
        ),
        "series_dropped_train_insufficient_windows": int(len(dropped)),
        "series_insufficient_for_per_series_split": int(len(dropped)),
        "train_windows_per_series_min": int(min(train_sizes)) if train_sizes else 0,
        "train_windows_per_series_median": _safe_float(
            float(np.median(train_sizes)) if train_sizes else np.nan
        ),
        "train_windows_per_series_max": int(max(train_sizes)) if train_sizes else 0,
        "online_split_strategy": "per_series_temporal",
        "calibration_frac": float(calibration_frac),
        "fit_rows": int(len(fit_out)),
        "calibration_rows": int(len(calibration_out)),
        "evaluation_rows_total": int(len(eval_out)),
    }
    return fit_out, calibration_out, eval_out, split_summary


def _empty_online_split_summary(
    *,
    strategy: str,
    fit_rows: int = 0,
    calibration_rows: int = 0,
    eval_rows: int = 0,
    skip_reason: str | None = None,
) -> dict:
    out = {
        "series_total_train": 0,
        "series_used_train": 0,
        "series_used_calibration": 0,
        "series_dropped_train_insufficient_windows": 0,
        "series_insufficient_for_per_series_split": 0,
        "train_windows_per_series_min": 0,
        "train_windows_per_series_median": np.nan,
        "train_windows_per_series_max": 0,
        "online_split_strategy": str(strategy),
        "calibration_frac": np.nan,
        "fit_rows": int(fit_rows),
        "calibration_rows": int(calibration_rows),
        "evaluation_rows_total": int(eval_rows),
    }
    if skip_reason is not None:
        out["skip_reason"] = str(skip_reason)
    return out


def _split_online_all_target_windows(
    all_windows: pd.DataFrame,
    *,
    fit_frac: float = 0.2,
    calibration_frac: float = 0.2,
    min_fit_windows: int = 1,
    min_calibration_windows: int = 1,
    min_eval_windows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = (
        all_windows.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["window_smape"])
    )
    fit_frac = float(np.clip(fit_frac, 0.05, 0.9))
    calibration_frac = float(np.clip(calibration_frac, 0.05, 0.9))
    min_fit_windows = max(1, int(min_fit_windows))
    min_calibration_windows = max(1, int(min_calibration_windows))
    min_eval_windows = max(1, int(min_eval_windows))

    fit_parts = []
    calibration_parts = []
    eval_parts = []
    train_sizes = []
    dropped = []
    min_required = int(min_fit_windows + min_calibration_windows + min_eval_windows)

    for uid, g in df.groupby(ID_COL, observed=True):
        g = _sort_online_windows(g)
        n = int(len(g))
        train_sizes.append(n)
        if n < min_required:
            dropped.append((str(uid), n))
            continue

        n_fit = max(int(min_fit_windows), int(np.floor(float(fit_frac) * n)))
        n_calibration = max(
            int(min_calibration_windows),
            int(np.ceil(float(calibration_frac) * n)),
        )
        max_fit_calibration = n - int(min_eval_windows)
        if n_fit + n_calibration > max_fit_calibration:
            overflow = int(n_fit + n_calibration - max_fit_calibration)
            reduce_calibration = min(
                overflow,
                max(0, n_calibration - int(min_calibration_windows)),
            )
            n_calibration -= reduce_calibration
            overflow -= reduce_calibration
            if overflow > 0:
                n_fit -= min(overflow, max(0, n_fit - int(min_fit_windows)))

        n_eval = int(n - n_fit - n_calibration)
        if (
            n_fit < min_fit_windows
            or n_calibration < min_calibration_windows
            or n_eval < min_eval_windows
        ):
            dropped.append((str(uid), n))
            continue

        fit_df = g.iloc[:n_fit].copy()
        calibration_df = g.iloc[n_fit : n_fit + n_calibration].copy()
        eval_df = g.iloc[n_fit + n_calibration :].copy()
        fit_df["online_source"] = "fit"
        calibration_df["online_source"] = "calibration"
        eval_df["online_source"] = "holdout_evaluation"
        eval_df["online_series_step"] = np.arange(1, len(eval_df) + 1, dtype=int)
        fit_parts.append(fit_df)
        calibration_parts.append(calibration_df)
        eval_parts.append(eval_df)

    if not fit_parts or not calibration_parts or not eval_parts:
        preview = ", ".join(f"{uid}:{n}" for uid, n in dropped[:10])
        more = "" if len(dropped) <= 10 else f", ... (+{len(dropped) - 10} more)"
        raise RuntimeError(
            "Target-domain online fallback split is infeasible because no series has "
            "enough target windows to reserve fit, calibration, and evaluation "
            f"subsets. Need at least {min_required} target windows per series, "
            f"but {len(dropped)} series failed: {preview}{more}."
        )

    fit_out = pd.concat(fit_parts, axis=0, ignore_index=True)
    calibration_out = pd.concat(calibration_parts, axis=0, ignore_index=True)
    eval_out = _sort_online_windows(pd.concat(eval_parts, axis=0, ignore_index=True))
    split_summary = {
        "series_total_train": int(df[ID_COL].nunique()),
        "series_used_train": int(fit_out[ID_COL].nunique()),
        "series_used_calibration": int(calibration_out[ID_COL].nunique()),
        "series_dropped_train_insufficient_windows": int(len(dropped)),
        "series_insufficient_for_per_series_split": int(len(dropped)),
        "train_windows_per_series_min": int(min(train_sizes)) if train_sizes else 0,
        "train_windows_per_series_median": _safe_float(
            float(np.median(train_sizes)) if train_sizes else np.nan
        ),
        "train_windows_per_series_max": int(max(train_sizes)) if train_sizes else 0,
        "online_split_strategy": "per_series_temporal_all_target_fallback",
        "online_split_fallback": True,
        "fit_frac": float(fit_frac),
        "calibration_frac": float(calibration_frac),
        "fit_rows": int(len(fit_out)),
        "calibration_rows": int(len(calibration_out)),
        "evaluation_rows_total": int(len(eval_out)),
    }
    return fit_out, calibration_out, eval_out, split_summary


def _choose_online_example_series(
    eval_df: pd.DataFrame,
    *,
    n_examples: int,
    seed: int,
    min_windows: int = 1,
    score_col: str | None = None,
) -> list[str]:
    if eval_df is None or len(eval_df) == 0:
        return []

    n_examples = max(1, int(n_examples))
    min_windows = max(1, int(min_windows))
    grouped = list(eval_df.groupby(ID_COL, observed=True))
    all_uids = [str(uid) for uid, _ in grouped]
    focus_uids = [str(uid) for uid, g in grouped if int(len(g)) >= int(min_windows)]
    chosen: list[str] = []

    if score_col is not None and score_col in eval_df.columns:
        scored_candidates = []
        for uid, g in grouped:
            if int(len(g)) < int(min_windows):
                continue
            scores = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
            finite = scores[np.isfinite(scores)]
            if finite.size == 0:
                continue
            scored_candidates.append(
                (
                    str(uid),
                    float(np.max(finite) - np.min(finite)),
                    float(np.max(finite)),
                    int(len(g)),
                )
            )
        scored_candidates.sort(key=lambda item: (-item[1], -item[2], -item[3], item[0]))
        chosen = [uid for uid, _, _, _ in scored_candidates[:n_examples]]
        if len(chosen) >= n_examples:
            return chosen[:n_examples]
        remaining = [uid for uid in all_uids if uid not in set(chosen)]
        focus_uids = [uid for uid in focus_uids if uid in remaining]
        all_uids = remaining

    rng = np.random.default_rng(int(seed))

    def _sample(values: list[str], k: int) -> list[str]:
        if k <= 0 or not values:
            return []
        if len(values) <= k:
            return list(values)
        idx = rng.choice(len(values), size=k, replace=False)
        return [values[int(i)] for i in np.sort(idx)]

    n_needed = n_examples - len(chosen)
    chosen.extend(_sample(focus_uids, min(n_needed, len(focus_uids))))
    if len(chosen) < n_examples:
        remaining = [uid for uid in all_uids if uid not in set(chosen)]
        chosen.extend(_sample(remaining, n_examples - len(chosen)))
    return chosen[:n_examples]


def _plot_online_example_series(
    decision_df: pd.DataFrame,
    *,
    outpath: Path,
    title: str,
    selected_uids: Sequence[str],
    metric: str,
) -> list[str]:
    if decision_df is None or len(decision_df) == 0 or not selected_uids:
        return []

    metric = str(metric)
    if metric not in {"cumulative_incurred", "cumulative_mean_smape"}:
        raise ValueError(f"Unknown online example metric {metric!r}.")

    n_rows = len(selected_uids)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.8, max(2.7 * n_rows, 4.8)),
        squeeze=False,
    )

    for i, uid in enumerate(selected_uids):
        ax = axes[i, 0]
        g = _sort_online_windows(
            decision_df[decision_df[ID_COL].astype(str) == uid].copy()
        )
        if len(g) == 0:
            ax.set_visible(False)
            continue

        fallback_steps = pd.Series(
            np.arange(1, len(g) + 1, dtype=int), index=g.index, dtype=float
        )
        if "online_series_step" in g.columns:
            step_series = pd.to_numeric(g["online_series_step"], errors="coerce")
            step_series = step_series.where(step_series.notna(), fallback_steps)
        else:
            step_series = fallback_steps
        steps = step_series.to_numpy(dtype=int)
        err = pd.to_numeric(g["window_smape"], errors="coerce").to_numpy(dtype=float)
        meta_reject = g["reject_meta_model"].to_numpy(dtype=bool)

        valid_err = np.isfinite(err)
        meta_accept = valid_err & (~meta_reject)

        if metric == "cumulative_incurred":
            keep_curve = _cumulative_incurred_error(err, valid_err)
            meta_curve = _cumulative_incurred_error(err, meta_accept)
            valid_n = int(np.sum(valid_err))
            keep_final = (float(keep_curve[-1]) / valid_n) if valid_n else np.nan
            meta_final = (float(meta_curve[-1]) / valid_n) if valid_n else np.nan
            y_label = "Cumulative\nsMAPE"
            keep_label = "Keep all"
            meta_label = "Meta abstain"
        else:
            keep_curve = _cumulative_accepted_mean(err, valid_err)
            meta_curve = _cumulative_accepted_mean(err, meta_accept)
            keep_final = _last_finite(keep_curve)
            meta_final = _last_finite(meta_curve)
            y_label = "Accepted mean\nsMAPE"
            keep_label = "Keep all"
            meta_label = "Meta kept"

        ax.plot(
            steps,
            keep_curve,
            color="#4C78A8",
            linestyle=":",
            marker="o",
            label=keep_label,
        )
        ax.plot(
            steps,
            meta_curve,
            color="#54A24B",
            linestyle="-",
            marker="D",
            label=meta_label,
        )

        ax.set_title(
            f"{uid}: keep={_format_plot_metric(keep_final)}, meta={_format_plot_metric(meta_final)}",
            loc="left",
            fontsize=17,
        )
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
        ax.set_ylabel(y_label)
        ax.legend(loc="best", frameon=True)
        if len(steps) <= 14:
            ax.set_xticks(steps)
        if i < n_rows - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("Evaluation window")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    _save_paper_figure(fig, outpath)
    plt.close(fig)
    return [str(uid) for uid in selected_uids]


def _plot_online_distribution_summary(
    decision_df: pd.DataFrame,
    *,
    outpath: Path,
    title: str,
    clip_upper_quantile: float | None = None,
    score_names: Sequence[str] | None = None,
) -> None:
    if decision_df is None or len(decision_df) == 0:
        return

    err = pd.to_numeric(decision_df["window_smape"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(err)

    if score_names is None:
        inferred = []
        for name in ("u_hat", "pi_width", "resid_scale", "err_var", "random", "oracle"):
            if _online_reject_col(name) in decision_df.columns:
                inferred.append(name)
        score_names = inferred
    score_names = _online_score_order(score_names or ["u_hat"])

    box_items = [
        {
            "label": "All",
            "values": err[valid],
            "color": "#6F6F6F",
            "kind": "all",
            "position": 1.0,
        }
    ]
    tick_positions = [1.0]
    tick_labels = ["All"]
    count_annotations = [(1.0, f"all\nn={int(np.sum(valid))}", "#555555")]
    center = 2.0
    for name in score_names:
        reject_col = _online_reject_col(name)
        if reject_col not in decision_df.columns:
            continue
        reject_mask = _bool_array(decision_df[reject_col])
        if reject_mask.shape[0] != err.shape[0]:
            continue
        color = str(_curve_plot_style(name).get("color") or "#777777")
        kept_values = err[valid & (~reject_mask)]
        rejected_values = err[valid & reject_mask]
        if kept_values.size == 0 and rejected_values.size == 0:
            continue
        tick_positions.append(center)
        tick_labels.append(_reject_score_tick_label(name))
        if kept_values.size > 0 and rejected_values.size > 0:
            kept_pos = center - 0.18
            rejected_pos = center + 0.18
        else:
            kept_pos = rejected_pos = center
        if kept_values.size > 0:
            box_items.append(
                {
                    "label": f"{_reject_score_tick_label(name)} kept",
                    "values": kept_values,
                    "color": color,
                    "kind": "kept",
                    "position": kept_pos,
                }
            )
        if rejected_values.size > 0:
            box_items.append(
                {
                    "label": f"{_reject_score_tick_label(name)} rejected",
                    "values": rejected_values,
                    "color": color,
                    "kind": "rejected",
                    "position": rejected_pos,
                }
            )
        count_annotations.append((center, f"rej.\nn={rejected_values.size}", color))
        center += 1.0

    box_items = [item for item in box_items if item["values"].size > 0]
    if not box_items:
        return

    clip_upper = None
    if clip_upper_quantile is not None:
        pooled = err[valid]
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size:
            clip_q = float(np.clip(clip_upper_quantile, 0.0, 1.0))
            clip_candidate = float(np.nanquantile(pooled, clip_q))
            if np.isfinite(clip_candidate) and clip_candidate > 0:
                clip_upper = clip_candidate

    fig_width = max(8.8, 1.22 * len(tick_labels) + 2.4)
    fig = plt.figure(figsize=(fig_width, 5.9))
    ax = fig.add_subplot(111)

    positions = np.asarray([item["position"] for item in box_items], dtype=float)
    bp = ax.boxplot(
        [item["values"] for item in box_items],
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 2.0},
    )
    for patch, item in zip(bp["boxes"], box_items):
        color = item["color"]
        kind = item["kind"]
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        if kind == "all":
            patch.set_facecolor("white")
            patch.set_edgecolor("#555555")
            patch.set_alpha(0.95)
            patch.set_hatch("..")
            patch.set_linewidth(1.8)
        elif kind == "kept":
            patch.set_alpha(0.34)
        else:
            patch.set_facecolor("white")
            patch.set_alpha(0.95)
            patch.set_hatch("//")
            patch.set_linewidth(1.8)

    rng = np.random.default_rng(0)
    for item in box_items:
        x = float(item["position"])
        values = item["values"]
        color = item["color"]
        kind = item["kind"]
        if values.size == 0:
            continue
        if values.size <= 180:
            sample = values
        else:
            sample = rng.choice(values, size=180, replace=False)
        jitter = rng.uniform(-0.055, 0.055, size=sample.size)
        marker = "x" if kind == "rejected" else "o"
        scatter_kwargs = {
            "s": 26 if kind == "rejected" else 20,
            "alpha": 0.46 if kind == "rejected" else 0.28,
            "color": color,
            "marker": marker,
        }
        if kind != "rejected":
            scatter_kwargs.update({"edgecolors": "white", "linewidths": 0.25})
        else:
            scatter_kwargs.update({"linewidths": 0.8})
        ax.scatter(
            np.full(sample.size, x) + jitter,
            sample,
            **scatter_kwargs,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel("Window sMAPE")
    ax.set_title(title)
    if clip_upper is not None and np.isfinite(clip_upper) and clip_upper > 0:
        ax.set_ylim(0.0, clip_upper)
    ax.set_xlim(0.55, max(tick_positions) + 0.55)
    for x, text, color in count_annotations:
        ax.text(
            x,
            0.985,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color=color,
            linespacing=0.95,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch="..",
            label="All",
        ),
        Patch(
            facecolor="#777777",
            edgecolor="#555555",
            alpha=0.34,
            label="Kept",
        ),
        Patch(
            facecolor="white",
            edgecolor="#555555",
            hatch="//",
            label="Rejected",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        borderaxespad=0.9,
        ncol=3,
        frameon=False,
    )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    plt.tight_layout()
    _save_paper_figure(fig, outpath)
    plt.close(fig)


def _plot_online_forecast_windows(
    decision_df: pd.DataFrame,
    pointwise_df: pd.DataFrame | None,
    *,
    outpath: Path,
    title: str,
    selected_uids: Sequence[str],
) -> list[str]:
    if (
        decision_df is None
        or len(decision_df) == 0
        or pointwise_df is None
        or len(pointwise_df) == 0
        or not selected_uids
    ):
        return []

    required_pointwise = {ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"}
    if not required_pointwise.issubset(pointwise_df.columns):
        return []

    decisions = decision_df[
        (
            [ID_COL, "cutoff", "reject_meta_model", "online_series_step"]
            if "online_series_step" in decision_df.columns
            else [ID_COL, "cutoff", "reject_meta_model"]
        )
    ].copy()
    decisions[ID_COL] = decisions[ID_COL].astype(str)
    decisions = decisions.drop_duplicates(subset=[ID_COL, "cutoff"])

    pw = pointwise_df[[ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]].copy()
    pw[ID_COL] = pw[ID_COL].astype(str)
    plot_df = pw.merge(decisions, on=[ID_COL, "cutoff"], how="inner")
    if len(plot_df) == 0:
        return []

    n_rows = len(selected_uids)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(11.5, max(4.2 * n_rows, 5.6)),
        squeeze=False,
        sharex=True,
    )

    plotted = []
    for i, uid in enumerate(selected_uids):
        ax = axes[i, 0]
        uid_str = str(uid)
        g = plot_df[plot_df[ID_COL] == uid_str].copy()
        if len(g) == 0:
            ax.set_visible(False)
            continue

        g = g.sort_values(["cutoff", DS_COL])
        actual = (
            g[[DS_COL, Y_COL]]
            .drop_duplicates(subset=[DS_COL])
            .sort_values(DS_COL)
            .copy()
        )
        ax.plot(
            actual[DS_COL],
            actual[Y_COL],
            color="#4C78A8",
            linestyle="-",
            marker="o",
            markersize=3.4,
            linewidth=2.2,
            label="Actual",
            zorder=4,
        )

        used_labels = {"accepted": False, "rejected": False}
        for cutoff, wg in g.groupby("cutoff", sort=False):
            wg = wg.sort_values(DS_COL)
            rejected = bool(wg["reject_meta_model"].iloc[0])
            key = "rejected" if rejected else "accepted"
            ax.plot(
                wg[DS_COL],
                wg["y_hat"],
                color=("#E45756" if rejected else "#54A24B"),
                linestyle=("--" if rejected else "-"),
                marker=("x" if rejected else "D"),
                markersize=(4.2 if rejected else 3.8),
                linewidth=(1.8 if rejected else 2.0),
                alpha=(0.82 if rejected else 0.88),
                label=(
                    "Rejected forecast"
                    if rejected and not used_labels[key]
                    else (
                        "Accepted forecast"
                        if not rejected and not used_labels[key]
                        else None
                    )
                ),
                zorder=(3 if rejected else 5),
            )
            used_labels[key] = True

        n_windows = int(g[["cutoff"]].drop_duplicates().shape[0])
        n_rejected = int(
            g[[ID_COL, "cutoff", "reject_meta_model"]]
            .drop_duplicates()["reject_meta_model"]
            .sum()
        )
        ax.set_title(
            f"rejected={n_rejected}/{n_windows}",
            loc="left",
        )
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.yaxis.set_visible(False)
        for spine_name in ("left", "right"):
            ax.spines[spine_name].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.03),
            borderaxespad=0.0,
            frameon=True,
            ncol=3,
            fontsize=16,
        )
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_xlabel("Evaluation target time")
        plotted.append(uid_str)

    fig.suptitle(title, y=1.01)
    for ax in axes[:, 0]:
        if not ax.get_visible():
            continue
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), visible=True, rotation=30, ha="right")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), h_pad=2.0)
    _save_paper_figure(fig, outpath)
    plt.close(fig)
    return plotted


def _evaluate_online_q(
    *,
    calibration_scored: pd.DataFrame,
    eval_scored: pd.DataFrame,
    art: RunArtifacts,
    artifact_prefix: str,
    tag: str,
    title: str,
    q: float,
    seed: int,
    selected_series_ids: Sequence[str],
    pointwise_df: pd.DataFrame | None = None,
    score_col: str = "u_hat",
    err_col: str = "window_smape",
) -> dict:
    calib_df = calibration_scored.copy()
    eval_df = _sort_online_windows(eval_scored.copy())
    eval_df[err_col] = pd.to_numeric(eval_df[err_col], errors="coerce")

    err = eval_df[err_col].to_numpy(dtype=float)
    oracle_mask = _oracle_reject_mask_from_q(err, q=float(q))
    random_seed = int(seed) + int(round(float(q) * 1_000_000)) + 7919
    random_mask = _random_reject_mask_from_q(err, q=float(q), seed=random_seed)
    keep_all_stats = _accept_reject_stats(err, np.zeros(len(eval_df), dtype=bool))
    oracle_stats = _accept_reject_stats(err, oracle_mask)
    random_stats = _accept_reject_stats(err, random_mask)

    keep_all_acc = keep_all_stats["accepted_mean_window_smape"]
    oracle_acc = oracle_stats["accepted_mean_window_smape"]

    score_specs = [
        (name, col)
        for name, col in _available_reject_scores(eval_df)
        if col in calib_df.columns
    ]
    if score_col in eval_df.columns and score_col in calib_df.columns:
        if not any(col == score_col for _, col in score_specs):
            score_specs.insert(0, ("u_hat", score_col))
    score_specs = [
        (name, col)
        for name, col in score_specs
        if pd.to_numeric(calib_df[col], errors="coerce").notna().any()
    ]

    score_baselines: dict[str, dict] = {}
    score_reject_masks: dict[str, np.ndarray] = {}

    def _score_gain_payload(stats: dict) -> dict:
        model_acc = stats["accepted_mean_window_smape"]
        abs_gain_vs_keep_all = np.nan
        rel_gain_vs_keep_all = np.nan
        gap_to_oracle = np.nan
        if np.isfinite(model_acc) and np.isfinite(keep_all_acc):
            abs_gain_vs_keep_all = keep_all_acc - model_acc
            if keep_all_acc > 0:
                rel_gain_vs_keep_all = abs_gain_vs_keep_all / keep_all_acc
        if np.isfinite(model_acc) and np.isfinite(oracle_acc):
            gap_to_oracle = model_acc - oracle_acc
        return {
            "accepted_mean_window_smape_gain_vs_keep_all": _safe_float(
                abs_gain_vs_keep_all
            ),
            "accepted_mean_window_smape_relative_gain_vs_keep_all": _safe_float(
                rel_gain_vs_keep_all
            ),
            "accepted_mean_window_smape_gap_to_oracle_q": _safe_float(gap_to_oracle),
        }

    for name, col in score_specs:
        calib_df[col] = pd.to_numeric(calib_df[col], errors="coerce")
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
        try:
            threshold, target_n_reject, calib_reject_rate = (
                _fixed_reject_threshold_from_scores(
                    calib_df[col].to_numpy(dtype=float), q=float(q)
                )
            )
        except RuntimeError:
            continue

        score_values = eval_df[col].to_numpy(dtype=float)
        invalid_score_mask = ~np.isfinite(score_values)
        reject_mask = invalid_score_mask.copy()
        if np.isfinite(threshold):
            reject_mask |= score_values >= float(threshold)
        stats = _accept_reject_stats(err, reject_mask)
        score_reject_masks[str(name)] = reject_mask
        score_baselines[str(name)] = {
            "label": _reject_score_display_name(str(name)),
            "score_col": str(col),
            "threshold": None if not np.isfinite(threshold) else float(threshold),
            "calibration_rows": int(
                np.isfinite(calib_df[col].to_numpy(dtype=float)).sum()
            ),
            "calibration_target_n_reject": int(target_n_reject),
            "calibration_realized_reject_rate": _safe_float(calib_reject_rate),
            "missing_score_rows": int(np.sum(invalid_score_mask)),
            "stats": _stats_payload(stats),
            **_score_gain_payload(stats),
        }

    if "u_hat" not in score_baselines:
        raise RuntimeError("Online evaluation could not calibrate the u_hat score.")

    model_stats = score_baselines["u_hat"]["stats"]
    reference_baselines = {
        "random": {
            "label": _reject_score_display_name("random"),
            "stats": _stats_payload(random_stats),
            "seed": int(random_seed),
            **_score_gain_payload(random_stats),
        },
        "oracle": {
            "label": _reject_score_display_name("oracle"),
            "stats": _stats_payload(oracle_stats),
            **_score_gain_payload(oracle_stats),
        },
    }

    decision_df = eval_df.copy()
    decision_df["q"] = float(q)
    for name, reject_mask in score_reject_masks.items():
        reject_col = _online_reject_col(name)
        accept_col = _online_accept_col(name)
        decision_df[reject_col] = reject_mask.astype(bool)
        decision_df[accept_col] = (~reject_mask).astype(bool)
    decision_df["reject_oracle_q"] = oracle_mask.astype(bool)
    decision_df["accept_oracle_q"] = (~oracle_mask).astype(bool)
    decision_df["reject_random_q"] = random_mask.astype(bool)
    decision_df["accept_random_q"] = (~random_mask).astype(bool)

    keep_cols = [
        c
        for c in [
            ID_COL,
            "cutoff",
            "target_start",
            "target_end",
            "online_source",
            "online_series_step",
            *[col for _, col in score_specs],
            err_col,
            "q",
            *[
                col
                for name in _online_score_order(score_reject_masks.keys())
                for col in (_online_reject_col(name), _online_accept_col(name))
            ],
            "reject_random_q",
            "accept_random_q",
            "reject_oracle_q",
            "accept_oracle_q",
        ]
        if c in decision_df.columns
    ]
    keep_cols = list(dict.fromkeys(keep_cols))
    decision_df = decision_df[keep_cols].copy()

    q_tag = _format_float_tag(float(q))
    log_path = art.outdir / f"{artifact_prefix}_{tag}_q{q_tag}_decisions.csv"
    online_plots_dir = art.plots_dir / "online_plots"
    incurred_dir = online_plots_dir / "cumulative_incurred"
    mean_dir = online_plots_dir / "cumulative_mean_smape"
    distribution_dir = online_plots_dir / "distributions"
    distribution_clipped_dir = online_plots_dir / "distributions_clipped"
    forecast_dir = online_plots_dir / "forecast_windows"
    online_plots_dir.mkdir(parents=True, exist_ok=True)
    incurred_dir.mkdir(parents=True, exist_ok=True)
    mean_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir.mkdir(parents=True, exist_ok=True)
    distribution_clipped_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)

    example_plot_paths = {
        "cumulative_incurred": str(
            incurred_dir / f"{artifact_prefix}_{tag}_q{q_tag}_examples.pdf"
        ),
        "cumulative_mean_smape": str(
            mean_dir / f"{artifact_prefix}_{tag}_q{q_tag}_examples.pdf"
        ),
    }
    forecast_plot_path = (
        forecast_dir / f"{artifact_prefix}_{tag}_q{q_tag}_forecast_windows.pdf"
    )
    forecast_pointwise_path = (
        forecast_dir
        / f"{artifact_prefix}_{tag}_q{q_tag}_forecast_windows_pointwise.csv"
    )
    distribution_plot_path = (
        distribution_dir / f"{artifact_prefix}_{tag}_q{q_tag}_distribution.pdf"
    )
    distribution_plot_path_clipped = (
        distribution_clipped_dir
        / f"{artifact_prefix}_{tag}_q{q_tag}_distribution_clipped.pdf"
    )
    decision_df.to_csv(log_path, index=False)
    short_title = f"q={q:.2f}"
    if str(title or "").strip():
        short_title = f"{short_title} | {title}"
    selected_series = _plot_online_example_series(
        decision_df,
        outpath=Path(example_plot_paths["cumulative_incurred"]),
        title=f"Cumulative incurred | {short_title}",
        selected_uids=list(selected_series_ids),
        metric="cumulative_incurred",
    )
    _plot_online_example_series(
        decision_df,
        outpath=Path(example_plot_paths["cumulative_mean_smape"]),
        title=f"Accepted mean | {short_title}",
        selected_uids=list(selected_series_ids),
        metric="cumulative_mean_smape",
    )
    forecast_pointwise_rows = 0
    if pointwise_df is not None and len(pointwise_df) > 0:
        required_pointwise = {ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"}
        if required_pointwise.issubset(pointwise_df.columns):
            decision_keys = decision_df[[ID_COL, "cutoff"]].drop_duplicates().copy()
            decision_keys[ID_COL] = decision_keys[ID_COL].astype(str)
            pointwise_save = pointwise_df[
                [ID_COL, "cutoff", DS_COL, Y_COL, "y_hat"]
            ].copy()
            pointwise_save[ID_COL] = pointwise_save[ID_COL].astype(str)
            pointwise_save = pointwise_save.merge(
                decision_keys,
                on=[ID_COL, "cutoff"],
                how="inner",
            )
            if len(pointwise_save) > 0:
                pointwise_save = pointwise_save.sort_values([ID_COL, "cutoff", DS_COL])
                pointwise_save.to_csv(forecast_pointwise_path, index=False)
                forecast_pointwise_rows = int(len(pointwise_save))
    forecast_series = _plot_online_forecast_windows(
        decision_df,
        pointwise_df,
        outpath=forecast_plot_path,
        title=f"Forecast windows | {short_title}",
        selected_uids=list(selected_series_ids),
    )
    _plot_online_distribution_summary(
        decision_df,
        outpath=distribution_plot_path,
        title=f"Kept vs rejected | q={q:.2f}",
        score_names=[*score_baselines.keys(), "random", "oracle"],
    )
    _plot_online_distribution_summary(
        decision_df,
        outpath=distribution_plot_path_clipped,
        title=f"Kept vs rejected, clipped | q={q:.2f}",
        clip_upper_quantile=float(DEFAULT_ONLINE_DISTRIBUTION_CLIP_QUANTILE),
        score_names=[*score_baselines.keys(), "random", "oracle"],
    )

    return {
        "q": float(q),
        "threshold": score_baselines["u_hat"].get("threshold"),
        "calibration_rows": int(score_baselines["u_hat"].get("calibration_rows") or 0),
        "calibration_target_n_reject": int(
            score_baselines["u_hat"].get("calibration_target_n_reject") or 0
        ),
        "calibration_realized_reject_rate": _safe_float(
            score_baselines["u_hat"].get("calibration_realized_reject_rate")
        ),
        "missing_score_rows": int(
            score_baselines["u_hat"].get("missing_score_rows") or 0
        ),
        "keep_all": _stats_payload(keep_all_stats),
        "meta_model": model_stats,
        "oracle_q": _stats_payload(oracle_stats),
        "random_q": _stats_payload(random_stats),
        "score_baselines": score_baselines,
        "reference_baselines": reference_baselines,
        "accepted_mean_window_smape_gain_vs_keep_all": _safe_float(
            score_baselines["u_hat"].get("accepted_mean_window_smape_gain_vs_keep_all")
        ),
        "accepted_mean_window_smape_relative_gain_vs_keep_all": _safe_float(
            score_baselines["u_hat"].get(
                "accepted_mean_window_smape_relative_gain_vs_keep_all"
            )
        ),
        "accepted_mean_window_smape_gap_to_oracle_q": _safe_float(
            score_baselines["u_hat"].get("accepted_mean_window_smape_gap_to_oracle_q")
        ),
        "decision_log_path": str(log_path),
        "example_plot_path": example_plot_paths["cumulative_incurred"],
        "example_plot_paths": example_plot_paths,
        "example_plot_path_cumulative_incurred": example_plot_paths[
            "cumulative_incurred"
        ],
        "example_plot_path_cumulative_mean_smape": example_plot_paths[
            "cumulative_mean_smape"
        ],
        "forecast_windows_plot_path": (
            str(forecast_plot_path) if forecast_series else None
        ),
        "forecast_windows_pointwise_path": (
            str(forecast_pointwise_path) if forecast_pointwise_rows > 0 else None
        ),
        "forecast_windows_pointwise_rows": int(forecast_pointwise_rows),
        "distribution_plot_path": str(distribution_plot_path),
        "distribution_plot_path_clipped": str(distribution_plot_path_clipped),
        "example_series_ids": [str(uid) for uid in selected_series],
    }


def _online_summary_row(run_summary: dict) -> dict:
    keep_all_stats = run_summary["keep_all"]
    meta_stats = run_summary["meta_model"]
    oracle_stats = run_summary["oracle_q"]
    example_paths = run_summary.get("example_plot_paths", {}) or {}
    row = {
        "q": _safe_float(run_summary["q"]),
        "threshold": _safe_float(run_summary["threshold"]),
        "reject_rate": _safe_float(meta_stats["reject_rate"]),
        "coverage": _safe_float(meta_stats["coverage"]),
        "oracle_q_reject_rate": _safe_float(oracle_stats["reject_rate"]),
        "oracle_q_coverage": _safe_float(oracle_stats["coverage"]),
        "keep_all_accepted_mean_window_smape": _safe_float(
            keep_all_stats["accepted_mean_window_smape"]
        ),
        "meta_model_accepted_mean_window_smape": _safe_float(
            meta_stats["accepted_mean_window_smape"]
        ),
        "oracle_q_accepted_mean_window_smape": _safe_float(
            oracle_stats["accepted_mean_window_smape"]
        ),
        "meta_model_rejected_mean_window_smape": _safe_float(
            meta_stats["rejected_mean_window_smape"]
        ),
        "oracle_q_rejected_mean_window_smape": _safe_float(
            oracle_stats["rejected_mean_window_smape"]
        ),
        "accepted_mean_window_smape_gain_vs_keep_all": _safe_float(
            run_summary["accepted_mean_window_smape_gain_vs_keep_all"]
        ),
        "accepted_mean_window_smape_relative_gain_vs_keep_all": _safe_float(
            run_summary["accepted_mean_window_smape_relative_gain_vs_keep_all"]
        ),
        "accepted_mean_window_smape_gap_to_oracle_q": _safe_float(
            run_summary["accepted_mean_window_smape_gap_to_oracle_q"]
        ),
        "decision_log_path": str(run_summary["decision_log_path"]),
        "example_plot_path": str(run_summary["example_plot_path"]),
        "example_plot_path_cumulative_incurred": str(
            example_paths.get(
                "cumulative_incurred",
                run_summary.get("example_plot_path_cumulative_incurred"),
            )
        ),
        "example_plot_path_cumulative_mean_smape": str(
            example_paths.get(
                "cumulative_mean_smape",
                run_summary.get("example_plot_path_cumulative_mean_smape"),
            )
        ),
        "forecast_windows_plot_path": run_summary.get("forecast_windows_plot_path"),
        "forecast_windows_pointwise_path": run_summary.get(
            "forecast_windows_pointwise_path"
        ),
        "forecast_windows_pointwise_rows": int(
            run_summary.get("forecast_windows_pointwise_rows") or 0
        ),
        "distribution_plot_path": str(run_summary["distribution_plot_path"]),
        "distribution_plot_path_clipped": str(
            run_summary.get("distribution_plot_path_clipped", "")
        ),
        "example_series_ids": "|".join(run_summary.get("example_series_ids", [])),
    }
    for name, payload in (run_summary.get("score_baselines") or {}).items():
        stats = payload.get("stats", {}) or {}
        prefix = str(name)
        row[f"{prefix}_threshold"] = _safe_float(payload.get("threshold"))
        row[f"{prefix}_coverage"] = _safe_float(stats.get("coverage"))
        row[f"{prefix}_reject_rate"] = _safe_float(stats.get("reject_rate"))
        row[f"{prefix}_accepted_mean_window_smape"] = _safe_float(
            stats.get("accepted_mean_window_smape")
        )
        row[f"{prefix}_rejected_mean_window_smape"] = _safe_float(
            stats.get("rejected_mean_window_smape")
        )
        row[f"{prefix}_accepted_mean_window_smape_gap_to_oracle_q"] = _safe_float(
            payload.get("accepted_mean_window_smape_gap_to_oracle_q")
        )
    for name, payload in (run_summary.get("reference_baselines") or {}).items():
        stats = payload.get("stats", {}) or {}
        prefix = str(name)
        row[f"{prefix}_coverage"] = _safe_float(stats.get("coverage"))
        row[f"{prefix}_reject_rate"] = _safe_float(stats.get("reject_rate"))
        row[f"{prefix}_accepted_mean_window_smape"] = _safe_float(
            stats.get("accepted_mean_window_smape")
        )
        row[f"{prefix}_rejected_mean_window_smape"] = _safe_float(
            stats.get("rejected_mean_window_smape")
        )
        row[f"{prefix}_accepted_mean_window_smape_gap_to_oracle_q"] = _safe_float(
            payload.get("accepted_mean_window_smape_gap_to_oracle_q")
        )
    return row


def _run_online_panel_suite(
    *,
    fit_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pointwise_df: pd.DataFrame | None = None,
    art: RunArtifacts,
    artifact_prefix: str,
    tag: str,
    title_prefix: str,
    q_values: Sequence[float],
    seed: int,
    meta_model: str,
    feature_cols: list[str],
    tune_meta: bool,
    n_examples: int,
    split_summary: dict,
    label: str = "Metamodel (Temporal panel online)",
) -> dict:
    out = {
        "label": str(label),
        "score_col": "u_hat",
        "q_values": [float(q) for q in q_values],
        "status": "skipped",
        "summary_csv_path": None,
        "summary_plot_path": None,
        "runs": {},
        "fit_rows": int(len(fit_df)),
        "calibration_rows": int(len(calibration_df)),
        "eval_rows": int(len(eval_df)),
        **split_summary,
    }
    if len(fit_df) == 0:
        out.setdefault("skip_reason", "empty fit set")
        return out
    if len(calibration_df) == 0:
        out.setdefault("skip_reason", "empty calibration set")
        return out
    if len(eval_df) == 0:
        out.setdefault("skip_reason", "empty evaluation set")
        return out

    online_reg, online_train = _fit_reject_meta_regressor(
        fit_df,
        feature_cols=feature_cols,
        meta_model=str(meta_model),
        seed=int(seed),
        tune_meta=bool(tune_meta),
    )
    calibration_scored = _score_meta_frame(
        calibration_df,
        online_reg,
        feature_cols=feature_cols,
    )
    eval_scored = _score_meta_frame(
        eval_df,
        online_reg,
        feature_cols=feature_cols,
    )
    if len(calibration_scored) == 0:
        out.setdefault("skip_reason", "empty scored calibration set")
        return out
    if len(eval_scored) == 0:
        out.setdefault("skip_reason", "empty scored evaluation set")
        return out

    selected_series_ids = _choose_online_example_series(
        eval_scored,
        n_examples=int(n_examples),
        seed=int(seed),
        min_windows=1,
        score_col="u_hat",
    )
    run_summaries = {}
    row_summaries = []
    for q in q_values:
        q = float(q)
        q_key = f"q_{_format_float_tag(q)}"
        run_summary = _evaluate_online_q(
            calibration_scored=calibration_scored,
            eval_scored=eval_scored,
            art=art,
            artifact_prefix=str(artifact_prefix),
            tag=tag,
            title=title_prefix,
            q=q,
            seed=int(seed),
            selected_series_ids=selected_series_ids,
            pointwise_df=pointwise_df,
            score_col="u_hat",
            err_col="window_smape",
        )
        run_summaries[q_key] = run_summary
        row_summaries.append(_online_summary_row(run_summary))

    summary_csv_path = art.outdir / f"{artifact_prefix}_{tag}_summary.csv"
    summary_plot_dir = art.plots_dir / "online_plots" / "summary"
    summary_plot_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_path = summary_plot_dir / f"{artifact_prefix}_{tag}_summary.pdf"
    pd.DataFrame(row_summaries).to_csv(summary_csv_path, index=False)
    _plot_online_summary_across_qs(
        list(run_summaries.values()),
        outpath=summary_plot_path,
        title=f"Online summary | {title_prefix}",
    )
    first_run_scores = (
        next(iter(run_summaries.values())).get("score_baselines") or {}
        if run_summaries
        else {}
    )

    out.update(
        {
            "status": "completed",
            "summary_csv_path": str(summary_csv_path),
            "summary_plot_path": str(summary_plot_path),
            "fit_rows_used": int(len(online_train)),
            "calibration_rows_scored": int(len(calibration_scored)),
            "eval_rows_scored": int(len(eval_scored)),
            "example_series_ids": [str(uid) for uid in selected_series_ids],
            "score_columns": [
                name
                for name, _ in _available_reject_scores(eval_scored)
                if name in first_run_scores
            ],
            "reference_columns": ["random", "oracle"],
            "runs": run_summaries,
        }
    )
    return out


def _plot_online_summary_across_qs(
    run_summaries: Sequence[dict],
    *,
    outpath: Path,
    title: str,
) -> None:
    if run_summaries is None or len(run_summaries) == 0:
        return

    runs = sorted(run_summaries, key=lambda r: float(r["q"]))
    q_vals = np.asarray([float(r["q"]) for r in runs], dtype=float)
    keep_all_vals = np.asarray(
        [r["keep_all"]["accepted_mean_window_smape"] for r in runs], dtype=float
    )

    score_names = []
    reference_names = []
    for run in runs:
        score_names.extend((run.get("score_baselines") or {}).keys())
        reference_names.extend((run.get("reference_baselines") or {}).keys())
    if not score_names:
        score_names = ["u_hat"]
    if not reference_names:
        reference_names = ["oracle"]
    score_names = _online_score_order(score_names)
    reference_names = _online_score_order(reference_names)

    def _accepted_values(name: str) -> np.ndarray:
        values = []
        for run in runs:
            if name in (run.get("score_baselines") or {}):
                stats = (run["score_baselines"][name] or {}).get("stats", {}) or {}
                values.append(stats.get("accepted_mean_window_smape"))
            elif name == "u_hat":
                values.append(
                    (run.get("meta_model") or {}).get("accepted_mean_window_smape")
                )
            elif name == "oracle":
                values.append(
                    (run.get("oracle_q") or {}).get("accepted_mean_window_smape")
                )
            elif name == "random":
                stats = (run.get("reference_baselines") or {}).get("random", {})
                values.append(
                    (stats.get("stats", {}) or {}).get("accepted_mean_window_smape")
                )
            else:
                values.append(np.nan)
        return np.asarray(values, dtype=float)

    title_main, title_note = _split_plot_title(title)
    fig = plt.figure(figsize=(8.6, 5.5))
    ax = fig.add_subplot(111)
    ax.plot(
        q_vals,
        keep_all_vals,
        color="#6F6F6F",
        linestyle="--",
        marker="o",
        label="Keep all",
        markerfacecolor="white",
        markeredgewidth=1.5,
    )
    for name in score_names:
        vals = _accepted_values(name)
        if not np.isfinite(vals).any():
            continue
        style = _curve_plot_style(name)
        ax.plot(
            q_vals,
            vals,
            label=f"{_reject_score_display_name(name)} kept",
            markerfacecolor="white",
            markeredgewidth=1.5,
            **style,
        )
    for name in reference_names:
        vals = _accepted_values(name)
        if not np.isfinite(vals).any():
            continue
        style = _curve_plot_style(name)
        ax.plot(
            q_vals,
            vals,
            label=f"{_reject_score_display_name(name)} kept",
            markerfacecolor="white",
            markeredgewidth=1.5,
            **style,
        )
    ax.set_xlabel("Target reject fraction q")
    ax.set_ylabel("Accepted-window sMAPE")
    ax.set_title(title_main or title)
    ax.set_xticks(q_vals)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.legend(loc="best", frameon=True)
    if title_note:
        ax.text(
            0.01,
            0.99,
            title_note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    plt.tight_layout()
    _save_paper_figure(fig, outpath)
    plt.close(fig)


def _run_reject_evaluation_stage(
    args,
    art: RunArtifacts,
    forecast_stage: ForecastStageArtifacts,
    meta_stage: MetaStageArtifacts,
) -> tuple[dict, dict, dict]:
    save_model_artifacts = bool(getattr(args, "save_model_artifacts", False))
    ds1_label = _dataset_label(args.ds1_data, args.ds1_group, args.ds1_csv_path)
    ds2_label = _dataset_label(args.ds2_data, args.ds2_group, args.ds2_csv_path)
    plot_experiment_label = " | ".join(
        part
        for part in [
            f"{ds1_label}->{ds2_label}" if ds1_label or ds2_label else "",
            str(args.forecast_model or ""),
            (
                f"H={int(args.horizon)}"
                if getattr(args, "horizon", None) is not None
                else ""
            ),
        ]
        if part
    )
    ds1_final_display_name = (
        "In-domain evaluation with temporal holdout " "(out-of-sample forecasting)"
    )
    ds2_transfer_display_name = "Cross-domain generalization (zero-shot transfer)"
    ds2_oracle_display_name = (
        "Cross-domain generalization with target-domain adaptation "
        "(refit upper bound)"
    )

    transfer_meta_reg, transfer_meta_train = _fit_reject_meta_regressor(
        meta_stage.meta_train_base,
        feature_cols=meta_stage.feature_cols,
        meta_model=str(args.meta_model),
        seed=int(args.seed),
        tune_meta=bool(args.tune_meta),
    )

    transfer_meta_bundle_path = None
    if save_model_artifacts:
        if art.models_dir is None:
            raise RuntimeError("models_dir is required when saving model artifacts.")
        transfer_meta_bundle_path = _save_meta_bundle(
            art.models_dir
            / f"meta_bundle_{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}_{args.meta_model}_reject_transfer.joblib",
            meta_reg=transfer_meta_reg,
            feature_cols=meta_stage.feature_cols,
            train_meta=transfer_meta_train,
            train_domain=ds1_label,
            eval_domain=ds2_label,
        )

    ds1_final_scored = _score_meta_frame(
        meta_stage.meta_test_base,
        transfer_meta_reg,
        feature_cols=meta_stage.feature_cols,
    )
    ds1_final_rc, ds1_final_overlay = _evaluate_reject_scores(
        ds1_final_scored,
        art=art,
        tag=f"{meta_stage.safe_ds1}_final_test",
        title=f"Source holdout | {plot_experiment_label}",
        seed=int(args.seed),
        overlay_filename=f"in_domain_temporal_holdout_{meta_stage.safe_ds1}.pdf",
    )
    ds1_final_summary = _summarize_reject_regime(
        ds1_final_scored,
        meta_holdout_scored=None,
        rc_summaries=ds1_final_rc,
        overlay_paths=ds1_final_overlay,
        train_rows=len(transfer_meta_train),
    )
    ds1_final_summary["display_name"] = ds1_final_display_name
    ds1_final_summary["dataset_label"] = ds1_label
    online_qs = list(getattr(args, "online_qs", []) or [])

    meta2_eval, meta2_train_win, meta2_holdout, ds2_boundary = (
        _split_meta_holdout_windows(meta_stage.meta2_eval_base)
    )
    _dbg(
        "DS2_SPLIT",
        f"train_win={len(meta2_train_win)} holdout_win={len(meta2_holdout)} boundary_dropped={ds2_boundary}",
    )
    ds2_holdout_cutoffs = meta2_holdout[[ID_COL, "cutoff"]].drop_duplicates()
    ds2_transfer_forecast_holdout_summary = _summary_from_pointwise_subset(
        meta_stage.ds2_transfer_pointwise,
        ds2_holdout_cutoffs,
        horizon=int(args.horizon),
    )
    ds2_transfer_seasonal_naive_holdout_summary = _summary_from_pointwise_subset(
        meta_stage.ds2_seasonal_naive_pointwise,
        ds2_holdout_cutoffs,
        horizon=int(args.horizon),
    )

    ds2_transfer_eval_scored = _score_meta_frame(
        meta2_eval,
        transfer_meta_reg,
        feature_cols=meta_stage.feature_cols,
    )
    ds2_transfer_holdout_scored = _score_meta_frame(
        meta2_holdout,
        transfer_meta_reg,
        feature_cols=meta_stage.feature_cols,
    )
    ds2_transfer_rc, ds2_transfer_overlay = _evaluate_reject_scores(
        ds2_transfer_holdout_scored,
        art=art,
        tag=f"{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}_transfer_holdout",
        title=f"Zero-shot transfer | {plot_experiment_label}",
        seed=int(args.seed),
        overlay_filename=(
            f"cross_domain_zero_shot_transfer_{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}.pdf"
        ),
    )
    ds2_transfer_summary = _summarize_reject_regime(
        ds2_transfer_eval_scored,
        meta_holdout_scored=ds2_transfer_holdout_scored,
        rc_summaries=ds2_transfer_rc,
        overlay_paths=ds2_transfer_overlay,
        train_rows=len(transfer_meta_train),
        boundary_dropped=ds2_boundary,
    )
    ds2_transfer_summary["display_name"] = ds2_transfer_display_name
    ds2_transfer_summary["source_dataset_label"] = ds1_label
    ds2_transfer_summary["target_dataset_label"] = ds2_label

    ds2_oracle_bundle_path = None
    ds2_oracle_reg = None
    if len(meta2_train_win) > 0:
        ds2_oracle_reg, ds2_oracle_train = _fit_reject_meta_regressor(
            meta2_train_win,
            feature_cols=meta_stage.feature_cols,
            meta_model=str(args.meta_model),
            seed=int(args.seed),
            tune_meta=bool(args.tune_meta),
        )
        if save_model_artifacts:
            if art.models_dir is None:
                raise RuntimeError(
                    "models_dir is required when saving model artifacts."
                )
            ds2_oracle_bundle_path = _save_meta_bundle(
                art.models_dir
                / f"meta_bundle_{meta_stage.safe_ds2}_oracle_{args.meta_model}_reject.joblib",
                meta_reg=ds2_oracle_reg,
                feature_cols=meta_stage.feature_cols,
                train_meta=ds2_oracle_train,
                train_domain=f"{ds2_label} temporal-train",
                eval_domain=f"{ds2_label} temporal-holdout",
            )
        ds2_oracle_eval_scored = _score_meta_frame(
            meta2_eval,
            ds2_oracle_reg,
            feature_cols=meta_stage.feature_cols,
        )
        ds2_oracle_holdout_scored = _score_meta_frame(
            meta2_holdout,
            ds2_oracle_reg,
            feature_cols=meta_stage.feature_cols,
        )
    else:
        ds2_oracle_train = meta2_train_win.copy()
        ds2_oracle_eval_scored = meta2_eval.copy()
        ds2_oracle_holdout_scored = meta2_holdout.copy()
        for df in (ds2_oracle_eval_scored, ds2_oracle_holdout_scored):
            if "u_hat" not in df.columns:
                df["u_hat"] = pd.Series(dtype=float)

    ds2_oracle_rc, ds2_oracle_overlay = _evaluate_reject_scores(
        ds2_oracle_holdout_scored,
        art=art,
        tag=f"{meta_stage.safe_ds2}_oracle_holdout",
        title=f"Target adapted | {plot_experiment_label}",
        seed=int(args.seed),
        overlay_filename=(f"cross_domain_target_adaptation_{meta_stage.safe_ds2}.pdf"),
    )
    ds2_oracle_summary = _summarize_reject_regime(
        ds2_oracle_eval_scored,
        meta_holdout_scored=ds2_oracle_holdout_scored,
        rc_summaries=ds2_oracle_rc,
        overlay_paths=ds2_oracle_overlay,
        train_rows=len(ds2_oracle_train),
        boundary_dropped=ds2_boundary,
    )
    ds2_oracle_summary["display_name"] = ds2_oracle_display_name
    ds2_oracle_summary["source_dataset_label"] = ds1_label
    ds2_oracle_summary["target_dataset_label"] = ds2_label

    ds2_online_summary = None
    if online_qs:
        ds2_online_tag = (
            f"{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}_{args.meta_model}"
        )
        ds2_online_title = plot_experiment_label
        ds2_online_scope = "ds2_target_holdout_windows"
        primary_online_error = None
        try:
            if len(meta2_train_win) == 0 or len(meta2_holdout) == 0:
                raise RuntimeError(
                    "empty target temporal split for online evaluation: "
                    f"train_win={len(meta2_train_win)} holdout_win={len(meta2_holdout)}"
                )
            (
                ds2_online_fit,
                ds2_online_calibration,
                ds2_online_eval,
                ds2_online_split,
            ) = _split_online_fit_calibration_windows(
                meta2_train_win,
                meta2_holdout,
                calibration_frac=0.2,
                min_fit_windows=1,
                min_calibration_windows=1,
            )
        except RuntimeError as e:
            primary_online_error = str(e)
            _dbg(
                "ONLINE",
                f"Target-holdout online split unavailable: {primary_online_error} "
                "Trying all-target temporal fallback.",
            )
            try:
                (
                    ds2_online_fit,
                    ds2_online_calibration,
                    ds2_online_eval,
                    ds2_online_split,
                ) = _split_online_all_target_windows(
                    meta2_eval,
                    fit_frac=0.2,
                    calibration_frac=0.2,
                    min_fit_windows=1,
                    min_calibration_windows=1,
                    min_eval_windows=1,
                )
                ds2_online_scope = "ds2_target_all_windows_temporal_fallback"
                ds2_online_split["primary_split_skip_reason"] = primary_online_error
            except RuntimeError as fallback_error:
                _dbg("ONLINE", f"Online evaluation skipped: {fallback_error}")
                empty_online = meta2_eval.iloc[0:0].copy()
                ds2_online_fit = empty_online
                ds2_online_calibration = empty_online
                ds2_online_eval = empty_online
                ds2_online_scope = "ds2_target_online_unavailable"
                ds2_online_split = _empty_online_split_summary(
                    strategy="unavailable",
                    skip_reason=(
                        f"primary split failed: {primary_online_error}; "
                        f"fallback split failed: {fallback_error}"
                    ),
                )
        ds2_online_summary = _run_online_panel_suite(
            fit_df=ds2_online_fit,
            calibration_df=ds2_online_calibration,
            eval_df=ds2_online_eval,
            pointwise_df=meta_stage.ds2_transfer_pointwise,
            art=art,
            artifact_prefix="ds2_adapted_online",
            tag=ds2_online_tag,
            title_prefix=ds2_online_title,
            q_values=online_qs,
            seed=int(args.seed),
            meta_model=str(args.meta_model),
            feature_cols=meta_stage.feature_cols,
            tune_meta=bool(args.tune_meta),
            n_examples=int(args.online_n_example_series),
            split_summary={
                "display_name": "Target-domain adapted online abstention",
                "evaluation_scope": ds2_online_scope,
                **ds2_online_split,
            },
            label="Metamodel (Target-domain adapted online)",
        )
    else:
        ds2_online_summary = {
            "label": "Metamodel (Target-domain adapted online)",
            "score_col": "u_hat",
            "q_values": [],
            "status": "skipped",
            "skip_reason": "no online q values configured",
            "summary_csv_path": None,
            "summary_plot_path": None,
            "runs": {},
            "fit_rows": 0,
            "calibration_rows": 0,
            "eval_rows": 0,
            "display_name": "Target-domain adapted online abstention",
            "evaluation_scope": "ds2_target_online_unavailable",
            **_empty_online_split_summary(
                strategy="unavailable",
                skip_reason="no online q values configured",
            ),
        }

    shap_paths = {}
    if args.compute_shap:
        ds1_shap_tag = f"ds1_final_test_{meta_stage.safe_ds1}_{args.meta_model}_reject"
        ds2_transfer_shap_tag = f"ds2_transfer_holdout_{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}_{args.meta_model}_reject"
        ds2_oracle_shap_tag = (
            f"ds2_target_oracle_holdout_{meta_stage.safe_ds2}_{args.meta_model}_reject"
        )
        save_global_shap_plots(
            meta_clf=transfer_meta_reg,
            meta_df=ds1_final_scored,
            feature_cols=meta_stage.feature_cols,
            outdir=art.outdir,
            tag=ds1_shap_tag,
            max_display=int(DEFAULT_SHAP_MAX_DISPLAY),
            max_rows=int(DEFAULT_SHAP_MAX_ROWS),
            seed=int(args.seed),
        )
        if len(ds2_transfer_holdout_scored) > 0:
            save_global_shap_plots(
                meta_clf=transfer_meta_reg,
                meta_df=ds2_transfer_holdout_scored,
                feature_cols=meta_stage.feature_cols,
                outdir=art.outdir,
                tag=ds2_transfer_shap_tag,
                max_display=int(DEFAULT_SHAP_MAX_DISPLAY),
                max_rows=int(DEFAULT_SHAP_MAX_ROWS),
                seed=int(args.seed),
            )
        if ds2_oracle_reg is not None and len(ds2_oracle_holdout_scored) > 0:
            save_global_shap_plots(
                meta_clf=ds2_oracle_reg,
                meta_df=ds2_oracle_holdout_scored,
                feature_cols=meta_stage.feature_cols,
                outdir=art.outdir,
                tag=ds2_oracle_shap_tag,
                max_display=int(DEFAULT_SHAP_MAX_DISPLAY),
                max_rows=int(DEFAULT_SHAP_MAX_ROWS),
                seed=int(args.seed),
            )
        shap_paths = {
            "ds1_final_test_shap_summary": str(
                art.outdir / "plots" / f"shap_summary_{ds1_shap_tag}.pdf"
            ),
            "ds1_final_test_shap_global_importance": str(
                art.outdir / "plots" / f"shap_global_importance_{ds1_shap_tag}.csv"
            ),
            "ds2_transfer_holdout_shap_summary": str(
                art.outdir / "plots" / f"shap_summary_{ds2_transfer_shap_tag}.pdf"
            ),
            "ds2_transfer_holdout_shap_global_importance": str(
                art.outdir
                / "plots"
                / f"shap_global_importance_{ds2_transfer_shap_tag}.csv"
            ),
            "ds2_target_oracle_holdout_shap_summary": (
                None
                if ds2_oracle_reg is None
                else str(
                    art.outdir / "plots" / f"shap_summary_{ds2_oracle_shap_tag}.pdf"
                )
            ),
            "ds2_target_oracle_holdout_shap_global_importance": (
                None
                if ds2_oracle_reg is None
                else str(
                    art.outdir
                    / "plots"
                    / f"shap_global_importance_{ds2_oracle_shap_tag}.csv"
                )
            ),
        }

    evaluation = {
        "ds1_final_test": ds1_final_summary,
        "ds2_transfer_holdout": ds2_transfer_summary,
        "ds2_target_oracle_holdout": ds2_oracle_summary,
    }
    evaluation["ds2_target_adapted_online"] = ds2_online_summary

    forecast_holdout_evaluation = {
        "ds1_in_domain_temporal_holdout": {
            "display_name": (
                "In-domain evaluation with temporal holdout "
                "(out-of-sample forecasting)"
            ),
            "dataset_label": ds1_label,
            "forecast_model_name": str(args.forecast_model),
            "forecast_model": meta_stage.ds1_holdout_forecast_summary,
            "seasonal_naive": meta_stage.ds1_holdout_seasonal_naive_summary,
        },
        "ds2_cross_domain_temporal_holdout": {
            "display_name": "Cross-domain generalization (zero-shot transfer)",
            "source_dataset_label": ds1_label,
            "target_dataset_label": ds2_label,
            "forecast_model_name": str(args.forecast_model),
            "forecast_model": ds2_transfer_forecast_holdout_summary,
            "seasonal_naive": ds2_transfer_seasonal_naive_holdout_summary,
            "holdout_window_rows": int(len(meta2_holdout)),
            "boundary_window_rows_dropped": int(ds2_boundary),
        },
    }

    artifacts = {
        "transfer_meta_bundle_path": transfer_meta_bundle_path,
        "ds2_oracle_meta_bundle_path": ds2_oracle_bundle_path,
        **shap_paths,
    }
    return evaluation, artifacts, forecast_holdout_evaluation


def _run_single_pipeline(args):
    np.random.seed(int(args.seed))
    args.online_qs = _normalize_q_values(getattr(args, "online_qs", None))
    art = _prepare_outdirs(
        args.outdir,
        save_model_artifacts=bool(getattr(args, "save_model_artifacts", False)),
    )
    ds1_label = _dataset_label(args.ds1_data, args.ds1_group, args.ds1_csv_path)
    ds2_label = _dataset_label(args.ds2_data, args.ds2_group, args.ds2_csv_path)

    _dbg(
        "INIT",
        f"DS1={ds1_label} -> DS2={ds2_label} | "
        f"H={args.horizon} L_meta={args.meta_lags} forecast_model={args.forecast_model} "
        f"meta_model={args.meta_model} tsfel_fs={args.fs}",
    )

    forecast_stage = _run_forecast_stage(args, art)
    meta_stage = _run_meta_data_stage(args, art, forecast_stage)
    (
        reject_evaluation,
        reject_artifacts,
        forecast_holdout_evaluation,
    ) = _run_reject_evaluation_stage(
        args,
        art,
        forecast_stage,
        meta_stage,
    )

    results = {
        "config": {
            "ds1_data": args.ds1_data,
            "ds1_group": args.ds1_group,
            "ds1_csv_path": args.ds1_csv_path,
            "ds1_id_col": args.ds1_id_col,
            "ds1_ds_col": args.ds1_ds_col,
            "ds1_value_col": args.ds1_value_col,
            "ds1_freq": args.ds1_freq,
            "ds1_seasonality": args.ds1_seasonality,
            "ds2_data": args.ds2_data,
            "ds2_group": args.ds2_group,
            "ds2_csv_path": args.ds2_csv_path,
            "ds2_id_col": args.ds2_id_col,
            "ds2_ds_col": args.ds2_ds_col,
            "ds2_value_col": args.ds2_value_col,
            "ds2_freq": args.ds2_freq,
            "ds2_seasonality": args.ds2_seasonality,
            "forecast_model": str(args.forecast_model),
            "forecast_models_requested": [
                str(m) for m in getattr(args, "forecast_models", [args.forecast_model])
            ],
            "meta_model": str(args.meta_model),
            "horizon": int(args.horizon),
            "meta_lags": int(args.meta_lags),
            "tsfel_fs": int(args.fs),
            "standardize_features": bool(args.standardize_features),
            "tune_meta": bool(args.tune_meta),
            "compute_shap": bool(args.compute_shap),
            "online_qs": [float(q) for q in args.online_qs],
            "online_n_example_series": int(args.online_n_example_series),
            "save_model_artifacts": bool(args.save_model_artifacts),
            "seed": int(args.seed),
        },
        "implementation_defaults": {
            "nf_input_mult": int(DEFAULT_NF_INPUT_MULT),
            "nf_input_mult_used": int(args.input_mult),
            "nf_start_padding_enabled": bool(args.start_padding_enabled),
            "nf_batch_size": int(DEFAULT_NF_BATCH_SIZE),
            "cv_refit": bool(DEFAULT_CV_REFIT),
            "cv_val_size": int(DEFAULT_CV_VAL_SIZE),
            "cv_test_size": DEFAULT_CV_TEST_SIZE,
            "tsfel_fs_default": int(DEFAULT_TSFEL_FS),
            "tsfel_nan_col_thresh": float(DEFAULT_TSFEL_NAN_COL_THRESH),
            "meta_n_jobs": int(DEFAULT_META_N_JOBS),
            "tune_iter": int(DEFAULT_TUNE_ITER),
            "tune_cv_folds": int(DEFAULT_TUNE_CV_FOLDS),
            "ds2_step": int(
                args.horizon if DEFAULT_DS2_STEP is None else DEFAULT_DS2_STEP
            ),
            "diff_warmup": int(DEFAULT_DIFF_WARMUP),
            "ds2_holdout_frac": float(DEFAULT_DS2_HOLDOUT_FRAC),
            "ds2_holdout_min_windows": int(DEFAULT_DS2_HOLDOUT_MIN_WINDOWS),
            "random_reps": int(DEFAULT_RANDOM_REPS),
            "online_fit_windows_default": int(DEFAULT_ONLINE_FIT_WINDOWS),
            "online_calibration_windows_default": int(
                DEFAULT_ONLINE_CALIBRATION_WINDOWS
            ),
            "online_n_example_series_default": int(DEFAULT_ONLINE_N_EXAMPLE_SERIES),
            "uq_pi_level": int(DEFAULT_UQ_PI_LEVEL),
            "uq_pi_agg": str(DEFAULT_UQ_PI_AGG),
            "uq_resid_stat": str(DEFAULT_UQ_RESID_STAT),
            "uq_resid_m": int(max(5, 2 * int(args.horizon))),
            "save_model_artifacts_default": False,
        },
        "forecast_stage": {
            "ds1_seasonal_naive_cv_summary": forecast_stage.ds1_seasonal_naive_cv_summary,
            "ds1_forecast_cv_summary": forecast_stage.ds1_forecast_cv_summary,
            "fit_time_seconds": float(forecast_stage.fit_time),
            "input_size": int(forecast_stage.input_size),
            "cv_windows": (
                None
                if forecast_stage.cv_windows is None
                else int(forecast_stage.cv_windows)
            ),
            "paths": {
                "nf_cache_path": forecast_stage.nf_cache_path,
                "nf_alias_path": forecast_stage.nf_alias_path,
            },
        },
        "meta_data_stage": {
            "feature_count": int(len(meta_stage.feature_cols)),
            "baseline_score_cols": meta_stage.baseline_score_cols,
            "meta_train_all_rows": int(len(meta_stage.meta_train_all_base)),
            "meta_train_rows": int(len(meta_stage.meta_train_base)),
            "meta_test_rows": int(len(meta_stage.meta_test_base)),
            "meta2_rows": int(len(meta_stage.meta2_base)),
        },
        "forecast_holdout_evaluation": forecast_holdout_evaluation,
        "reject_evaluation": reject_evaluation,
        "artifacts": reject_artifacts,
    }

    out_json = art.outdir / (
        f"results_{meta_stage.safe_ds1}_to_{meta_stage.safe_ds2}_H{int(args.horizon)}_L{int(args.meta_lags)}_reject.json"
    )
    with open(out_json, "w") as f:
        json.dump(_json_safe(results), f, indent=2, allow_nan=False)
    _dbg("DONE", f"Saved results to {out_json}")
    return out_json


def run_pipeline(args):
    forecast_models = _normalize_forecast_models(
        getattr(args, "forecast_models", None)
        or [getattr(args, "forecast_model", "AutoKAN")]
    )

    if len(forecast_models) == 1:
        single_args = _clone_args(
            args,
            forecast_model=forecast_models[0],
            forecast_models=forecast_models,
        )
        return _run_single_pipeline(single_args)

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base_outdir": str(base_outdir),
        "forecast_models": [str(m) for m in forecast_models],
        "runs": [],
    }

    for model_name in forecast_models:
        model_outdir = base_outdir / str(model_name)
        model_args = _clone_args(
            args,
            forecast_model=str(model_name),
            forecast_models=forecast_models,
            outdir=str(model_outdir),
        )
        _dbg(
            "MULTI_MODEL",
            f"Running forecast_model={model_name} -> outdir={model_outdir}",
        )
        out_json = _run_single_pipeline(model_args)
        manifest["runs"].append(
            {
                "forecast_model": str(model_name),
                "outdir": str(model_outdir),
                "results_json": str(out_json),
            }
        )

    manifest_path = base_outdir / "multi_forecast_model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, allow_nan=False)
    _dbg("DONE", f"Saved multi-model manifest to {manifest_path}")
    return manifest_path


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            "Reject-option transfer pipeline: train a forecaster on DS1, build "
            "window-level meta-data, fit one meta-regressor that predicts error "
            "percentiles, and evaluate selective rejection on DS2."
        )
    )

    p.add_argument("--outdir", type=str, default="reject_transfer_results")

    p.add_argument("--ds1_data", type=str, default="M3")
    p.add_argument("--ds1_group", type=str, default="Monthly")
    p.add_argument("--ds1_csv_path", type=str, default=None)
    p.add_argument("--ds1_id_col", type=str, default=None)
    p.add_argument("--ds1_ds_col", type=str, default=None)
    p.add_argument("--ds1_value_col", type=str, default=None)
    p.add_argument("--ds1_freq", type=str, default=None)
    p.add_argument("--ds1_seasonality", type=int, default=None)
    p.add_argument("--ds1_long_horizon_max_points", type=int, default=None)
    p.add_argument("--ds1_long_horizon_max_points_factor", type=int, default=None)

    p.add_argument("--ds2_data", type=str, default="M1")
    p.add_argument("--ds2_group", type=str, default="Monthly")
    p.add_argument("--ds2_csv_path", type=str, default=None)
    p.add_argument("--ds2_id_col", type=str, default=None)
    p.add_argument("--ds2_ds_col", type=str, default=None)
    p.add_argument("--ds2_value_col", type=str, default=None)
    p.add_argument("--ds2_freq", type=str, default=None)
    p.add_argument("--ds2_seasonality", type=int, default=None)
    p.add_argument("--ds2_long_horizon_max_points", type=int, default=None)
    p.add_argument("--ds2_long_horizon_max_points_factor", type=int, default=None)

    p.add_argument(
        "--forecast_model",
        "--forecast_models",
        dest="forecast_models",
        type=str,
        nargs="+",
        default=["AutoKAN", "AutoNHITS"],
        choices=list(AUTO_MODELS.keys()),
        metavar="MODEL",
        help=(
            "One or more forecasting models to run. If multiple models are passed, "
            "the pipeline is executed once per model and saves each run under "
            "<outdir>/<model_name>/."
        ),
    )
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--meta_lags", type=int, default=18)
    p.add_argument("--fs", type=int, default=1)
    p.add_argument("--input_mult", type=int, default=DEFAULT_NF_INPUT_MULT)
    p.add_argument("--start_padding_enabled", action="store_true")
    p.add_argument(
        "--meta_model", type=str, choices=["catboost", "lgbm"], default="catboost"
    )

    p.add_argument("--standardize_features", action="store_true")
    p.add_argument("--tune_meta", action="store_true")
    p.add_argument("--compute_shap", action="store_true")
    p.add_argument(
        "--online_q",
        "--online_qs",
        dest="online_qs",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.4],
        metavar="Q",
        help=(
            "One or more online Meta-model reject fractions, for example "
            "--online_q 0.05 0.1 0.2, 0.3, 0.4"
        ),
    )
    p.add_argument(
        "--online_n_example_series",
        type=int,
        default=DEFAULT_ONLINE_N_EXAMPLE_SERIES,
        help="Number of example series to plot per q in the online application demo.",
    )
    p.add_argument("--save_model_artifacts", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_pipeline(args)
