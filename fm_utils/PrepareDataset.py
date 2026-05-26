import inspect
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from datasetsforecast.m3 import M3, M3Info
except ImportError:
    M3 = None
    M3Info = None

try:
    from datasetsforecast.m4 import M4, M4Info
except ImportError:
    M4 = None
    M4Info = None

try:
    from datasetsforecast.long_horizon import LongHorizon
except ImportError:
    LongHorizon = None

try:
    from datasetsforecast.long_horizon2 import LongHorizon2
except ImportError:
    LongHorizon2 = None

try:
    from fm_utils.data.load_data.gluonts_data import GluontsDataset
except ImportError:
    GluontsDataset = None

try:
    from fm_utils.loaders.chronos_data import ChronosDataset
except ImportError:
    ChronosDataset = None


LONG_HORIZON_DATASET_ALIASES = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "Traffic": "Traffic",
    "TrafficL": "Traffic",
    "Electricity": "ECL",
    "ECL": "ECL",
    "Weather": "Weather",
    "ILI": "ILI",
    "Exchange": "Exchange",
}

LONG_HORIZON_DATASET_LOOKUP = {
    key.lower(): value for key, value in LONG_HORIZON_DATASET_ALIASES.items()
}

# Load long-horizon data without benchmark normalization or pre-defined splits.
LONG_HORIZON_DATASET_INFO = {
    "ETTh1": {
        "folder": "ETTh1",
        "kind": "S",
        "target_col": "OT",
        "raw_names": ["ETTh1"],
        "freq": "H",
        "seasonality": 24,
    },
    "ETTh2": {
        "folder": "ETTh2",
        "kind": "S",
        "target_col": "OT",
        "raw_names": ["ETTh2"],
        "freq": "H",
        "seasonality": 24,
    },
    "ETTm1": {
        "folder": "ETTm1",
        "kind": "M",
        "raw_names": ["ETTm1"],
        "freq": "15T",
        "seasonality": 96,
    },
    "ETTm2": {
        "folder": "ETTm2",
        "kind": "M",
        "raw_names": ["ETTm2"],
        "freq": "15T",
        "seasonality": 96,
    },
    "TrafficL": {
        "folder": "traffic",
        "kind": "M",
        "raw_names": ["Traffic", "traffic", "TrafficL"],
        "freq": "H",
        "seasonality": 24,
    },
    "ECL": {
        "folder": "ECL",
        "kind": "M",
        "raw_names": ["ECL", "Electricity", "electricity"],
        "freq": "H",
        "seasonality": 24,
    },
    "Weather": {
        "folder": "weather",
        "kind": "M",
        "raw_names": ["Weather", "weather"],
        "freq": "10T",
        "seasonality": 144,
    },
    "ILI": {
        "folder": "ili",
        "kind": "M",
        "raw_names": ["ILI", "ili"],
        "freq": "W",
        "seasonality": 52,
    },
    "Exchange": {
        "folder": "Exchange",
        "kind": "M",
        "raw_names": ["Exchange", "exchange"],
        "freq": "D",
        "seasonality": 7,
    },
}


class PrepareDataset:
    """
    Class to load and prepare datasets for forecasting.

    Attributes:
        dataset (str): Dataset identifier or a CSV path.
        group (str): Name of the dataset group to load for built-in datasets.
        directory (str): Directory where the dataset is stored.
        csv_path (str | Path | None): Optional path to a long-format CSV dataset.
        df (pd.DataFrame): DataFrame containing the dataset.
        seasonality (int): Seasonality of the dataset.
        frequency (str): Frequency of the dataset.
        standard_horizons (list[int] | None): Standard horizons for built-in datasets.
        train (pd.DataFrame): DataFrame containing the training set.
        test (pd.DataFrame): DataFrame containing the test set.

    Methods:
        load_dataset: Load the dataset.
        train_test_split: Split the dataset into training and test sets.
    """

    def __init__(
        self,
        dataset,
        group,
        directory="utils/data/assets/datasets",
        csv_path=None,
        id_col=None,
        ds_col=None,
        value_col=None,
        freq=None,
        frequency=None,
        seasonality=None,
        long_horizon_max_points=None,
        long_horizon_max_points_factor=None,
    ):
        self.directory = directory
        self.dataset = dataset
        self.group = group
        self.csv_path = csv_path
        self.id_col = id_col
        self.ds_col = ds_col
        self.value_col = value_col
        self.df = None
        self.seasonality = seasonality
        self.frequency = frequency if frequency is not None else freq
        self.long_horizon_max_points = long_horizon_max_points
        self.long_horizon_max_points_factor = long_horizon_max_points_factor
        self.standard_horizons = None
        self.train = None
        self.val = None
        self.test = None
        self.is_long_horizon_dataset = False

    def load_dataset(self, horizon=None, lags=0, drop_short_series_factor=None):
        self.is_long_horizon_dataset = False

        if self._uses_csv_loader():
            self.df = self._load_csv_dataset()
            if self.frequency is None:
                raise ValueError(
                    "CSV loading requires freq to be provided via freq=... or frequency=..."
                )
            if self.seasonality is None:
                self.seasonality = self._infer_seasonality(self.frequency)
        else:
            match self.dataset:
                # from datasetsforecast
                case "M3":
                    if M3 is None or M3Info is None:
                        raise ImportError(
                            "datasetsforecast is required to load the M3 dataset."
                        )
                    self.df, *_ = M3.load(directory=self.directory, group=self.group)
                    self.seasonality = M3Info[self.group].seasonality
                    self.frequency = M3Info[self.group].freq
                case "M4":
                    if M4 is None or M4Info is None:
                        raise ImportError(
                            "datasetsforecast is required to load the M4 dataset."
                        )
                    self.df, *_ = M4.load(directory=self.directory, group=self.group)
                    self.seasonality = M4Info[self.group].seasonality
                    self.frequency = M4Info[self.group].freq
                case _ if isinstance(self.dataset, str) and (
                    self.dataset.lower() in LONG_HORIZON_DATASET_LOOKUP
                ):
                    self.is_long_horizon_dataset = True
                    self.df = self._load_long_horizon_dataset(horizon=horizon)
                # from gluonts
                case "M1":
                    if GluontsDataset is None:
                        raise ImportError(
                            "GluontsDataset dependencies are required to load the M1 dataset."
                        )
                    match self.group:
                        case "Monthly":
                            self.df = GluontsDataset.load_data("m1_monthly")
                            self.seasonality = GluontsDataset.frequency_map[
                                "m1_monthly"
                            ]
                            self.frequency = GluontsDataset.frequency_pd["m1_monthly"]
                        case "Quarterly":
                            self.df = GluontsDataset.load_data("m1_quarterly")
                            self.seasonality = GluontsDataset.frequency_map[
                                "m1_quarterly"
                            ]
                            self.frequency = GluontsDataset.frequency_pd["m1_quarterly"]
                        case "Yearly":
                            self.df = GluontsDataset.load_data("m1_yearly")
                            self.seasonality = GluontsDataset.frequency_map["m1_yearly"]
                            self.frequency = GluontsDataset.frequency_pd["m1_yearly"]
                        case _:
                            raise Exception(
                                "Invalid group: either choose Monthly, Quarterly or Yearly"
                            )
                case "Tourism":
                    if ChronosDataset is None:
                        raise ImportError(
                            "ChronosDataset dependencies are required to load the Tourism dataset."
                        )
                    match self.group:
                        case "Monthly":
                            self.df = ChronosDataset.load_data("monash_tourism_monthly")
                            self.frequency = ChronosDataset.FREQUENCY_MAP_DATASETS[
                                "monash_tourism_monthly"
                            ]
                            self.seasonality = ChronosDataset.FREQUENCY_MAP[
                                self.frequency
                            ]
                        case "Quarterly":
                            self.df = ChronosDataset.load_data(
                                "monash_tourism_quarterly"
                            )
                            self.frequency = ChronosDataset.FREQUENCY_MAP_DATASETS[
                                "monash_tourism_quarterly"
                            ]
                            self.seasonality = ChronosDataset.FREQUENCY_MAP[
                                self.frequency
                            ]
                        case "Yearly":
                            self.df = ChronosDataset.load_data("monash_tourism_yearly")
                            self.frequency = ChronosDataset.FREQUENCY_MAP_DATASETS[
                                "monash_tourism_yearly"
                            ]
                            self.seasonality = ChronosDataset.FREQUENCY_MAP[
                                self.frequency
                            ]
                        case _:
                            raise Exception(
                                "Invalid group: either choose Monthly, Quarterly or Yearly"
                            )
                case _:
                    raise Exception(
                        "Invalid dataset: choose Tourism, M1, M3, M4, one of the "
                        "datasetsforecast long-horizon datasets, or provide a CSV path"
                    )
        # convert "ds" column to int if not a datetime
        if isinstance(self.df["ds"].iloc[0], (int, np.int32, np.int64)):
            self.df["ds"] = self.df["ds"].astype(int)

        if self.is_long_horizon_dataset:
            max_points = self._resolve_long_horizon_max_points(horizon=horizon)
            self._truncate_long_horizon_history(max_points=max_points)

        if drop_short_series_factor is not None:
            if horizon is None:
                raise ValueError(
                    "horizon must be provided when drop_short_series_factor is not None"
                )

            min_series_length = drop_short_series_factor * (horizon + lags)
            self._drop_short_series(min_series_length=min_series_length)

    def _resolve_long_horizon_group(self):
        for candidate in (self.group, self.dataset):
            if isinstance(candidate, str):
                canonical_group = LONG_HORIZON_DATASET_LOOKUP.get(candidate.lower())
                if canonical_group is not None:
                    return canonical_group

        valid_groups = ", ".join(sorted(LONG_HORIZON_DATASET_INFO))
        raise ValueError(
            f"Invalid long-horizon dataset/group combination: dataset={self.dataset!r}, "
            f"group={self.group!r}. For datasetsforecast long-horizon datasets, "
            "group must be the dataset identifier itself "
            "(for example 'ETTh1', 'ETTm1', 'Traffic', 'ECL'), not a cadence label "
            "such as 'Hourly'. "
            f"Valid groups: {valid_groups}"
        )

    def _load_long_horizon_dataset(self, horizon=None):
        group = self._resolve_long_horizon_group()
        metadata = LONG_HORIZON_DATASET_INFO[group]

        self.standard_horizons = None if horizon is None else [int(horizon)]
        self.frequency = metadata["freq"]
        self.seasonality = metadata["seasonality"]

        if LongHorizon2 is not None:
            try:
                signature = inspect.signature(LongHorizon2.load)
                load_kwargs = {
                    "directory": self.directory,
                    "group": group,
                }
                if "normalize" in signature.parameters:
                    load_kwargs["normalize"] = False
                df = LongHorizon2.load(**load_kwargs)
                if isinstance(df, tuple):
                    df = df[0]
                return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            except Exception:
                pass

        if LongHorizon is not None:
            try:
                signature = inspect.signature(LongHorizon.load)
                if "normalize" in signature.parameters:
                    df, *_ = LongHorizon.load(
                        directory=self.directory,
                        group=group,
                        normalize=False,
                    )
                    return df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            except Exception:
                pass

        return self._load_local_long_horizon_raw_dataset(
            group=group,
            metadata=metadata,
        )

    def _resolve_long_horizon_max_points(self, horizon=None):
        if self.long_horizon_max_points is not None:
            max_points = int(self.long_horizon_max_points)
        elif self.long_horizon_max_points_factor is not None:
            if horizon is None:
                raise ValueError(
                    "horizon must be provided when "
                    "long_horizon_max_points_factor is not None"
                )
            max_points = int(self.long_horizon_max_points_factor) * int(horizon)
        else:
            return None

        if max_points <= 0:
            raise ValueError("long_horizon_max_points must be a positive integer")

        if horizon is not None and max_points <= int(horizon):
            raise ValueError(
                "long_horizon_max_points must be greater than horizon so the "
                "training split keeps some history before the final test window"
            )

        return max_points

    def _truncate_long_horizon_history(self, max_points=None):
        if max_points is None:
            return

        self.df = (
            self.df.sort_values(["unique_id", "ds"])
            .groupby("unique_id", group_keys=False)
            .tail(int(max_points))
            .reset_index(drop=True)
        )

    def _iter_long_horizon_roots(self):
        directory = Path(self.directory)
        candidates = (
            directory,
            directory / "raw",
            directory / "datasets",
            directory / "longhorizon",
            directory / "longhorizon" / "raw",
            directory / "longhorizon2",
            directory / "longhorizon2" / "all_six_datasets",
            Path("."),
            Path("data"),
            Path("data") / "raw",
            Path("data") / "longhorizon",
            Path("data") / "longhorizon" / "raw",
            Path("data") / "longhorizon2",
            Path("data") / "longhorizon2" / "all_six_datasets",
        )

        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            yield candidate

    def _iter_long_horizon_raw_csv_paths(self, metadata):
        stems = []
        for candidate in (
            self.dataset,
            self.group,
            metadata["folder"],
            *metadata.get("raw_names", []),
        ):
            if isinstance(candidate, str) and candidate:
                lowered = candidate.lower()
                if lowered not in stems:
                    stems.append(lowered)

        folder = metadata["folder"]
        kind = metadata.get("kind")
        nested_candidates = []
        for root in self._iter_long_horizon_roots():
            nested_candidates.extend(
                [
                    root / folder / "Y_df.csv",
                    root / folder / "df_y.csv",
                    root / "datasets" / folder / "Y_df.csv",
                    root / "datasets" / folder / "df_y.csv",
                    root / "all_six_datasets" / folder / "Y_df.csv",
                    root / "all_six_datasets" / folder / "df_y.csv",
                ]
            )
            if kind:
                nested_candidates.extend(
                    [
                        root / folder / str(kind) / "Y_df.csv",
                        root / folder / str(kind) / "df_y.csv",
                        root / "datasets" / folder / str(kind) / "Y_df.csv",
                        root / "datasets" / folder / str(kind) / "df_y.csv",
                    ]
                )

        seen_nested = set()
        for csv_path in nested_candidates:
            key = str(csv_path)
            if key in seen_nested:
                continue
            seen_nested.add(key)
            if csv_path.exists():
                yield csv_path

        for root in self._iter_long_horizon_roots():
            try:
                csv_paths = sorted(root.glob("*.csv"))
            except OSError:
                csv_paths = []
            for csv_path in csv_paths:
                stem = csv_path.stem.lower()
                if any(
                    stem == prefix or stem.startswith(f"{prefix}_") for prefix in stems
                ):
                    yield csv_path

    def _load_local_long_horizon_raw_dataset(self, group, metadata):
        for csv_path in self._iter_long_horizon_raw_csv_paths(metadata):
            if csv_path.exists():
                return self._read_long_horizon_raw_csv(
                    csv_path=csv_path,
                    metadata=metadata,
                )

        names = ", ".join(dict.fromkeys(metadata.get("raw_names", [])))
        raise FileNotFoundError(
            "Raw long-horizon CSV not found for "
            f"group {group!r}. Expected an unnormalized CSV such as: {names or group}. "
            "If datasetsforecast is installed, this fallback is used when "
            "LongHorizon2/LongHorizon do not expose normalize=False."
        )

    @staticmethod
    def _read_long_horizon_raw_csv(csv_path, metadata):
        df = pd.read_csv(csv_path)
        lower_to_original = {col.lower(): col for col in df.columns}

        has_long_columns = {"unique_id", "ds", "y"}.issubset(lower_to_original)
        if has_long_columns:
            rename_map = {
                lower_to_original["unique_id"]: "unique_id",
                lower_to_original["ds"]: "ds",
                lower_to_original["y"]: "y",
            }
            out = df.rename(columns=rename_map)[["unique_id", "ds", "y"]].copy()
        else:
            ds_col = next(
                (
                    lower_to_original[name]
                    for name in ("ds", "date", "datetime", "timestamp")
                    if name in lower_to_original
                ),
                None,
            )
            if ds_col is None:
                raise ValueError(
                    f"Could not infer datetime column in long-horizon CSV: {csv_path}"
                )

            value_cols = [col for col in df.columns if col != ds_col]
            if metadata.get("kind") == "S":
                target_col = metadata.get("target_col")
                if target_col is not None:
                    if target_col not in df.columns:
                        raise ValueError(
                            f"Expected target column {target_col!r} in {csv_path}"
                        )
                    value_cols = [target_col]
                elif value_cols:
                    value_cols = [value_cols[-1]]

            if not value_cols:
                raise ValueError(
                    f"No value columns found in long-horizon CSV: {csv_path}"
                )

            out = (
                df[[ds_col, *value_cols]]
                .melt(
                    id_vars=[ds_col],
                    value_vars=value_cols,
                    var_name="unique_id",
                    value_name="y",
                )
                .rename(columns={ds_col: "ds"})
            )

        out["ds"] = pd.to_datetime(out["ds"], errors="raise")
        out["y"] = pd.to_numeric(out["y"], errors="raise")
        return out.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def _uses_csv_loader(self):
        if self.csv_path is not None:
            return True

        if not isinstance(self.dataset, str):
            return False

        dataset_path = Path(self.dataset)
        return dataset_path.suffix.lower() == ".csv"

    def _get_csv_path(self):
        if self.csv_path is not None:
            return Path(self.csv_path)
        return Path(self.dataset)

    def _load_csv_dataset(self):
        csv_path = self._get_csv_path()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        required_args = {
            "id_col": self.id_col,
            "ds_col": self.ds_col,
            "value_col": self.value_col,
        }
        missing_args = [name for name, value in required_args.items() if value is None]
        if missing_args:
            missing_str = ", ".join(missing_args)
            raise ValueError(
                f"Missing CSV column mapping(s): {missing_str}. "
                "Pass the source column names via id_col, ds_col and value_col."
            )

        df = pd.read_csv(csv_path)

        source_to_target = {
            self.id_col: "unique_id",
            self.ds_col: "ds",
            self.value_col: "y",
        }
        missing_columns = [
            source_col
            for source_col in source_to_target
            if source_col not in df.columns
        ]
        if missing_columns:
            missing_str = ", ".join(missing_columns)
            raise ValueError(
                f"CSV file {csv_path} is missing required columns: {missing_str}"
            )

        df = df.rename(columns=source_to_target)[["unique_id", "ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"], errors="raise")
        df["y"] = pd.to_numeric(df["y"], errors="raise")
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        return df

    @staticmethod
    def _infer_seasonality(frequency):
        if frequency is None:
            return 1

        if not isinstance(frequency, str):
            return int(frequency)

        freq_key = frequency.upper()
        freq_to_seasonality = {
            "H": 24,
            "HOURLY": 24,
            "15T": 96,
            "15MIN": 96,
            "10T": 144,
            "10MIN": 144,
            "D": 7,
            "B": 5,
            "W": 52,
            "W-SUN": 52,
            "M": 12,
            "ME": 12,
            "MS": 12,
            "Q": 4,
            "QE": 4,
            "QS": 4,
            "Y": 1,
            "YE": 1,
            "YS": 1,
        }
        return freq_to_seasonality.get(freq_key, 1)

    def _drop_short_series(self, min_series_length):
        if min_series_length is None:
            return

        series_lengths = self.df.groupby("unique_id").size()
        valid_ids = series_lengths[series_lengths >= min_series_length].index
        self.df = self.df[self.df["unique_id"].isin(valid_ids)].reset_index(drop=True)

    def train_test_split(self, horizon):
        self.test = self.df.groupby("unique_id").tail(horizon)
        self.train = self.df.drop(self.test.index).reset_index(drop=True)

    def train_val_test_split(self, horizon):
        self.test = self.df.groupby("unique_id").tail(horizon)
        remaining = self.df.drop(self.test.index)
        self.val = remaining.groupby("unique_id").tail(horizon)
        self.train = remaining.drop(self.val.index).reset_index(drop=True)
        self.val = self.val.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)
