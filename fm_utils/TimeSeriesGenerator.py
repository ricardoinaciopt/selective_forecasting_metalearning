from metaforecast.synth.generators.tsmixup import TSMixup
from metaforecast.synth.generators.kernelsynth import KernelSynth

from metaforecast.synth.generators.mbb import SeasonalMBB
from metaforecast.synth.generators.dba import DBA
from metaforecast.synth.generators.jittering import Jittering
from metaforecast.synth.generators.scaling import Scaling
from metaforecast.synth.generators.warping_mag import MagnitudeWarping
from metaforecast.synth.generators.warping_time import TimeWarping

import pandas as pd
import inspect


class TimeSeriesGenerator:
    def __init__(
        self,
        df,
        dataset=None,
        group=None,
        seasonality=None,
        frequency=None,
        min_len=None,
        max_len=None,
    ):
        self.df = df
        self.dataset = dataset
        self.group = group
        self.seasonality = seasonality
        self.frequency = frequency
        self.min_len = min_len
        self.max_len = max_len
        self.methods = {
            "TSMixup": [
                TSMixup,
                TSMixup(max_n_uids=3, min_len=self.min_len, max_len=self.max_len),
            ],
            "KernelSynth": [
                KernelSynth,
                KernelSynth(max_kernels=5, freq=self.frequency, n_obs=self.min_len),
            ],
            "DBA": [DBA, DBA(max_n_uids=3)],
            "Scaling": [Scaling, Scaling()],
            "MagnitudeWarping": [MagnitudeWarping, MagnitudeWarping()],
            "TimeWarping": [TimeWarping, TimeWarping()],
            "SeasonalMBB": [SeasonalMBB, SeasonalMBB(seas_period=self.seasonality)],
            "Jittering": [Jittering, Jittering()],
        }

    def get_class_methods(self, cls):
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        return [
            name
            for name, func in methods
            if func.__qualname__.startswith(cls.__name__ + ".")
        ]

    def generate_synthetic_dataset(self, method_name: str, num_series: int):
        if method_name not in self.methods:
            raise ValueError(f"Unknown method_name: {method_name}")
        if not isinstance(num_series, int) or num_series < 1:
            raise ValueError("num_series must be a positive integer")

        if method_name in {"DBA", "TSMixup"}:
            self.df["unique_id"] = self.df["unique_id"].astype("str")

        method = self.methods[method_name][1]
        cls = self.methods[method_name][0]

        base_df = self.df.copy()
        augmented_dfs = []

        if "transform" in self.get_class_methods(cls):
            if method_name == "KernelSynth":
                augmented_df = method.transform(num_series)
            else:
                augmented_df = method.transform(base_df, num_series)
            if "unique_id" in augmented_df.columns:
                augmented_df["unique_id"] = (
                    augmented_df["unique_id"].astype(str) + "_SYN"
                )
            augmented_dfs.append(augmented_df.copy())
        else:
            for i in range(num_series):
                augmented_df = method._create_synthetic_ts(base_df)
                if "unique_id" in augmented_df.columns:
                    augmented_df["unique_id"] = (
                        augmented_df["unique_id"].astype(str).str.split("_").str[0]
                        + f"_SYN{i+1}"
                    )
                augmented_dfs.append(augmented_df.copy())

        return pd.concat(augmented_dfs, axis=0).reset_index(drop=True)
