from math import e
from typing import Optional

import datasets
import numpy as np
import pandas as pd

from huggingface_hub import dataset_info


class ChronosDataset:
    # https://github.com/autogluon/fev/blob/main/benchmarks/chronos_zeroshot/results/auto_arima.csv
    # https://github.com/SalesforceAIResearch/gift-eval/blob/main/results/naive/all_results.csv

    DATASET_NAME = "CHRONOS"
    REPO_ID = "autogluon/chronos_datasets"

    M4_HORIZON_MAP = {
        "Y": 6,
        "Q": 8,
        "M": 18,
        "MS": 18,
        "ME": 18,
        "W": 13,
        "D": 14,
        "H": 48,
    }

    HORIZON_MAP = {
        "Y": 6,
        "Q": 8,
        "M": 12,
        "MS": 12,
        "ME": 12,
        "W": 8,
        "D": 30,
        "H": 48,
        "T": 48,
        "S": 60,
    }

    FREQUENCY_MAP = {
        "Y": 1,
        "Q": 4,
        "M": 12,
        "MS": 12,
        "ME": 12,
        "W": 52,  # ?
        "D": 365,  # 7?
        "H": 24,
        "T": 1,  # ?
        "S": 1,  # ?
    }

    # LAGS_BY_FREQUENCY = {k: int(v * 1.25) for k, v in HORIZON_MAP.items()}

    SPECIAL_HORIZON_MAP = {
        "monash_m1_monthly": 6,  # time series are too short for 12 or 18
    }

    FREQUENCY_MAP_DATASETS = {
        "monash_m1_monthly": "M",
        "monash_m1_quarterly": "Q",
        "monash_m1_yearly": "Y",
        "monash_m3_monthly": "M",
        "monash_m3_quarterly": "Q",
        "monash_m3_yearly": "Y",
        "monash_tourism_monthly": "M",
        "monash_tourism_quarterly": "Q",
        "monash_tourism_yearly": "Y",
        "m4_hourly": "H",
        "m4_monthly": "M",
        "m4_quarterly": "Q",
        "m4_weekly": "W",
        "m4_daily": "D",
        "m4_yearly": "Y",
        "m5": "D",
        "m5-RESAMPLE-MS-sum": "MS",
        "monash_hospital": "MS",
        "monash_car_parts": "MS",
    }

    @classmethod
    def load_data(
        cls,
        group: str,
        split: str = "train",
        min_n_instances: Optional[int] = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        resample_to: Optional[str] = None,
        resample_stat: str = "sum",
    ):

        assert group in [*cls.FREQUENCY_MAP_DATASETS], "Unknown dataset"

        ds = datasets.load_dataset(path=cls.REPO_ID, name=group, split=split)
        ds.set_format("numpy")

        df = cls.to_pandas(ds).reset_index(drop=True)
        df = df.rename(
            columns={"id": id_col, "timestamp": time_col, "target": target_col}
        )

        if "category" in df.columns:
            df = df.drop(columns=["category"])

        if min_n_instances is not None:
            df = cls.prune_uids_by_size(df, min_n_instances)

        # todo this was done for yearly time series (m4_yearly) ... format may not be appropriate for others
        if df[time_col].dtype == "O":
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except pd.errors.OutOfBoundsDatetime:
                df[time_col] = pd.to_datetime(
                    df[time_col], errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f"
                )
                if df[time_col].isna().any():
                    df[time_col] = df[time_col].astype(str).str.slice(0, 10)
                    df["ds"] = pd.to_datetime(
                        df[time_col], errors="coerce", format="%Y-%m-%d"
                    )

        if resample_to is not None:
            df = cls.resample_df(df, resample_to, time_col, resample_stat)

        return df

    @classmethod
    def load_everything(
        cls,
        group: str,
        split: str = "train",
        min_n_instances: Optional[int] = None,
        sample_n_uid: Optional[int] = None,
    ):

        # if split == 'both':
        #     df_train = cls.load_data(group=group, split='train', min_n_instances=min_n_instances)
        #     df_test = cls.load_data(group=group, split='test', min_n_instances=min_n_instances)
        #     df = pd.concat([df_train, df_test], axis=0).sort_values([id_col, time_col]).reset_index(drop=True)

        group_name, resample_to, resample_stat = cls.resample_info_from_group(group)

        df = cls.load_data(
            group=group_name,
            split=split,
            min_n_instances=min_n_instances,
            resample_to=resample_to,
            resample_stat=resample_stat,
        )

        freq = cls.FREQUENCY_MAP_DATASETS.get(group)

        if group in [*cls.SPECIAL_HORIZON_MAP]:
            horizon = cls.SPECIAL_HORIZON_MAP[group]
        else:
            if group.startswith("m4"):
                horizon = cls.M4_HORIZON_MAP.get(freq)
            else:
                horizon = cls.HORIZON_MAP.get(freq)

        n_lags = int(horizon * 1.25)

        seas_len = cls.FREQUENCY_MAP.get(freq)

        if sample_n_uid is not None:
            assert isinstance(df, pd.DataFrame)
            df = cls.sample_first_uids(df, sample_n_uid)

        return df, horizon, n_lags, freq, seas_len

    @staticmethod
    def resample_df(
        df: pd.DataFrame, resample_to: str, time_col: str, resample_stat: str
    ):
        return df.resample(resample_to, on=time_col).agg(resample_stat)

    @staticmethod
    def resample_info_from_group(group: str):
        # 'm5-RESAMPLE-MS-sum'

        if "RESAMPLE" not in group:
            return group, None, None

        group_name = group.split("-RESAMPLE-")[0]

        rs_info = group.split("-RESAMPLE-")[1].split("-")

        resample_to = rs_info[0]
        resample_stat = rs_info[1]

        return group_name, resample_to, resample_stat

    @staticmethod
    def prune_uids_by_size(
        df: pd.DataFrame, min_n_instances: int, id_col: str = "unique_id"
    ):
        large_ts = df[id_col].value_counts() >= min_n_instances
        large_ts_uid = large_ts[large_ts].index.tolist()

        df = df.query(f"{id_col}== @large_ts_uid").reset_index(drop=True)

        return df

    @staticmethod
    def sample_first_uids(df: pd.DataFrame, n_uid: int, id_col: str = "unique_id"):
        uid_sample = df[id_col].unique()[:n_uid].tolist()
        df = df.query(f"{id_col}==@uid_sample").reset_index(drop=True)

        return df

    @staticmethod
    def dummify_series(df, id_col: str = "unique_id", target_col: str = "y"):
        df_uid = df.copy().groupby(id_col)

        dummied_l = []
        for g, uid_df in df_uid:
            uid_df[target_col] = range(uid_df.shape[0])

            dummied_l.append(uid_df)

        dummy_df = pd.concat(dummied_l, axis=0).reset_index(drop=True)

        return dummy_df

    @staticmethod
    def get_uid_tails(df, tail_size: int, id_col: str = "unique_id"):
        df_list = []
        for g, df_ in df.groupby(id_col):
            df_list.append(df_.tail(tail_size))

        tail_df = pd.concat(df_list, axis=0).reset_index(drop=True)

        return tail_df

    @staticmethod
    def time_wise_split(
        df: pd.DataFrame, horizon: int, id_col: str = "unique_id", time_col: str = "ds"
    ):
        df_by_unq = df.groupby(id_col)

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values(time_col)

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df

    @staticmethod
    def difference_series(df):
        df_uid = df.copy().groupby("unique_id")

        diff_l = []
        for g, uid_df in df_uid:
            uid_df["y"] = uid_df["y"].diff()

            diff_l.append(uid_df.tail(-1))

        diff_df = pd.concat(diff_l, axis=0).reset_index(drop=True)

        return diff_df

    @staticmethod
    def get_chronos_datasets_names(repo_id="autogluon/chronos_datasets"):
        ds_info = dataset_info(repo_id)

        dataset_names = np.unique([s.rfilename.split("/")[0] for s in ds_info.siblings])
        dataset_names = [
            x for x in dataset_names if x not in [".gitattributes", "README.md"]
        ]

        return dataset_names

    @staticmethod
    def to_pandas(ds: datasets.Dataset) -> "pd.DataFrame":
        """Convert dataset to long data frame format."""
        sequence_columns = [
            col
            for col in ds.features
            if isinstance(ds.features[col], datasets.Sequence)
        ]
        return ds.to_pandas().explode(sequence_columns).infer_objects()
