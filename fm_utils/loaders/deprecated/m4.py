import pandas as pd
from datasetsforecast.m4 import M4

from loaders.deprecated.base import LoadDataset

"""
DEPRECATED -- USE CHRONOS API
KEEPING FOR FUTURE REFERENCE
"""


class M4Dataset(LoadDataset):
    DATASET_NAME = 'M4'

    horizons_map = {
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 12,
        'Daily': 31,
    }

    frequency_map = {
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 52,
        'Daily': 365,
    }

    context_length = {
        'Quarterly': 10,
        'Monthly': 24,
        'Weekly': 12,
        'Daily': 31,
    }

    min_samples = {
        'Quarterly': 20,
        'Monthly': 48,
        'Weekly': 52,
        'Daily': 400,
    }

    frequency_pd = {
        'Quarterly': 'Q',
        'Monthly': 'M',
        'Weekly': 'W',
        'Daily': 'D',
        'Hourly': 'H',
    }

    data_group = [*horizons_map]
    horizons = [*horizons_map.values()]
    frequency = [*frequency_map.values()]

    @classmethod
    def load_data(cls, group, min_n_instances=None, extended: bool = False):

        ds_og = cls._m4_loader(group=group, min_n_instances=min_n_instances)

        if extended:
            if group == 'Daily':
                ds_h = cls._m4_loader(group='Hourly', min_n_instances=min_n_instances)
                ds_h_d = ds_h.groupby('unique_id').resample(on='ds', rule='D').mean().reset_index()

                ds = pd.concat([ds_og, ds_h_d]).reset_index(drop=True)
            elif group == 'Weekly':
                ds_h = cls._m4_loader(group='Hourly', min_n_instances=min_n_instances)
                ds_d = cls._m4_loader(group='Daily', min_n_instances=min_n_instances)
                ds_h_w = ds_h.groupby('unique_id').resample(on='ds', rule='W').mean().reset_index()
                ds_d_w = ds_d.groupby('unique_id').resample(on='ds', rule='W').mean().reset_index()

                ds = pd.concat([ds_og, ds_h_w, ds_d_w]).reset_index(drop=True)
            else:
                ds = ds_og
        else:
            ds = ds_og

        return ds

    @classmethod
    def _m4_loader(cls, group, min_n_instances=None):
        ds, *_ = M4.load(cls.DATASET_PATH, group=group)
        ds['ds'] = ds['ds'].astype(int)

        if group == 'Quarterly':
            ds = ds.query('unique_id!="Q23425"').reset_index(drop=True)

        unq_periods = ds['ds'].sort_values().unique()

        dates = pd.date_range(end='2025-03-01',
                              periods=len(unq_periods),
                              freq=cls.frequency_pd[group])

        new_ds = {k: v for k, v in zip(unq_periods, dates)}

        ds['ds'] = ds['ds'].map(new_ds)

        if min_n_instances is not None:
            ds = cls.prune_df_by_size(ds, min_n_instances)

        return ds
