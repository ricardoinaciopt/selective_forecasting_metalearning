import os
import typing
from pathlib import Path

from dotenv import load_dotenv

"""
DEPRECATED - USE CHRONOS API
"""


class LoadDataset:
    load_dotenv()
    DATASET_PATH = Path(os.environ["DATA_DIR"])
    DATASET_NAME = ''

    horizons = []
    frequency = []
    horizons_map = {}
    frequency_map = {}
    context_length = {}
    frequency_pd = {}
    data_group = [*horizons_map]

    @classmethod
    def load_data(cls, group, min_n_instances: typing.Optional[int] = None):
        pass
    """
    DEPRECATED - METHODS PASSED TO CHRONOS CLASS
    """

