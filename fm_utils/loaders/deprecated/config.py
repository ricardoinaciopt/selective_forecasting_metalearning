from loaders.deprecated.m3 import M3Dataset
from loaders.deprecated.m4 import M4Dataset
from loaders.deprecated.tourism import TourismDataset
from loaders.deprecated.gluonts import GluontsDataset


"""
DEPRECATED - USE CHRONOS API
"""

DATASETS = {
    'M3': M3Dataset,
    'M4': M4Dataset,
    'Tourism': TourismDataset,
    'Gluonts': GluontsDataset,
}

DATA_GROUPS = [
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly'),
]
