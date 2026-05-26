from utils.data.codebase.load_data.m4 import M4Dataset
from utils.data.codebase.load_data.m3 import M3Dataset

# from utils.data.codebase.load_data.longhorizon import LHDataset
from utils.data.codebase.load_data.tourism import TourismDataset

# from utils.data.codebase.load_data.gluonts import GluontsDataset

DATASETS = {
    "M4": M4Dataset,
    "M3": M3Dataset,
    # 'LH': LHDataset,
    "Tourism": TourismDataset,
    # 'Gluonts': GluontsDataset,
}
