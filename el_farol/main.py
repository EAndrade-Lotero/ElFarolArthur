import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm.auto import tqdm

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../el_farol')

from sim_utils import InteractiveBar
from data_utils import DataUtils
from config import PATHS


def main(ks, ds, num_rounds, num_experiments):

    random.seed(42)  # For reproducibility

    df = InteractiveBar.run_sweep(
        memories=ds,
        predictors=ks,
        num_experiments=num_experiments,
        num_agents=[100],
        threshold=0.6,
        num_rounds=[num_rounds]
    ).reset_index()

    return df