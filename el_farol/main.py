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

from sim_utils import InteractiveBar, InteractiveGaussianBar
from config import PATHS


def main(ks, ds, num_rounds, num_experiments, seed=42):
    """Run the simulation for a range of memory lengths and predictors."""
    random.seed(seed)  # For reproducibility
    np.random.seed(seed) # For reproducibility

    df = InteractiveBar.run_sweep(
        memories=ds,
        predictors=ks,
        num_experiments=num_experiments,
        num_agents=[100],
        threshold=0.6,
        num_rounds=[num_rounds]
    ).reset_index()

    return df

def main_gaussian(ks, ds, sds, num_rounds, num_experiments, seed=42):
    """Run the simulation for a range of memory lengths, 
    predictors, and threshold's standard deviations."""

    random.seed(seed)  # For reproducibility
    np.random.seed(seed) # For reproducibility

    df = InteractiveGaussianBar.run_sweep(
        memories=ds,
        predictors=ks,
        std_thresholds=sds,
        seed=seed,
        num_experiments=num_experiments,
        num_agents=[100],
        threshold=0.6,
        num_rounds=[num_rounds]
    ).reset_index()

    return df