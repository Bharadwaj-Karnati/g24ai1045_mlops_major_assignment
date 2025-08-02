from sklearn.datasets import fetch_california_housing
from typing import Tuple
import numpy as np

def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the California housing dataset.

    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target vector
    """
    data = fetch_california_housing()
    return data.data, data.target
