from typing import Union
import numpy as np

def centro_gravidade(indices: np.array, valores: np.array):
    """Pega uma série numérica e retorna seu centro de gravidade"""

    return (indices*valores).sum() / valores.sum()

    