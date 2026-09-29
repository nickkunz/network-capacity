## libraries
import numpy as np
from numpy.typing import ArrayLike

## set floating-point round-off differences to exact zero before rank-based tests
def _clean_differences(diff: ArrayLike, tol: float = 1e-9) -> np.ndarray:
    values = np.array(diff, dtype = float)
    values[np.abs(values) < tol] = 0.0
    return values
