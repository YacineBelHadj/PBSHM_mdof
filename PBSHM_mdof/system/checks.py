import numpy as np
from typing import Tuple

def is_diagonal(matrix: np.ndarray) -> bool:
    """
    Returns True if the matrix is diagonal, False otherwise.
    """
    # Check if matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        return False
    
    # Check if all off-diagonal elements are 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and np.abs(matrix[i, j]) > 1e-6:
                return False
    return True


def check_rank_matrix(system: Tuple[np.ndarray]):
    """
    Checks if the matrix M is full rank.
    """
    for matrix in system:
        return (np.linalg.matrix_rank(matrix) == matrix.shape[0])