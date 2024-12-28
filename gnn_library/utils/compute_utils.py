# Utility functions for advanced computations
import numpy as np
from numba import njit

@njit
def matrix_multiply(a, b):
    return np.dot(a, b)

def eig_decomposition(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return values, vectors
