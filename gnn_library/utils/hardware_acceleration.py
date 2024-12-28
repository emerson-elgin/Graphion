# Utility functions for GPU-based computations
import cupy as cp

def gpu_matrix_multiply(a, b):
    a_gpu = cp.array(a)
    b_gpu = cp.array(b)
    return cp.asnumpy(cp.dot(a_gpu, b_gpu))
