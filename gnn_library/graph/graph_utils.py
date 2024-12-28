# Graph representation and utility functions
import numpy as np
from scipy.sparse import csr_matrix

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_sparse_adjacency_matrix(nodes, edges)

    def _build_sparse_adjacency_matrix(self, nodes, edges):
        row, col = zip(*edges)
        data = np.ones(len(edges))
        return csr_matrix((data, (row, col)), shape=(nodes, nodes))

    def normalize_adjacency(self, add_self_loops=True):
        if add_self_loops:
            self.adjacency_matrix += csr_matrix(np.eye(self.nodes))
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degree_inv = csr_matrix(np.diag(1.0 / degrees))
        return degree_inv @ self.adjacency_matrix

    def compute_laplacian(self):
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degree_matrix = np.diag(degrees)
        return csr_matrix(degree_matrix) - self.adjacency_matrix
