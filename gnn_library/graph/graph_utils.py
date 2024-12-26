# Graph representation and utility functions
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_adjacency_matrix(nodes, edges)

    def _build_adjacency_matrix(self, nodes, edges):
        adj_matrix = np.zeros((nodes, nodes))
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1
            if len(edge) == 2 or not edge[2]:  # Undirected graph
                adj_matrix[edge[1], edge[0]] = 1
        return adj_matrix

    def normalize_adjacency(self, add_self_loops=True):
        if add_self_loops:
            self.adjacency_matrix += np.eye(len(self.adjacency_matrix))
        degrees = np.sum(self.adjacency_matrix, axis=1)
        degree_matrix = np.diag(degrees)
        return np.linalg.inv(degree_matrix) @ self.adjacency_matrix

    def compute_laplacian(self):
        degrees = np.sum(self.adjacency_matrix, axis=1)
        degree_matrix = np.diag(degrees)
        return degree_matrix - self.adjacency_matrix
