# Graph representation and utilities
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_adjacency_matrix(nodes, edges)

    def _build_adjacency_matrix(self, nodes, edges):
        adj_matrix = np.zeros((nodes, nodes))
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1  # For undirected graphs
        return adj_matrix
