# Sampling methods for large-scale graphs
import numpy as np

class GraphSampler:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def node_sampling(self, num_samples):
        indices = np.arange(self.adjacency_matrix.shape[0])
        return np.random.choice(indices, size=num_samples, replace=False)

    def layer_sampling(self, num_layers, nodes_per_layer):
        return [self.node_sampling(nodes_per_layer) for _ in range(num_layers)]

    def subgraph_sampling(self, subgraph_size):
        nodes = self.node_sampling(subgraph_size)
        return self.adjacency_matrix[nodes][:, nodes]
