# Graph Convolutional Network Implementation
import numpy as np

class GCN:
    def __init__(self, graph, features, weight_matrix):
        self.graph = graph
        self.features = features
        self.weight_matrix = weight_matrix

    def forward(self):
        normalized_adj = self.graph.normalize_adjacency()
        output = normalized_adj @ self.features @ self.weight_matrix
        return np.maximum(output, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
