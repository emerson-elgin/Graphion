# Advanced GCN with Spectral Analysis
import numpy as np
from gnn_library.utils.compute_utils import eig_decomposition

class AdvancedGCN:
    def __init__(self, graph, features, weight_matrix):
        self.graph = graph
        self.features = features
        self.weight_matrix = weight_matrix

    def spectral_analysis(self):
        laplacian = self.graph.compute_laplacian()
        eigenvalues, eigenvectors = eig_decomposition(laplacian)
        return eigenvalues, eigenvectors
