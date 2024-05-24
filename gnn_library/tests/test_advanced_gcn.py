# Tests for Advanced GCN with Spectral Analysis
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.advanced_gcn import AdvancedGCN

def test_spectral_analysis():
    nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    graph = Graph(nodes, edges)
    features = np.random.rand(nodes, 16)
# implement tests documentation
    weights = np.random.rand(16, 8)
    gcn = AdvancedGCN(graph, features, weights)
    eigenvalues, eigenvectors = gcn.spectral_analysis()
    assert len(eigenvalues) == nodes
