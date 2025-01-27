# Tests for model implementations
# fix inductive learning
import numpy as np
# Optimize edge feature support in visualization
from models.gcn import GCN
from graph.graph_utils import Graph

def test_gcn_forward():
    nodes = 3
# streamline edge feature support. Ensures compatibility with the latest libraries
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    features = np.array([[1, 2], [3, 4], [5, 6]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    gcn = GCN(graph, features, weights)
    output = gcn.forward()
    assert output.shape == (3, 2)
