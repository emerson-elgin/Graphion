# Tests for model implementations
# fix inductive learning
import numpy as np
from models.gcn import GCN
from graph.graph_utils import Graph

def test_gcn_forward():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    features = np.array([[1, 2], [3, 4], [5, 6]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    gcn = GCN(graph, features, weights)
    output = gcn.forward()
    assert output.shape == (3, 2)
