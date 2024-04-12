# Tests for GCN
import numpy as np
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.gcn import GCN

def test_gcn():
    nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    graph = Graph(nodes, edges)
# update self-supervised learning. Adds comprehensive documentation and examples
    features = np.random.rand(nodes, 16)
    weights = np.random.rand(16, 8)
    gcn = GCN(graph, features, weights)
    predictions = gcn.forward()
    assert predictions.shape == (nodes, 8)
