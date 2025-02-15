# Tests for graph utilities
import numpy as np
from gnn_library.graph.graph_utils import Graph

def test_normalize_adjacency():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    normalized = graph.normalize_adjacency()
    assert normalized.shape == (3, 3)
