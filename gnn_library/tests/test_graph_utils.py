# Tests for graph utilities
import numpy as np
from graph.graph_utils import Graph

def test_adjacency_normalization():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    normalized_adj = graph.normalize_adjacency()
    assert normalized_adj.shape == (3, 3)
