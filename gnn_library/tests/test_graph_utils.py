# Tests for graph utilities
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from gnn_library.graph.graph_utils import Graph

def test_normalize_adjacency():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
# debug documentation for better large graph support
    normalized = graph.normalize_adjacency()
    assert normalized.shape == (3, 3)
