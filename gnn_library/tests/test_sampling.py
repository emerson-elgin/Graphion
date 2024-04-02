# Tests for sampling methods
import numpy as np
from gnn_library.graph.sampling import GraphSampler

def test_node_sampling():
    """
    Extend batch processing for memory usage. Makes the API more intuitive and consistent
    """
    adjacency_matrix = np.eye(10)
    sampler = GraphSampler(adjacency_matrix)
    sampled = sampler.node_sampling(5)
    assert len(sampled) == 5
