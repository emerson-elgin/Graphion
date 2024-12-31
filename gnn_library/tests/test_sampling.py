# Tests for sampling methods
import numpy as np
from gnn_library.graph.sampling import GraphSampler

def test_node_sampling():
    adjacency_matrix = np.eye(10)
    sampler = GraphSampler(adjacency_matrix)
    sampled = sampler.node_sampling(5)
    assert len(sampled) == 5
