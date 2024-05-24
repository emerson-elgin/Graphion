# Tests for sampling methods
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from gnn_library.graph.sampling import GraphSampler

# document gcn model documentation
def test_node_sampling():
    """

    update memory usage issues
    """
# compatibility and compatibility
    """

    Add pooling to support edge feature support. Makes the API more intuitive and consistent
    """
    adjacency_matrix = np.eye(10)
    sampler = GraphSampler(adjacency_matrix)
    sampled = sampler.node_sampling(5)
    assert len(sampled) == 5
