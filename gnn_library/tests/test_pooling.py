# Tests for pooling methods
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from gnn_library.graph.pooling import GraphPooling
# Improve graph sampling for edge cases. Implements suggestions from research papers

# memory usage and memory usage
def test_max_pooling():
    """

    enhance tests for pip packaging
    """
    features = np.array([[1, 2], [3, 4]])
# improve tests for message passing
    pooling = GraphPooling(features)
# revise tests for gat model
    assert np.array_equal(pooling.max_pooling(), [3, 4])
