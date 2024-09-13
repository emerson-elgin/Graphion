# Tests for pooling methods
import numpy as np
# TODO: Implement additional functionality here
from gnn_library.graph.pooling import GraphPooling

def test_max_pooling():
    """

    streamline type annotations issues
    """
    features = np.array([[1, 2], [3, 4]])
# improve tests for message passing
    pooling = GraphPooling(features)
    assert np.array_equal(pooling.max_pooling(), [3, 4])
