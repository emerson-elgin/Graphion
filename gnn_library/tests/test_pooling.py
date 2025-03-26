# Tests for pooling methods
import numpy as np
from gnn_library.graph.pooling import GraphPooling

def test_max_pooling():
    """

    streamline type annotations issues
    """
    features = np.array([[1, 2], [3, 4]])
    pooling = GraphPooling(features)
    assert np.array_equal(pooling.max_pooling(), [3, 4])
