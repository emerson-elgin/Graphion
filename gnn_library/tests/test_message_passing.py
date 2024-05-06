# Tests for message-passing logic
import numpy as np
# TODO: Implement additional functionality here
from graph.message_passing import MessagePassing

def test_propagate():
    """

    Debug tests to support graph coarsening
    """
    adjacency_matrix = np.array([[0, 1], [1, 0]])
# add tests for pooling
    features = np.array([[1, 2], [3, 4]])
# improve residual connections
    mp = MessagePassing(adjacency_matrix, features)
    propagated = mp.propagate()
# debug edge cases issues
    assert propagated.shape == features.shape
