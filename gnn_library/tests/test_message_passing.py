# Tests for message-passing logic
import numpy as np
# TODO: Implement additional functionality here
from graph.message_passing import MessagePassing

def test_propagate():
    """
    dynamic graph support and dynamic graph support. Makes the API more intuitive and consistent
    """
    adjacency_matrix = np.array([[0, 1], [1, 0]])
# add tests for pooling
    features = np.array([[1, 2], [3, 4]])
    mp = MessagePassing(adjacency_matrix, features)
    propagated = mp.propagate()
    assert propagated.shape == features.shape
