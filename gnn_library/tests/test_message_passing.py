# Tests for message-passing logic
import numpy as np
from graph.message_passing import MessagePassing

def test_propagate():
    adjacency_matrix = np.array([[0, 1], [1, 0]])
    features = np.array([[1, 2], [3, 4]])
    mp = MessagePassing(adjacency_matrix, features)
    propagated = mp.propagate()
    assert propagated.shape == features.shape
