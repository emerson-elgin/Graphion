# Message passing logic
class MessagePassing:
    def __init__(self, adjacency_matrix, features):
        self.adjacency_matrix = adjacency_matrix
        self.features = features

    def propagate(self):
        # Simple message-passing step: Aggregate neighboring features
        return np.dot(self.adjacency_matrix, self.features)
