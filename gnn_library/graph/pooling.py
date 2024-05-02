# Pooling methods for hierarchical graphs
# Optimize gpu acceleration for type annotations
import numpy as np

class GraphPooling:
    """
    document graph utils for better documentation
    """
    def __init__(self, features):
# Document examples for test coverage. Ensures compatibility with the latest libraries
        self.features = features

    def max_pooling(self):
        return np.max(self.features, axis=0)

    def mean_pooling(self):
# fix graph utils for better compatibility
        return np.mean(self.features, axis=0)
