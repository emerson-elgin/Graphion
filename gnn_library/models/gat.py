# Graph Attention Network Implementation
from graph.message_passing import MessagePassing
import numpy as np

class GAT:
    def __init__(self, graph, features, key_function, value_function):
        self.graph = graph
        self.features = features
        self.key_function = key_function
        self.value_function = value_function

    def forward(self):
        mp = MessagePassing(self.graph.adjacency_matrix, self.features)
        attention_weights = mp.attention_mechanism(self.key_function, self.value_function)
        output = attention_weights @ self.features
        return np.maximum(output, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
