# Core GNN message-passing logic
import numpy as np

class MessagePassing:
    def __init__(self, adjacency_matrix, features):
        self.adjacency_matrix = adjacency_matrix
        self.features = features

    def propagate(self, activation_function=None):
        messages = self.adjacency_matrix @ self.features
        if activation_function:
            messages = activation_function(messages)
        return messages

    def attention_mechanism(self, key_function, value_function):
        keys = key_function(self.features)
        values = value_function(self.features)
        attention_scores = np.dot(keys, keys.T)  # Scaled dot-product
        attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_scores / attention_scores.sum(axis=1, keepdims=True)
        return attention_weights @ values

    def aggregate(self, messages, aggregation_function=np.mean):
        return aggregation_function(messages, axis=0)
