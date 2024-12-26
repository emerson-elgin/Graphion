import os

# Define the project structure with modern GNN features
project_structure = {
    "gnn_library": {
        "__init__.py": "",
        "graph": {
            "__init__.py": "",
            "graph_utils.py": """# Graph representation and utility functions
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_adjacency_matrix(nodes, edges)

    def _build_adjacency_matrix(self, nodes, edges):
        adj_matrix = np.zeros((nodes, nodes))
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1
            if len(edge) == 2 or not edge[2]:  # Undirected graph
                adj_matrix[edge[1], edge[0]] = 1
        return adj_matrix

    def normalize_adjacency(self, add_self_loops=True):
        if add_self_loops:
            self.adjacency_matrix += np.eye(len(self.adjacency_matrix))
        degrees = np.sum(self.adjacency_matrix, axis=1)
        degree_matrix = np.diag(degrees)
        return np.linalg.inv(degree_matrix) @ self.adjacency_matrix

    def compute_laplacian(self):
        degrees = np.sum(self.adjacency_matrix, axis=1)
        degree_matrix = np.diag(degrees)
        return degree_matrix - self.adjacency_matrix
""",
            "message_passing.py": """# Core GNN message-passing logic
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
"""
        },
        "models": {
            "__init__.py": "",
            "gcn.py": """# Graph Convolutional Network Implementation
import numpy as np
from graph.message_passing import MessagePassing

class GCN:
    def __init__(self, graph, features, weight_matrix):
        self.graph = graph
        self.features = features
        self.weight_matrix = weight_matrix

    def forward(self):
        normalized_adj = self.graph.normalize_adjacency()
        output = np.dot(normalized_adj, np.dot(self.features, self.weight_matrix))
        return np.maximum(output, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
""",
            "gat.py": """# Graph Attention Network Implementation
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
""",
            "graphsage.py": """# GraphSAGE Implementation
class GraphSAGE:
    def __init__(self, graph, features, aggregator):
        self.graph = graph
        self.features = features
        self.aggregator = aggregator

    def forward(self):
        aggregated_features = self.aggregator(self.graph.adjacency_matrix, self.features)
        return np.maximum(aggregated_features, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
"""
        },
        "tests": {
            "test_graph_utils.py": """# Tests for graph utilities
import numpy as np
from graph.graph_utils import Graph

def test_adjacency_normalization():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    normalized_adj = graph.normalize_adjacency()
    assert normalized_adj.shape == (3, 3)
""",
            "test_message_passing.py": """# Tests for message-passing logic
import numpy as np
from graph.message_passing import MessagePassing

def test_propagate():
    adjacency_matrix = np.array([[0, 1], [1, 0]])
    features = np.array([[1, 2], [3, 4]])
    mp = MessagePassing(adjacency_matrix, features)
    propagated = mp.propagate()
    assert propagated.shape == features.shape
""",
            "test_models.py": """# Tests for model implementations
import numpy as np
from models.gcn import GCN
from graph.graph_utils import Graph

def test_gcn_forward():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    features = np.array([[1, 2], [3, 4], [5, 6]])
    weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    gcn = GCN(graph, features, weights)
    output = gcn.forward()
    assert output.shape == (3, 2)
"""
        }
    }
}

# Function to create the project structure
def create_project_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):  # It's a folder
            os.makedirs(path, exist_ok=True)
            create_project_structure(path, content)
        else:  # It's a file
            with open(path, "w") as file:
                file.write(content)

# Create the project structure
if __name__ == "__main__":
    base_path = os.getcwd()
    create_project_structure(base_path, project_structure)
    print(f"Project structure created at {base_path}")