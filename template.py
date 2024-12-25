import os

# Define the project structure
project_structure = {
    "gnn_library": {
        "__init__.py": "",
        "graph": {
            "__init__.py": "",
            "graph_utils.py": """# Graph representation and utilities
import numpy as np

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_adjacency_matrix(nodes, edges)

    def _build_adjacency_matrix(self, nodes, edges):
        adj_matrix = np.zeros((nodes, nodes))
        for edge in edges:
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1  # For undirected graphs
        return adj_matrix
""",
            "message_passing.py": """# Message passing logic
class MessagePassing:
    def __init__(self, adjacency_matrix, features):
        self.adjacency_matrix = adjacency_matrix
        self.features = features

    def propagate(self):
        # Simple message-passing step: Aggregate neighboring features
        return np.dot(self.adjacency_matrix, self.features)
"""
        },
        "models": {
            "__init__.py": "",
            "gnn.py": """# Base GNN implementation
# Placeholder for future GNN logic
""",
            "dt_gnn.py": """# Decision Tree + GNN integration
from graph.message_passing import MessagePassing
from decision_tree.tree import DecisionTree

class DTGNN:
    def __init__(self, graph, features, max_depth=3):
        self.graph = graph
        self.features = features
        self.message_passing = MessagePassing(graph.adjacency_matrix, features)
"""
        },
        "decision_tree": {
            "__init__.py": "",
            "tree.py": """# Decision tree implementation
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
""",
            "pruning.py": """# Pruning algorithms
class Pruning:
    def __init__(self):
        pass

    def prune_tree(self, tree):
        pass
"""
        },
        "utils": {
            "__init__.py": "",
            "data_loader.py": """# Data loading and preprocessing
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        pass
""",
            "visualization.py": """# Visualization tools
class Visualization:
    def __init__(self):
        pass

    def plot_graph(self, graph):
        pass

    def plot_tree(self, tree):
        pass
"""
        },
        "examples": {
            "train_dt_gnn.py": """# Example training script
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.dt_gnn import DTGNN

nodes = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
graph = Graph(nodes, edges)

features = [[1, 0], [0, 1], [1, 1], [0, 0]]
dt_gnn = DTGNN(graph, features)
"""
        },
        "tests": {
            "test_graph_utils.py": "# Tests for graph_utils",
            "test_message_passing.py": "# Tests for message_passing",
            "test_dt_gnn.py": "# Tests for dt_gnn"
        }
    }
}

# Function to create directories and files
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
