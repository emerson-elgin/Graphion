import os

# Define the project structure with modern GNN features
project_structure = {
    "gnn_library": {
        "__init__.py": "",
        "graph": {
            "__init__.py": "",
            "graph_utils.py": """# Graph representation and utility functions
import numpy as np
from scipy.sparse import csr_matrix

class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.adjacency_matrix = self._build_sparse_adjacency_matrix(nodes, edges)

    def _build_sparse_adjacency_matrix(self, nodes, edges):
        row, col = zip(*edges)
        data = np.ones(len(edges))
        return csr_matrix((data, (row, col)), shape=(nodes, nodes))

    def normalize_adjacency(self, add_self_loops=True):
        if add_self_loops:
            self.adjacency_matrix += csr_matrix(np.eye(self.nodes))
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degree_inv = csr_matrix(np.diag(1.0 / degrees))
        return degree_inv @ self.adjacency_matrix

    def compute_laplacian(self):
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degree_matrix = np.diag(degrees)
        return csr_matrix(degree_matrix) - self.adjacency_matrix
""",
        },
        "models": {
            "__init__.py": "",
            "gcn.py": """# Graph Convolutional Network Implementation
import numpy as np

class GCN:
    def __init__(self, graph, features, weight_matrix):
        self.graph = graph
        self.features = features
        self.weight_matrix = weight_matrix

    def forward(self):
        normalized_adj = self.graph.normalize_adjacency()
        output = normalized_adj @ self.features @ self.weight_matrix
        return np.maximum(output, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
""",
            "advanced_gcn.py": """# Advanced GCN with Spectral Analysis
import numpy as np
from gnn_library.utils.compute_utils import eig_decomposition

class AdvancedGCN:
    def __init__(self, graph, features, weight_matrix):
        self.graph = graph
        self.features = features
        self.weight_matrix = weight_matrix

    def spectral_analysis(self):
        laplacian = self.graph.compute_laplacian()
        eigenvalues, eigenvectors = eig_decomposition(laplacian)
        return eigenvalues, eigenvectors
""",
        },
        "utils": {
            "__init__.py": "",
            "compute_utils.py": """# Utility functions for advanced computations
import numpy as np
from numba import njit

@njit
def matrix_multiply(a, b):
    return np.dot(a, b)

def eig_decomposition(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return values, vectors
""",
            "hardware_acceleration.py": """# Utility functions for GPU-based computations
import cupy as cp

def gpu_matrix_multiply(a, b):
    a_gpu = cp.array(a)
    b_gpu = cp.array(b)
    return cp.asnumpy(cp.dot(a_gpu, b_gpu))
""",
        },
        "tests": {
            "test_graph_utils.py": """# Tests for graph utilities
import numpy as np
from gnn_library.graph.graph_utils import Graph

def test_normalize_adjacency():
    nodes = 3
    edges = [(0, 1), (1, 2)]
    graph = Graph(nodes, edges)
    normalized = graph.normalize_adjacency()
    assert normalized.shape == (3, 3)
""",
            "test_gcn.py": """# Tests for GCN
import numpy as np
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.gcn import GCN

def test_gcn():
    nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    graph = Graph(nodes, edges)
    features = np.random.rand(nodes, 16)
    weights = np.random.rand(16, 8)
    gcn = GCN(graph, features, weights)
    predictions = gcn.forward()
    assert predictions.shape == (nodes, 8)
""",
            "test_advanced_gcn.py": """# Tests for Advanced GCN with Spectral Analysis
import numpy as np
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.advanced_gcn import AdvancedGCN

def test_spectral_analysis():
    nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    graph = Graph(nodes, edges)
    features = np.random.rand(nodes, 16)
    weights = np.random.rand(16, 8)
    gcn = AdvancedGCN(graph, features, weights)
    eigenvalues, eigenvectors = gcn.spectral_analysis()
    assert len(eigenvalues) == nodes
""",
        },
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
    print(f"Updated project structure created at {base_path}")
