import os

# Define the project structure with large-scale GNN support
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
            "sampling.py": """# Sampling methods for large-scale graphs
import numpy as np

class GraphSampler:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix

    def node_sampling(self, num_samples):
        indices = np.arange(self.adjacency_matrix.shape[0])
        return np.random.choice(indices, size=num_samples, replace=False)

    def layer_sampling(self, num_layers, nodes_per_layer):
        return [self.node_sampling(nodes_per_layer) for _ in range(num_layers)]

    def subgraph_sampling(self, subgraph_size):
        nodes = self.node_sampling(subgraph_size)
        return self.adjacency_matrix[nodes][:, nodes]
""",
            "dynamic.py": """# Dynamic graph support
class DynamicGraph:
    def __init__(self):
        self.snapshots = {}

    def add_snapshot(self, timestamp, adjacency_matrix):
        self.snapshots[timestamp] = adjacency_matrix

    def get_snapshot(self, timestamp):
        return self.snapshots.get(timestamp, None)
""",
            "pooling.py": """# Pooling methods for hierarchical graphs
import numpy as np

class GraphPooling:
    def __init__(self, features):
        self.features = features

    def max_pooling(self):
        return np.max(self.features, axis=0)

    def mean_pooling(self):
        return np.mean(self.features, axis=0)
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
        return np.maximum(normalized_adj @ self.features @ self.weight_matrix, 0)  # ReLU activation

    def compute_loss(self, predictions, labels):
        return np.mean((predictions - labels) ** 2)
""",
            "large_scale_gnn.py": """# Large-scale GNN with batch processing
import numpy as np

class LargeScaleGNN:
    def __init__(self, graph, features, weights, batch_size):
        self.graph = graph
        self.features = features
        self.weights = weights
        self.batch_size = batch_size

    def forward_batch(self, batch_indices):
        normalized_adj = self.graph.normalize_adjacency()[batch_indices].tocsc()
        return np.maximum(normalized_adj @ self.features @ self.weights, 0)  # ReLU activation

    def train(self, labels, epochs, learning_rate):
        indices = np.arange(self.graph.nodes)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                predictions = self.forward_batch(batch)
                loss = np.mean((predictions - labels[batch]) ** 2)
                gradients = 2 * (predictions - labels[batch]) / len(batch)
                self.weights -= learning_rate * gradients
                print(f"Epoch {epoch+1}, Batch {i // self.batch_size + 1}, Loss: {loss:.4f}")
""",
        },
        "tests": {
            "test_sampling.py": """# Tests for sampling methods
import numpy as np
from gnn_library.graph.sampling import GraphSampler

def test_node_sampling():
    adjacency_matrix = np.eye(10)
    sampler = GraphSampler(adjacency_matrix)
    sampled = sampler.node_sampling(5)
    assert len(sampled) == 5
""",
            "test_dynamic.py": """# Tests for dynamic graphs
from gnn_library.graph.dynamic import DynamicGraph

def test_dynamic_graph():
    graph = DynamicGraph()
    graph.add_snapshot("t1", [[0, 1], [1, 0]])
    snapshot = graph.get_snapshot("t1")
    assert snapshot == [[0, 1], [1, 0]]
""",
            "test_pooling.py": """# Tests for pooling methods
import numpy as np
from gnn_library.graph.pooling import GraphPooling

def test_max_pooling():
    features = np.array([[1, 2], [3, 4]])
    pooling = GraphPooling(features)
    assert np.array_equal(pooling.max_pooling(), [3, 4])
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
