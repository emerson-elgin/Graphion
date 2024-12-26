# Graphion: Graph Neural Network Library

**Graphion** is a Python library designed for implementing Graph Neural Networks (GNNs) from scratch, without relying on external deep learning frameworks. This library is modular, extensible, and supports various GNN models such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and GraphSAGE.

---

## Features
- **Graph Representation**: Utility functions for adjacency matrix creation and normalization.
- **GNN Models**: Built-in support for GCN, GAT, and GraphSAGE.
- **Customizable Layers**: Message-passing and aggregation logic.
- **Explainability**: Visualize graphs and node interactions.

---

## Installation

## Installation

Install the library directly from PyPI:

```bash
pip install Graphion
```

Ensure you have Python 3.7+ and `numpy` installed.

---

## Example: Node Classification

Here is an example of how to use Graphion to perform node classification using a GCN model.

### 1. Create a Graph Dataset
```python
import numpy as np
from gnn_library.graph.graph_utils import Graph

# Define nodes and edges
nodes = 5
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
features = np.random.rand(nodes, 3)  # Node features
labels = np.array([0, 1, 0, 1, 0])  # Node labels

# Initialize graph
graph = Graph(nodes, edges)
```

### 2. Initialize the GCN Model
```python
from gnn_library.models.gcn import GCN

# Define GCN weight matrix
weight_matrix = np.random.rand(3, 2)

# Initialize GCN model
gcn = GCN(graph, features, weight_matrix)
```

### 3. Train the Model
```python
# Training loop
learning_rate = 0.01
for epoch in range(50):
    predictions = gcn.forward()
    loss = gcn.compute_loss(predictions, labels)
    print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    # Gradient descent
    gradients = 2 * (predictions - labels[:, None]) / len(labels)
    weight_matrix -= learning_rate * np.dot(features.T, gradients)
```

### 4. Visualize Node Interactions
```python
import matplotlib.pyplot as plt
import networkx as nx

# Visualize graph
G = nx.Graph()
G.add_edges_from(edges)

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10)
plt.title("Graph Structure")
plt.show()
```

---

## Animation: Node Feature Propagation

Graphion supports animations to visualize node feature propagation. Here's an example:

### 1. Add Animation Support
```python
import matplotlib.animation as animation

fig, ax = plt.subplots()
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', font_size=10, ax=ax)

def update(frame):
    ax.clear()
    colors = np.sin(features[:, 0] + frame / 10)  # Simulate feature propagation
    nx.draw(G, with_labels=True, node_color=colors, edge_color='gray', font_size=10, ax=ax)

ani = animation.FuncAnimation(fig, update, frames=50, repeat=True)
plt.show()
```

---

## Contributing
We welcome contributions! To add features or fix bugs:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a detailed description of your work.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or feedback, open an issue or reach out to the maintainers.
