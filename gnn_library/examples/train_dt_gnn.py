# Example training script
from gnn_library.graph.graph_utils import Graph
from gnn_library.models.dt_gnn import DTGNN

nodes = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
graph = Graph(nodes, edges)

features = [[1, 0], [0, 1], [1, 1], [0, 0]]
dt_gnn = DTGNN(graph, features)
