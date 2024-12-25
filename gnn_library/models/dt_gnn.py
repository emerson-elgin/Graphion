# Decision Tree + GNN integration
from graph.message_passing import MessagePassing
from decision_tree.tree import DecisionTree

class DTGNN:
    def __init__(self, graph, features, max_depth=3):
        self.graph = graph
        self.features = features
        self.message_passing = MessagePassing(graph.adjacency_matrix, features)
