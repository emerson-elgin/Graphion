# GraphSAGE Implementation
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
