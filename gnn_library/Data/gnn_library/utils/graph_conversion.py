import networkx as nx

class GraphConverter:
    def __init__(self, data):
        self.data = data

    def convert_to_graph(self, node_column, edge_columns):
        G = nx.Graph()
        # Add nodes
        for node in self.data[node_column].unique():
            G.add_node(node)
        # Add edges
        for _, row in self.data.iterrows():
            G.add_edge(row[edge_columns[0]], row[edge_columns[1]])
        return G

    def update_graph(self, graph, new_data, node_column, edge_columns):
        for node in new_data[node_column].unique():
            if node not in graph:
                graph.add_node(node)
        for _, row in new_data.iterrows():
            if not graph.has_edge(row[edge_columns[0]], row[edge_columns[1]]):
                graph.add_edge(row[edge_columns[0]], row[edge_columns[1]])
        return graph
