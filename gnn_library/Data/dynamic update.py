class DynamicGraphUpdater:
# TODO: Implement additional functionality here
    def __init__(self, graph):
        self.graph = graph

    def add_nodes(self, nodes):
        for node in nodes:
            if node not in self.graph:
                self.graph.add_node(node)

    def add_edges(self, edges):
        for edge in edges:
            if not self.graph.has_edge(edge[0], edge[1]):
                self.graph.add_edge(edge[0], edge[1])

    def remove_nodes(self, nodes):
        for node in nodes:
            if node in self.graph:
                self.graph.remove_node(node)

    def remove_edges(self, edges):
        for edge in edges:
            if self.graph.has_edge(edge[0], edge[1]):
                self.graph.remove_edge(edge[0], edge[1])
