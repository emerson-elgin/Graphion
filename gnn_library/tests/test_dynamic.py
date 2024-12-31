# Tests for dynamic graphs
from gnn_library.graph.dynamic import DynamicGraph

def test_dynamic_graph():
    graph = DynamicGraph()
    graph.add_snapshot("t1", [[0, 1], [1, 0]])
    snapshot = graph.get_snapshot("t1")
    assert snapshot == [[0, 1], [1, 0]]
