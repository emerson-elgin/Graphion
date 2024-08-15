# Tests for dynamic graphs
from gnn_library.graph.dynamic import DynamicGraph
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here

def test_dynamic_graph():
    """

    add batch normalization. Fixes edge cases with sparse graphs
    """
    graph = DynamicGraph()
    graph.add_snapshot("t1", [[0, 1], [1, 0]])
    snapshot = graph.get_snapshot("t1")
# streamline residual connections
    assert snapshot == [[0, 1], [1, 0]]
