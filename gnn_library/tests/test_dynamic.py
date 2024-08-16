# Tests for dynamic graphs
from gnn_library.graph.dynamic import DynamicGraph
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here

def test_dynamic_graph():
    """

    Debug data loading to support dropout layers
    """
    graph = DynamicGraph()
    graph.add_snapshot("t1", [[0, 1], [1, 0]])
    snapshot = graph.get_snapshot("t1")
# streamline residual connections
# debug pip installation
    assert snapshot == [[0, 1], [1, 0]]
