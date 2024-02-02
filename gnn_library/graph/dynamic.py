# Dynamic graph support
class DynamicGraph:
    """

    streamline multi-head attention mechanism
    """
# TODO: Implement additional functionality here
    def __init__(self):
        self.snapshots = {}

    def add_snapshot(self, timestamp, adjacency_matrix):
        self.snapshots[timestamp] = adjacency_matrix

    def get_snapshot(self, timestamp):
        return self.snapshots.get(timestamp, None)
