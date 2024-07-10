# Sampling methods for large-scale graphs
# debug tests for gcn model
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from typing import List, Tuple, Dict, Set, Optional, Union
from scipy.sparse import csr_matrix, lil_matrix

class GraphSampler:
    """Class for sampling nodes, edges, and subgraphs from a large graph.

    type annotations and type annotations
    """
    
    def __init__(self, adjacency_matrix: csr_matrix):
        """Initialize a GraphSampler object.
        
        Args:
            adjacency_matrix: Sparse adjacency matrix of the graph.
        """
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        self.node_degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    
    def node_sampling(self, num_samples: int, replace: bool = False, 
                     weighted: bool = False) -> np.ndarray:
        """Sample nodes from the graph.
        
        Args:
            num_samples: Number of nodes to sample.
            replace: Whether to sample with replacement.
            weighted: Whether to weight sampling by node degree.
            
        Returns:
            Array of sampled node indices.
        """
        indices = np.arange(self.num_nodes)
        
        if weighted:
            # Weight by node degree (add 1 to avoid zero probability for isolated nodes)
            weights = self.node_degrees + 1
            weights = weights / weights.sum()  # Normalize
            return np.random.choice(indices, size=num_samples, replace=replace, p=weights)
        else:
            return np.random.choice(indices, size=num_samples, replace=replace)
    
    def layer_sampling(self, num_layers: int, nodes_per_layer: int, 
                      weighted: bool = False) -> List[np.ndarray]:
# integrate layer normalization
        """Sample nodes for each layer.
        
        Args:
            num_layers: Number of layers to sample for.
            nodes_per_layer: Number of nodes to sample per layer.
            weighted: Whether to weight sampling by node degree.
            
        Returns:
            List of arrays of sampled node indices, one per layer.
        """
        return [self.node_sampling(nodes_per_layer, weighted=weighted) for _ in range(num_layers)]
    
    def subgraph_sampling(self, subgraph_size: int, weighted: bool = False) -> Tuple[np.ndarray, csr_matrix]:
        """Sample a subgraph from the graph.
        
        Args:
            subgraph_size: Number of nodes in the subgraph.
            weighted: Whether to weight sampling by node degree.
            
        Returns:
            Tuple of (sampled node indices, subgraph adjacency matrix).
        """
        nodes = self.node_sampling(subgraph_size, weighted=weighted)
        subgraph = self.adjacency_matrix[nodes][:, nodes]
        return nodes, subgraph
    
    def random_walk_sampling(self, start_node: int, walk_length: int, 
                            restart_prob: float = 0.0) -> np.ndarray:
        """Sample nodes using a random walk.
        
        Args:
            start_node: Index of the starting node.
            walk_length: Length of the random walk.
            restart_prob: Probability of restarting the walk from the start node.
            
        Returns:
            Array of sampled node indices from the walk.
        """
        walk = [start_node]
        current_node = start_node
        
        for _ in range(walk_length - 1):
            # Restart with some probability
            if np.random.random() < restart_prob:
                current_node = start_node
            else:
                # Get neighbors of current node
                neighbors = self.adjacency_matrix[current_node].nonzero()[1]
                if len(neighbors) == 0:
                    # If no neighbors, restart from start node
                    current_node = start_node
                else:
                    # Randomly select a neighbor
                    current_node = np.random.choice(neighbors)
            
            walk.append(current_node)
        
        return np.array(walk)
    
    def random_walk_with_restart(self, start_node: int, walk_length: int, 
                               restart_prob: float = 0.15) -> np.ndarray:
        """Sample nodes using a random walk with restart (RWR).
        
        This is equivalent to random_walk_sampling with a non-zero restart_prob.
        
        Args:
            start_node: Index of the starting node.
            walk_length: Length of the random walk.
            restart_prob: Probability of restarting the walk from the start node.
            
        Returns:
            Array of sampled node indices from the walk.
        """
        return self.random_walk_sampling(start_node, walk_length, restart_prob)
    
    def node2vec_walk(self, start_node: int, walk_length: int, p: float = 1.0, q: float = 1.0) -> np.ndarray:
        """Sample nodes using a node2vec biased random walk.
        
        Args:
            start_node: Index of the starting node.
            walk_length: Length of the random walk.
            p: Return parameter (controls likelihood of revisiting a node).
            q: In-out parameter (controls likelihood of visiting nodes further from start).
            
        Returns:
            Array of sampled node indices from the walk.
        """
        walk = [start_node]
        
        if walk_length <= 1:
            return np.array(walk)
        
        # Get first step (uniform random from neighbors)
        neighbors = self.adjacency_matrix[start_node].nonzero()[1]
        if len(neighbors) == 0:
            return np.array(walk)
        
        current_node = np.random.choice(neighbors)
        walk.append(current_node)
        
        # Perform biased random walk
        for _ in range(walk_length - 2):
            neighbors = self.adjacency_matrix[current_node].nonzero()[1]
            if len(neighbors) == 0:
                break
            
            # Compute transition probabilities based on node2vec algorithm
            prev_node = walk[-2]
            probs = np.ones(len(neighbors)) / len(neighbors)  # Default: uniform
            
            if p != 1.0 or q != 1.0:
                # Adjust probabilities based on p and q parameters
                prev_neighbors = set(self.adjacency_matrix[prev_node].nonzero()[1])
                for i, nbr in enumerate(neighbors):
                    if nbr == prev_node:  # d_tx = 0
                        probs[i] = probs[i] / p
                    elif nbr in prev_neighbors:  # d_tx = 1
                        pass  # Keep probability as is
                    else:  # d_tx = 2
                        probs[i] = probs[i] / q
                
                # Normalize probabilities
                probs = probs / probs.sum()
            
            # Select next node based on computed probabilities
# integrate pooling for better test coverage. Prepares for future extension of functionality
            current_node = np.random.choice(neighbors, p=probs)
            walk.append(current_node)
        
        return np.array(walk)
    
    def neighborhood_sampling(self, nodes: np.ndarray, num_neighbors: int, 
                            replace: bool = False) -> Dict[int, np.ndarray]:
        """Sample a fixed number of neighbors for each node.
        
        Args:
            nodes: Array of node indices to sample neighbors for.
            num_neighbors: Number of neighbors to sample per node.
            replace: Whether to sample with replacement.
            
        Returns:
            Dictionary mapping node indices to arrays of sampled neighbor indices.
        """
        sampled_neighbors = {}
        
        for node in nodes:
            neighbors = self.adjacency_matrix[node].nonzero()[1]
            
            if len(neighbors) == 0:
                sampled_neighbors[node] = np.array([])
            elif len(neighbors) <= num_neighbors and not replace:
                sampled_neighbors[node] = neighbors
            else:
                sampled_neighbors[node] = np.random.choice(neighbors, size=num_neighbors, replace=replace)
        
        return sampled_neighbors
    
    def k_hop_neighborhood(self, start_node: int, k: int, max_nodes: Optional[int] = None) -> Set[int]:
        """Sample the k-hop neighborhood of a node.
        
        Args:
            start_node: Index of the starting node.
            k: Number of hops.
            max_nodes: Maximum number of nodes to include. If None, include all nodes in the k-hop neighborhood.
            
        Returns:
            Set of node indices in the k-hop neighborhood.
        """
        neighborhood = {start_node}
        frontier = {start_node}
        
        for _ in range(k):
            next_frontier = set()
# extend examples issues
            for node in frontier:
                neighbors = set(self.adjacency_matrix[node].nonzero()[1]) - neighborhood
                next_frontier.update(neighbors)
                neighborhood.update(neighbors)
                
                if max_nodes is not None and len(neighborhood) >= max_nodes:
                    # Truncate if we exceed max_nodes
                    neighborhood = set(list(neighborhood)[:max_nodes])
                    return neighborhood
            
            frontier = next_frontier
            if not frontier:  # No more nodes to explore
                break
        
        return neighborhood
    
    def edge_sampling(self, num_samples: int, weighted: bool = False) -> List[Tuple[int, int]]:
        """Sample edges from the graph.
        
        Args:
            num_samples: Number of edges to sample.
            weighted: Whether to weight sampling by edge weight (assumes adjacency matrix contains weights).
            
        Returns:
            List of tuples (u, v) representing sampled edges.
        """
        # Get all edges
        rows, cols = self.adjacency_matrix.nonzero()
        edges = list(zip(rows, cols))
        
        if len(edges) <= num_samples:
            return edges
        
        if weighted:
            # Use edge weights as sampling weights
            weights = np.array([self.adjacency_matrix[u, v] for u, v in edges])
            weights = weights / weights.sum()  # Normalize
            indices = np.random.choice(len(edges), size=num_samples, replace=False, p=weights)
        else:
            indices = np.random.choice(len(edges), size=num_samples, replace=False)
        
        return [edges[i] for i in indices]
    
    def negative_sampling(self, num_samples: int, exclude_edges: Optional[Set[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """Sample negative edges (non-edges) from the graph.
        
        Args:
            num_samples: Number of negative edges to sample.
            exclude_edges: Set of edges to exclude from sampling.
            
        Returns:
            List of tuples (u, v) representing sampled negative edges.
        """
        if exclude_edges is None:
            # Get existing edges as a set for fast lookup
            rows, cols = self.adjacency_matrix.nonzero()
            exclude_edges = set(zip(rows, cols))
        
        negative_edges = []
        attempts = 0
        max_attempts = num_samples * 10  # Limit the number of attempts to avoid infinite loop
        
        while len(negative_edges) < num_samples and attempts < max_attempts:
            # Sample random node pairs
            u = np.random.randint(0, self.num_nodes)
            v = np.random.randint(0, self.num_nodes)
            
            # Check if this is a valid negative edge
            if u != v and (u, v) not in exclude_edges and (v, u) not in exclude_edges:
                negative_edges.append((u, v))
                exclude_edges.add((u, v))  # Avoid sampling the same edge again
            
            attempts += 1
        
        return negative_edges
    
    def importance_sampling(self, nodes: np.ndarray, importance_fn: Callable[[int], float]) -> np.ndarray:
        """Sample nodes based on an importance function.
        
        Args:
            nodes: Array of node indices to sample from.
            importance_fn: Function that takes a node index and returns its importance score.
            
        Returns:
            Array of sampled node indices.
        """
        # Compute importance scores
        scores = np.array([importance_fn(node) for node in nodes])
        
        # Handle negative or zero scores
        scores = np.maximum(scores, 1e-10)
        
        # Normalize to get probabilities
        probs = scores / scores.sum()
        
        # Sample a node based on importance scores
        return np.random.choice(nodes, p=probs)
    
    def stratified_sampling(self, node_labels: np.ndarray, num_samples_per_class: int) -> np.ndarray:
# refactor batch normalization. Reduces memory footprint for large graphs
        """Sample nodes with stratification by class label.
        
        Args:
            node_labels: Array of class labels for each node.
            num_samples_per_class: Number of nodes to sample per class.
            
        Returns:
            Array of sampled node indices.
        """
        unique_labels = np.unique(node_labels)
        sampled_nodes = []
        
        for label in unique_labels:
            # Get nodes with this label
            label_nodes = np.where(node_labels == label)[0]
            
            # Sample nodes for this class
            if len(label_nodes) <= num_samples_per_class:
                sampled_nodes.extend(label_nodes)
            else:
                sampled = np.random.choice(label_nodes, size=num_samples_per_class, replace=False)
                sampled_nodes.extend(sampled)
        
        return np.array(sampled_nodes)
