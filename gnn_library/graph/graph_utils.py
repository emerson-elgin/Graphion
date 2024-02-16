# Graph representation and utility functions
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from scipy.sparse import csr_matrix, eye, diags, linalg
from scipy.sparse.csgraph import connected_components
import warnings

class Graph:
    """A class representing a graph with nodes and edges.
    
    This class provides methods for building and manipulating graph structures,
    including adjacency matrix operations, normalization, and spectral analysis.
    
    Attributes:
        nodes (int): Number of nodes in the graph.
        node_features (numpy.ndarray): Features associated with nodes.
        edge_features (dict): Features associated with edges.
        adjacency_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix representation.
        edge_index (tuple): Tuple of (row, col) arrays representing edges.
        edge_weight (numpy.ndarray): Weights associated with edges.
    """
    
    def __init__(self, nodes, edges, node_features=None, edge_features=None, directed=False):
        """Initialize a Graph object.
        
        Args:
            nodes (int): Number of nodes in the graph.
            edges (list): List of tuples (u, v) representing edges from node u to node v.
            node_features (numpy.ndarray, optional): Features for each node. Default is None.
            edge_features (dict, optional): Dictionary mapping edge tuples to feature vectors. Default is None.
            directed (bool, optional): Whether the graph is directed. Default is False.
        """
        self.nodes = nodes
        self.directed = directed
        self.node_features = node_features if node_features is not None else np.eye(nodes)
        self.edge_features = edge_features if edge_features is not None else {}
        
        # Build edge index and adjacency matrix
        if edges and len(edges) > 0:
            row, col = zip(*edges)
            self.edge_index = (np.array(row), np.array(col))
            self.edge_weight = np.ones(len(edges))
            self.adjacency_matrix = self._build_sparse_adjacency_matrix(nodes, edges)
        else:
            self.edge_index = (np.array([]), np.array([]))
            self.edge_weight = np.array([])
            self.adjacency_matrix = csr_matrix((nodes, nodes))
    
    def _build_sparse_adjacency_matrix(self, nodes, edges):
        """Build a sparse adjacency matrix from edges.
        
        Args:
            nodes (int): Number of nodes in the graph.
            edges (list): List of tuples (u, v) representing edges.
            
        Returns:
            scipy.sparse.csr_matrix: Sparse adjacency matrix.
        """
        row, col = zip(*edges)
        data = np.ones(len(edges))
        adj = csr_matrix((data, (row, col)), shape=(nodes, nodes))
        
        # Make the graph undirected if specified
        if not self.directed:
            adj = adj + adj.T
            # Remove duplicate entries (set diagonal to 0 first to avoid doubling self-loops)
            adj.setdiag(0)
            adj.eliminate_zeros()
            # Now we can safely set values to 1
            adj.data = np.ones_like(adj.data)
        
        return adj
    
    def add_edge(self, u, v, weight=1.0, features=None):
        """Add an edge to the graph.
        
        Args:
            u (int): Source node index.
            v (int): Target node index.
            weight (float, optional): Edge weight. Default is 1.0.
            features (numpy.ndarray, optional): Features for the edge. Default is None.
        """
        # Update adjacency matrix
        self.adjacency_matrix[u, v] = weight
        if not self.directed:
            self.adjacency_matrix[v, u] = weight
        
        # Update edge features if provided
        if features is not None:
            self.edge_features[(u, v)] = features
            if not self.directed:
                self.edge_features[(v, u)] = features
        
        # Update edge index and weights
        row, col = self.edge_index
        row = np.append(row, u)
        col = np.append(col, v)
        self.edge_index = (row, col)
        self.edge_weight = np.append(self.edge_weight, weight)
        
        if not self.directed:
            row = np.append(row, v)
            col = np.append(col, u)
            self.edge_index = (row, col)
            self.edge_weight = np.append(self.edge_weight, weight)
    
    def remove_edge(self, u, v):
        """Remove an edge from the graph.
        
        Args:
            u (int): Source node index.
            v (int): Target node index.
        """
        # Update adjacency matrix
        self.adjacency_matrix[u, v] = 0
        if not self.directed:
            self.adjacency_matrix[v, u] = 0
        self.adjacency_matrix.eliminate_zeros()
        
        # Remove from edge features if present
        if (u, v) in self.edge_features:
            del self.edge_features[(u, v)]
        if not self.directed and (v, u) in self.edge_features:
            del self.edge_features[(v, u)]
        
        # Update edge index and weights
        row, col = self.edge_index
        mask = ~((row == u) & (col == v))
        if not self.directed:
            mask = mask & ~((row == v) & (col == u))
        
        self.edge_index = (row[mask], col[mask])
        self.edge_weight = self.edge_weight[mask]
    
    def normalize_adjacency(self, add_self_loops=True, symmetric=True):
        """Normalize the adjacency matrix using degree information.
        
        Args:
            add_self_loops (bool, optional): Whether to add self-loops before normalization. Default is True.
            symmetric (bool, optional): Whether to use symmetric normalization. Default is True.
            
        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.
        """
        adj = self.adjacency_matrix.copy()
        
        if add_self_loops:
            adj = adj + eye(self.nodes)
        
        # Calculate degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        
        # Handle isolated nodes (degree=0)
        degrees[degrees == 0] = 1.0
        
        if symmetric:
            # D^(-1/2) * A * D^(-1/2)
            degree_inv_sqrt = diags(1.0 / np.sqrt(degrees))
            return degree_inv_sqrt @ adj @ degree_inv_sqrt
        else:
            # D^(-1) * A
            degree_inv = diags(1.0 / degrees)
            return degree_inv @ adj
    
    def compute_laplacian(self, normalized=True):
        """Compute the graph Laplacian matrix.
        
        Args:
            normalized (bool, optional): Whether to compute the normalized Laplacian. Default is True.
            
        Returns:
            scipy.sparse.csr_matrix: Graph Laplacian matrix.
        """
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degree_matrix = diags(degrees)
        laplacian = degree_matrix - self.adjacency_matrix
        
        if normalized:
            # Handle isolated nodes
            degrees[degrees == 0] = 1.0
            
            # D^(-1/2) * L * D^(-1/2)
            degree_inv_sqrt = diags(1.0 / np.sqrt(degrees))
            return degree_inv_sqrt @ laplacian @ degree_inv_sqrt
        else:
            return laplacian
    
    def spectral_clustering(self, n_clusters=2):
        """Perform spectral clustering on the graph.
        
        Args:
            n_clusters (int, optional): Number of clusters. Default is 2.
            
        Returns:
            numpy.ndarray: Cluster assignments for each node.
        """
        # Compute normalized Laplacian
        laplacian = self.compute_laplacian(normalized=True)
        
        # Compute eigenvectors
        try:
            _, eigenvectors = linalg.eigsh(laplacian, k=n_clusters, which='SM')
        except Exception as e:
            warnings.warn(f"Eigenvector computation failed: {e}. Using connected components instead.")
            return connected_components(self.adjacency_matrix)[1]
        
        # Perform k-means clustering on eigenvectors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        return kmeans.fit_predict(eigenvectors)
    
    def get_neighbors(self, node_idx):
        """Get the neighbors of a node.
        
        Args:
            node_idx (int): Index of the node.
            
        Returns:
            numpy.ndarray: Indices of neighboring nodes.
        """
        return self.adjacency_matrix[node_idx].nonzero()[1]
    
    def get_subgraph(self, node_indices):
        """Extract a subgraph containing only the specified nodes.
        
        Args:
            node_indices (list): List of node indices to include in the subgraph.
            
        Returns:
            Graph: A new Graph object representing the subgraph.
        """
        # Create a mapping from original to new indices
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Extract the subgraph adjacency matrix
        sub_adj = self.adjacency_matrix[node_indices, :][:, node_indices]
        
        # Get edges from the subgraph adjacency matrix
        sub_edges = list(zip(*sub_adj.nonzero()))
        
        # Extract node features if available
        sub_node_features = self.node_features[node_indices] if self.node_features is not None else None
        
        # Extract edge features if available
        sub_edge_features = {}
        for (u, v), feat in self.edge_features.items():
            if u in node_map and v in node_map:
                sub_edge_features[(node_map[u], node_map[v])] = feat
        
        return Graph(len(node_indices), sub_edges, sub_node_features, sub_edge_features, self.directed)
    
    def to_networkx(self):
        """Convert the graph to a NetworkX graph.
        
        Returns:
            networkx.Graph or networkx.DiGraph: NetworkX graph representation.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX is required for this functionality.")
        
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes with features
        for i in range(self.nodes):
            if self.node_features is not None:
                G.add_node(i, features=self.node_features[i])
            else:
                G.add_node(i)
        
        # Add edges with weights and features
        row, col = self.edge_index
        for i in range(len(row)):
            u, v = row[i], col[i]
            edge_data = {"weight": self.edge_weight[i]}
            if (u, v) in self.edge_features:
                edge_data["features"] = self.edge_features[(u, v)]
            G.add_edge(u, v, **edge_data)
        
        return G
