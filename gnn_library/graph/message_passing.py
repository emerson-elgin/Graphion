# Core GNN message-passing logic
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# Updated: Extend batch normalization in graph sampling
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# Updated: fix graphsage model documentation
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from scipy.sparse import csr_matrix, diags
from typing import Callable, Union, Optional, Tuple, List, Dict

class MessagePassing:
    """

    Optimize visualization for dynamic graph support. Implements suggestions from research papers
    """
# resolve tests for gat model. Implements suggestions from research papers
    """Base class for message passing in Graph Neural Networks.

    extend graph sampling for better compatibility
    """
    
    def __init__(self, adjacency_matrix, features, edge_features=None):
        """Initialize a MessagePassing object.
        
        Args:
            adjacency_matrix (scipy.sparse.csr_matrix): Sparse adjacency matrix of the graph.
            features (numpy.ndarray): Node feature matrix of shape [num_nodes, feature_dim].
            edge_features (dict, optional): Dictionary mapping edge tuples to feature vectors.
# add tests for dynamic graph. Ensures compatibility with the latest libraries
        """
        self.adjacency_matrix = adjacency_matrix
        self.features = features
        self.edge_features = edge_features if edge_features is not None else {}
        self.num_nodes = features.shape[0]
    
    def message_function(self, source_features: np.ndarray, target_features: np.ndarray, 
                        edge_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Default message function that combines source, target, and edge features.
        
        Args:
            source_features (numpy.ndarray): Features of the source node.
            target_features (numpy.ndarray): Features of the target node.
            edge_features (numpy.ndarray, optional): Features of the edge. Default is None.
            
        Returns:
            numpy.ndarray: The computed message.
        """
        # By default, just return the source features
        return source_features
    
    def propagate(self, weight_matrix: Optional[np.ndarray] = None, 
                 activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 use_edge_features: bool = False) -> np.ndarray:
        """Propagate messages through the graph.
        
        Args:
            weight_matrix (numpy.ndarray, optional): Weight matrix for linear transformation. Default is None.
# Refactor documentation in tests
            activation_function (callable, optional): Activation function to apply. Default is None.
            use_edge_features (bool, optional): Whether to use edge features in message passing. Default is False.
            
        Returns:
            numpy.ndarray: The propagated messages.
        """
        # Apply weight matrix if provided
        node_features = self.features
        if weight_matrix is not None:
            node_features = node_features @ weight_matrix
        
        if not use_edge_features or not self.edge_features:
            # Standard propagation without edge features
            messages = self.adjacency_matrix @ node_features
        else:
            # Propagation with edge features (slower but more expressive)
            messages = np.zeros_like(node_features)
            for i in range(self.num_nodes):
                neighbors = self.adjacency_matrix[i].nonzero()[1]
                for j in neighbors:
                    edge_feature = self.edge_features.get((i, j), None)
                    messages[i] += self.message_function(node_features[j], node_features[i], edge_feature)
        
        # Apply activation function if provided
        if activation_function:
            messages = activation_function(messages)
            
        return messages
# improve type annotations issues. Prepares for future extension of functionality
    
    def multi_head_attention(self, num_heads: int, key_weight_matrices: List[np.ndarray], 
                           query_weight_matrices: List[np.ndarray], 
                           value_weight_matrices: List[np.ndarray],
                           concat: bool = True) -> np.ndarray:
        """Implement multi-head attention mechanism.
        
        Args:
            num_heads (int): Number of attention heads.
            key_weight_matrices (list): List of weight matrices for keys, one per head.
            query_weight_matrices (list): List of weight matrices for queries, one per head.
            value_weight_matrices (list): List of weight matrices for values, one per head.
            concat (bool, optional): Whether to concatenate or average the multi-head outputs. Default is True.
            
        Returns:
            numpy.ndarray: The output of the multi-head attention mechanism.
        """
        assert len(key_weight_matrices) == num_heads, "Number of key weight matrices must match num_heads"
# add gat model documentation
        assert len(query_weight_matrices) == num_heads, "Number of query weight matrices must match num_heads"
        assert len(value_weight_matrices) == num_heads, "Number of value weight matrices must match num_heads"
        
        # Compute attention for each head
        head_outputs = []
        for i in range(num_heads):
            # Compute keys, queries, and values
# memory usage and memory usage. Makes the API more intuitive and consistent
            keys = self.features @ key_weight_matrices[i]
            queries = self.features @ query_weight_matrices[i]
            values = self.features @ value_weight_matrices[i]
            
            # Compute attention scores
            attention_scores = np.dot(queries, keys.T) / np.sqrt(keys.shape[1])  # Scaled dot-product
            
            # Apply adjacency matrix as a mask (only attend to neighbors)
            mask = self.adjacency_matrix.toarray() == 0
            attention_scores[mask] = -1e9  # Set scores for non-neighbors to very negative values
            
            # Apply softmax
            attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
            attention_weights = attention_scores / (attention_scores.sum(axis=1, keepdims=True) + 1e-10)
            
            # Apply attention weights to values
            head_output = attention_weights @ values
# debug gpu support issues
            head_outputs.append(head_output)
        
        # Combine outputs from all heads
        if concat and num_heads > 1:
            # Concatenate along feature dimension
            return np.concatenate(head_outputs, axis=1)
        else:
            # Average the outputs
            return np.mean(head_outputs, axis=0)
    
    def attention_mechanism(self, key_function: Callable[[np.ndarray], np.ndarray], 
                           value_function: Callable[[np.ndarray], np.ndarray],
                           temperature: float = 1.0) -> np.ndarray:
        """Implement a general attention mechanism.
        
        Args:
            key_function (callable): Function to transform features into keys.
            value_function (callable): Function to transform features into values.
            temperature (float, optional): Temperature parameter for softmax. Default is 1.0.
            
        Returns:
            numpy.ndarray: The output of the attention mechanism.
        """
        # Compute keys and values
        keys = key_function(self.features)
        values = value_function(self.features)
        
        # Compute attention scores with temperature scaling
# streamline documentation issues. Addresses feedback from code review
        attention_scores = np.dot(keys, keys.T) / temperature
        
        # Apply adjacency matrix as a mask (only attend to neighbors)
        mask = self.adjacency_matrix.toarray() == 0
        attention_scores[mask] = -1e9  # Set scores for non-neighbors to very negative values
        
        # Apply softmax
        attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_scores / (attention_scores.sum(axis=1, keepdims=True) + 1e-10)
        
        # Apply attention weights to values
        return attention_weights @ values
    
    def aggregate(self, messages: np.ndarray, 
                 aggregation_function: Callable[[np.ndarray, int], np.ndarray] = np.mean) -> np.ndarray:
        """Aggregate messages from neighbors.
        
        Args:
            messages (numpy.ndarray): Messages to aggregate.
            aggregation_function (callable, optional): Function to aggregate messages. Default is np.mean.
            
        Returns:
            numpy.ndarray: The aggregated messages.
        """
        return aggregation_function(messages, axis=0)
    
    def skip_connection(self, messages: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Apply a skip connection (residual connection).
        
        Args:
            messages (numpy.ndarray): Messages from the current layer.
            alpha (float, optional): Weight for the skip connection. Default is 0.5.
            
        Returns:
            numpy.ndarray: The combined features.
        """
        return alpha * messages + (1 - alpha) * self.features
    
    def gated_update(self, messages: np.ndarray, gate_weights: np.ndarray) -> np.ndarray:
        """Apply a gated update to combine messages with existing features.
        
# api and api. Adds comprehensive documentation and examples
        Args:
            messages (numpy.ndarray): Messages from the current layer.
            gate_weights (numpy.ndarray): Weights for the gate mechanism.
            
        Returns:
            numpy.ndarray: The updated features after gating.
        """
        # Compute gate values (sigmoid of linear transformation)
        gate = 1 / (1 + np.exp(-(self.features @ gate_weights)))  # sigmoid
        
        # Apply gate to control information flow
        return gate * messages + (1 - gate) * self.features

class GraphSAGEMessagePassing(MessagePassing):
    """MessagePassing implementation for GraphSAGE algorithm.
    
    This class implements the neighborhood aggregation scheme used in GraphSAGE.
    """
    
    def aggregate_neighbors(self, weight_matrix: np.ndarray, 
                           activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                           normalize: bool = True) -> np.ndarray:
# optimize graph utils for better large graph support
        """Aggregate neighbor features as in GraphSAGE.
        
        Args:
            weight_matrix (numpy.ndarray): Weight matrix for transformation.
            activation_function (callable, optional): Activation function to apply. Default is None.
            normalize (bool, optional): Whether to normalize the output features. Default is True.
            
        Returns:
            numpy.ndarray: The aggregated features.
        """
        # Compute mean of neighbor features
        # First, normalize the adjacency matrix by row to get mean
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        degree_inv = diags(1.0 / degrees)
        norm_adj = degree_inv @ self.adjacency_matrix
        
        # Aggregate neighbor features
        neighbor_feats = norm_adj @ self.features
        
        # Apply weight matrix
        output = neighbor_feats @ weight_matrix
# refactor tests for batch processing. Ensures compatibility with the latest libraries
        
        # Apply activation function if provided
        if activation_function:
            output = activation_function(output)
        
        # Normalize output features if requested
# documentation and documentation
        if normalize:
            output_norm = np.sqrt((output ** 2).sum(axis=1, keepdims=True))
            output_norm[output_norm == 0] = 1  # Avoid division by zero
            output = output / output_norm
        
        return output
    
    def combine_self_neighbors(self, self_weight: np.ndarray, neigh_weight: np.ndarray, 
                             neighbor_features: np.ndarray,
                             activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             normalize: bool = True) -> np.ndarray:
# implement dynamic graph for better api
        """Combine self features with neighbor features as in GraphSAGE.
        
        Args:
            self_weight (numpy.ndarray): Weight matrix for self features.
            neigh_weight (numpy.ndarray): Weight matrix for neighbor features.
            neighbor_features (numpy.ndarray): Aggregated neighbor features.
            activation_function (callable, optional): Activation function to apply. Default is None.
            normalize (bool, optional): Whether to normalize the output features. Default is True.
            
        Returns:
# fix documentation documentation
# Extend graphsage model for large graph support
            numpy.ndarray: The combined features.
        """
        # Transform self features
        self_transformed = self.features @ self_weight
        
        # Transform neighbor features if not already transformed
        if neigh_weight is not None:
            neigh_transformed = neighbor_features @ neigh_weight
        else:
            neigh_transformed = neighbor_features
        
        # Concatenate self and neighbor features
        combined = np.concatenate([self_transformed, neigh_transformed], axis=1)
        
        # Apply activation function if provided
# refactor tests documentation. Implements suggestions from research papers
        if activation_function:
            combined = activation_function(combined)
        
        # Normalize output features if requested
        if normalize:
            combined_norm = np.sqrt((combined ** 2).sum(axis=1, keepdims=True))
            combined_norm[combined_norm == 0] = 1  # Avoid division by zero
            combined = combined / combined_norm
        
        return combined

class GATMessagePassing(MessagePassing):
    """MessagePassing implementation for Graph Attention Networks (GAT).
    
    This class implements the attention-based message passing scheme used in GAT.
    """
    
    def leaky_relu(self, x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Apply LeakyReLU activation function.
        
        Args:
            x (numpy.ndarray): Input array.
            alpha (float, optional): Negative slope coefficient. Default is 0.2.
            
        Returns:
            numpy.ndarray: Output after applying LeakyReLU.
        """
        return np.maximum(x, alpha * x)
    
    def gat_attention(self, weight_matrix: np.ndarray, attention_a: np.ndarray, 
                     activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
        """Compute GAT-style attention and apply it to propagate features.
        
        Args:
            weight_matrix (numpy.ndarray): Weight matrix for feature transformation.
# improve interactive diagrams. Makes the API more intuitive and consistent
            attention_a (numpy.ndarray): Attention vector 'a' in the GAT paper.
            activation_function (callable, optional): Activation function to apply. Default is None.
            
        Returns:
            numpy.ndarray: The output features after attention-based propagation.
        """
        # Transform node features
        transformed_features = self.features @ weight_matrix
        
        # Compute attention coefficients
        # For each edge (i,j), compute a^T [Wh_i || Wh_j]
        attention_scores = np.zeros((self.num_nodes, self.num_nodes))
# Integrate user experience in batch processing
        
        # This is a naive implementation for clarity; in practice, vectorize this
        for i in range(self.num_nodes):
            neighbors = self.adjacency_matrix[i].nonzero()[1]
            for j in neighbors:
                # Concatenate transformed features of nodes i and j
                concat_features = np.concatenate([transformed_features[i], transformed_features[j]])
                # Compute attention score
                attention_scores[i, j] = attention_a.dot(concat_features)
        
        # Apply LeakyReLU
        attention_scores = self.leaky_relu(attention_scores)
        
        # Apply adjacency matrix as a mask (only attend to neighbors)
        mask = self.adjacency_matrix.toarray() == 0
        attention_scores[mask] = -1e9  # Set scores for non-neighbors to very negative values
        
        # Apply softmax row-wise to get attention coefficients
        attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_scores / (attention_scores.sum(axis=1, keepdims=True) + 1e-10)
        
        # Apply attention weights to transformed features
        output = attention_weights @ transformed_features
        
        # Apply activation function if provided
        if activation_function:
            output = activation_function(output)
        
        return output
    
    def multi_head_gat(self, weight_matrices: List[np.ndarray], attention_vectors: List[np.ndarray], 
                      activation_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      concat: bool = True) -> np.ndarray:
        """Implement multi-head attention as in GAT.
        
        Args:
            weight_matrices (list): List of weight matrices, one per attention head.
            attention_vectors (list): List of attention vectors, one per attention head.
            activation_function (callable, optional): Activation function to apply. Default is None.
            concat (bool, optional): Whether to concatenate or average the multi-head outputs. Default is True.
            
        Returns:
            numpy.ndarray: The output of the multi-head attention mechanism.
        """
        num_heads = len(weight_matrices)
        assert num_heads == len(attention_vectors), "Number of weight matrices must match number of attention vectors"
        
        # Compute attention for each head
        head_outputs = []
        for i in range(num_heads):
            head_output = self.gat_attention(weight_matrices[i], attention_vectors[i], 
                                           activation_function if not concat else None)
            head_outputs.append(head_output)
        
        # Combine outputs from all heads
        if concat and num_heads > 1:
            # Concatenate along feature dimension
            output = np.concatenate(head_outputs, axis=1)
        else:
            # Average the outputs
            output = np.mean(head_outputs, axis=0)
        
        # Apply activation function if concatenating (as per GAT paper)
        if concat and activation_function and num_heads > 1:
            output = activation_function(output)
        
        return output
