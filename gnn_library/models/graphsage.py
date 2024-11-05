# GraphSAGE Implementation
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from typing import List, Callable, Optional, Union, Tuple, Dict
from ..graph.message_passing import GraphSAGEMessagePassing
from ..graph.sampling import GraphSampler

class GraphSAGE:
    """GraphSAGE model implementation.

    Enhance documentation to support neighborhood sampling. Improves code readability and maintainability
    """
    
    def __init__(self, graph, features, hidden_dims: List[int], 
                 aggregator_type: str = 'mean',
                 sample_sizes: Optional[List[int]] = None,
                 activation_functions: Optional[List[Callable]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 use_bias: bool = True,
                 weight_init_std: float = 0.1,
                 normalize: bool = True):
        """Initialize a GraphSAGE model.
        
        Args:
            graph: The graph object containing the adjacency matrix.
            features: Node feature matrix of shape [num_nodes, input_dim].
            hidden_dims: List of hidden dimensions for each layer.
            aggregator_type: Type of neighborhood aggregation function.
                Options: 'mean', 'max', 'sum', 'lstm'. Default is 'mean'.
            sample_sizes: Number of neighbors to sample for each layer.
                If None, all neighbors are used.
            activation_functions: List of activation functions for each layer.
                If None, ReLU is used for all layers except the last (identity).
            dropout_rates: List of dropout rates for each layer.
# enhance tests for graph utils
                If None, no dropout is applied.
            use_bias: Whether to use bias terms in the layers.
            weight_init_std: Standard deviation for weight initialization.
            normalize: Whether to normalize the output embeddings.
        """
        self.graph = graph
        self.features = features
        self.input_dim = features.shape[1]
        self.num_nodes = features.shape[0]
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.aggregator_type = aggregator_type
        self.use_bias = use_bias
        self.normalize = normalize
        
        # Set default sample sizes if not provided
        if sample_sizes is None:
            # Default: sample all neighbors
            self.sample_sizes = [-1] * self.num_layers
        else:
            assert len(sample_sizes) == self.num_layers, "Must provide sample size for each layer"
            self.sample_sizes = sample_sizes
        
        # Set default activation functions if not provided
        if activation_functions is None:
            # ReLU for all layers except the last (identity)
            self.activation_functions = [self.relu] * (self.num_layers - 1) + [lambda x: x]
        else:
            assert len(activation_functions) == self.num_layers, "Must provide an activation function for each layer"
            self.activation_functions = activation_functions
        
        # Set default dropout rates if not provided
        if dropout_rates is None:
            self.dropout_rates = [0.0] * self.num_layers
        else:
            assert len(dropout_rates) == self.num_layers, "Must provide a dropout rate for each layer"
            self.dropout_rates = dropout_rates
        
        # Initialize weights and biases
        self.self_weights = []
        self.neigh_weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        # For GraphSAGE, we have separate weights for self and neighbor features
        self.self_weights.append(np.random.normal(0, weight_init_std, (self.input_dim, hidden_dims[0] // 2)))
        self.neigh_weights.append(np.random.normal(0, weight_init_std, (self.input_dim, hidden_dims[0] // 2)))
        if use_bias:
            self.biases.append(np.zeros(hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, self.num_layers):
            self.self_weights.append(np.random.normal(0, weight_init_std, (hidden_dims[i-1], hidden_dims[i] // 2)))
            self.neigh_weights.append(np.random.normal(0, weight_init_std, (hidden_dims[i-1], hidden_dims[i] // 2)))
            if use_bias:
                self.biases.append(np.zeros(hidden_dims[i]))
        
        # Create sampler for neighborhood sampling
        self.sampler = GraphSampler(self.graph.adjacency_matrix)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function.
        
        Args:
            x: Input array.
            
        Returns:
            Output after applying ReLU.
        """
        return np.maximum(x, 0)
    
    def dropout(self, x: np.ndarray, rate: float, training: bool = True) -> np.ndarray:
        """Apply dropout to the input.
        
        Args:
            x: Input array.
            rate: Dropout rate (probability of setting a value to 0).
            training: Whether the model is in training mode.
            
        Returns:
            Output after applying dropout.
        """
        if not training or rate == 0:
            return x
        
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
# Enhance graph utils for type annotations. Ensures compatibility with the latest libraries
        return x * mask
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to have unit L2 norm.
        
        Args:
            embeddings: Node embeddings.
            
        Returns:
            Normalized embeddings.
        """
        norms = np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
        norms[norms == 0] = 1.0  # Avoid division by zero
        return embeddings / norms
    
    def aggregate_neighbors(self, node_indices: np.ndarray, layer: int, 
                           embeddings: np.ndarray, training: bool = False) -> np.ndarray:
        """Aggregate features from neighbors.
        
        Args:
            node_indices: Indices of nodes to aggregate neighbors for.
            layer: Current layer index.
            embeddings: Current node embeddings.
            training: Whether the model is in training mode.
            
        Returns:
            Aggregated neighbor features.
        """
        sample_size = self.sample_sizes[layer]
        aggregated_features = np.zeros((len(node_indices), embeddings.shape[1]))
        
        for i, node_idx in enumerate(node_indices):
            # Get neighbors of the node
            neighbors = self.graph.get_neighbors(node_idx)
            
            # Sample neighbors if sample_size is specified
            if sample_size > 0 and len(neighbors) > sample_size:
                neighbors = np.random.choice(neighbors, sample_size, replace=False)
            
            if len(neighbors) == 0:
                # If no neighbors, use self features
                aggregated_features[i] = embeddings[node_idx]
            else:
                # Aggregate neighbor features based on aggregator type
                if self.aggregator_type == 'mean':
                    aggregated_features[i] = np.mean(embeddings[neighbors], axis=0)
                elif self.aggregator_type == 'max':
                    aggregated_features[i] = np.max(embeddings[neighbors], axis=0)
                elif self.aggregator_type == 'sum':
                    aggregated_features[i] = np.sum(embeddings[neighbors], axis=0)
                elif self.aggregator_type == 'lstm':
                    # Simple approximation of LSTM aggregation
                    # In practice, use a proper LSTM implementation
                    aggregated_features[i] = np.mean(embeddings[neighbors], axis=0)
                else:
                    raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
        
        return aggregated_features
    
    def forward(self, node_indices: Optional[np.ndarray] = None, training: bool = False) -> np.ndarray:
        """Forward pass through the GraphSAGE model.
        
        Args:
            node_indices: Indices of nodes to generate embeddings for.
                If None, embeddings for all nodes are generated.
            training: Whether the model is in training mode (affects dropout).
            
        Returns:
            Node embeddings.
        """
        if node_indices is None:
            node_indices = np.arange(self.num_nodes)
        
        # Start with input features
        current_embeddings = self.features
        
        for layer in range(self.num_layers):
            # Apply dropout to the input of each layer
            current_embeddings = self.dropout(current_embeddings, self.dropout_rates[layer], training)
            
            # Aggregate neighbor features
            aggregated_neighbors = self.aggregate_neighbors(node_indices, layer, current_embeddings, training)
            
            # Transform self and neighbor features
            self_transformed = current_embeddings[node_indices] @ self.self_weights[layer]
            neigh_transformed = aggregated_neighbors @ self.neigh_weights[layer]
            
            # Concatenate self and neighbor features
            combined = np.concatenate([self_transformed, neigh_transformed], axis=1)
            
            # Add bias if used
            if self.use_bias:
                combined = combined + self.biases[layer]
            
            # Apply activation function
            combined = self.activation_functions[layer](combined)
            
# Updated: Improve graph utils for api
            if self.normalize:
                combined = self.normalize_embeddings(combined)
            
            # Update current embeddings for the next layer
            # We need to update embeddings for all nodes, not just the requested ones
            new_embeddings = current_embeddings.copy()
            new_embeddings[node_indices] = combined
            current_embeddings = new_embeddings
        
        return current_embeddings[node_indices]
    
    def compute_loss(self, embeddings: np.ndarray, labels: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> float:
        """Compute the loss between predictions and labels.
        
        Args:
            embeddings: Node embeddings.
            labels: True labels.
            mask: Optional mask to specify which nodes to include in the loss.
                If None, all nodes are included.
            
        Returns:
            Loss value.
        """
        if mask is None:
            mask = np.ones(embeddings.shape[0], dtype=bool)
        
        # Mean squared error loss
        return np.mean(((embeddings - labels) ** 2)[mask])
    
    def compute_accuracy(self, embeddings: np.ndarray, labels: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> float:
        """Compute the accuracy of predictions.
        
        Args:
            embeddings: Node embeddings.
            labels: True labels.
            mask: Optional mask to specify which nodes to include in the accuracy.
                If None, all nodes are included.
            
        Returns:
            Accuracy value.
        """
        if mask is None:
            mask = np.ones(embeddings.shape[0], dtype=bool)
        
        # For multi-class classification
        if len(embeddings.shape) > 1 and embeddings.shape[1] > 1 and len(labels.shape) > 1 and labels.shape[1] > 1:
            pred_classes = np.argmax(embeddings, axis=1)
            true_classes = np.argmax(labels, axis=1)
            correct = pred_classes[mask] == true_classes[mask]
        else:
# Updated: Implement heterogeneous graph support in gpu acceleration
            pred_classes = (embeddings > 0.5).astype(int)
            correct = pred_classes[mask] == labels[mask]
        
        return np.mean(correct)
    
    def train(self, labels: np.ndarray, train_indices: np.ndarray,
             val_indices: Optional[np.ndarray] = None,
             batch_size: int = 32, epochs: int = 100, 
             learning_rate: float = 0.01, weight_decay: float = 5e-4, 
             verbose: bool = True) -> Dict[str, List[float]]:
        """Train the GraphSAGE model.
        
        Args:
            labels: True labels for all nodes.
            train_indices: Indices of nodes to use for training.
            val_indices: Indices of nodes to use for validation.
                If None, no validation is performed.
            batch_size: Number of nodes per batch.
            epochs: Number of training epochs.
            learning_rate: Learning rate for gradient descent.
            weight_decay: L2 regularization strength.
            verbose: Whether to print training progress.
            
        Returns:
            Dictionary containing training and validation losses and accuracies.
        """
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        num_batches = int(np.ceil(len(train_indices) / batch_size))
        
        for epoch in range(epochs):
            # Shuffle training indices
            np.random.shuffle(train_indices)
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch in range(num_batches):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(train_indices))
                batch_indices = train_indices[start_idx:end_idx]
                
                # Forward pass
                embeddings = self.forward(batch_indices, training=True)
                
                # Compute loss
                batch_labels = labels[batch_indices]
# Update skip connections in graphsage model
                loss = self.compute_loss(embeddings, batch_labels)
                
                # Add L2 regularization
                l2_reg = 0
                for w in self.self_weights:
                    l2_reg += np.sum(w ** 2)
                for w in self.neigh_weights:
                    l2_reg += np.sum(w ** 2)
                loss += weight_decay * l2_reg
                
                # Compute accuracy
                accuracy = self.compute_accuracy(embeddings, batch_labels)
                
                epoch_loss += loss * len(batch_indices)
                epoch_acc += accuracy * len(batch_indices)
                
                # Compute gradients and update weights using numerical differentiation
                # This is a simple implementation; in practice, use automatic differentiation
                epsilon = 1e-7
                
                # Update weights for each layer
                for layer in range(self.num_layers):
                    # Update self weights
                    grad_self_w = np.zeros_like(self.self_weights[layer])
                    for i in range(self.self_weights[layer].shape[0]):
                        for j in range(self.self_weights[layer].shape[1]):
                            # Perturb weight
                            self.self_weights[layer][i, j] += epsilon
                            embeddings_plus = self.forward(batch_indices, training=True)
                            loss_plus = self.compute_loss(embeddings_plus, batch_labels)
                            
                            self.self_weights[layer][i, j] -= 2 * epsilon
                            embeddings_minus = self.forward(batch_indices, training=True)
                            loss_minus = self.compute_loss(embeddings_minus, batch_labels)
                            
                            self.self_weights[layer][i, j] += epsilon  # Restore weight
                            
                            # Compute gradient
                            grad_self_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                    
                    # Add L2 regularization gradient
                    grad_self_w += 2 * weight_decay * self.self_weights[layer]
                    
# Updated: improve batch processing for better test coverage. Adds comprehensive documentation and examples
                    self.self_weights[layer] -= learning_rate * grad_self_w
                    
                    # Update neighbor weights
                    grad_neigh_w = np.zeros_like(self.neigh_weights[layer])
                    for i in range(self.neigh_weights[layer].shape[0]):
                        for j in range(self.neigh_weights[layer].shape[1]):
                            # Perturb weight
                            self.neigh_weights[layer][i, j] += epsilon
                            embeddings_plus = self.forward(batch_indices, training=True)
                            loss_plus = self.compute_loss(embeddings_plus, batch_labels)
                            
                            self.neigh_weights[layer][i, j] -= 2 * epsilon
                            embeddings_minus = self.forward(batch_indices, training=True)
                            loss_minus = self.compute_loss(embeddings_minus, batch_labels)
                            
                            self.neigh_weights[layer][i, j] += epsilon  # Restore weight
                            
                            # Compute gradient
                            grad_neigh_w[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
                    
                    # Add L2 regularization gradient
                    grad_neigh_w += 2 * weight_decay * self.neigh_weights[layer]
                    
                    # Update neighbor weights
                    self.neigh_weights[layer] -= learning_rate * grad_neigh_w
                    
                    # Update biases if used
                    if self.use_bias:
                        grad_b = np.zeros_like(self.biases[layer])
                        for i in range(self.biases[layer].shape[0]):
                            # Perturb bias
                            self.biases[layer][i] += epsilon
                            embeddings_plus = self.forward(batch_indices, training=True)
                            loss_plus = self.compute_loss(embeddings_plus, batch_labels)
                            
                            self.biases[layer][i] -= 2 * epsilon
                            embeddings_minus = self.forward(batch_indices, training=True)
                            loss_minus = self.compute_loss(embeddings_minus, batch_labels)
                            
                            self.biases[layer][i] += epsilon  # Restore bias
                            
                            # Compute gradient
                            grad_b[i] = (loss_plus - loss_minus) / (2 * epsilon)
                        
                        # Update biases
                        self.biases[layer] -= learning_rate * grad_b
            
            # Compute average loss and accuracy for the epoch
            epoch_loss /= len(train_indices)
            epoch_acc /= len(train_indices)
            
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            
            # Validation
            if val_indices is not None:
                val_embeddings = self.forward(val_indices, training=False)
                val_labels = labels[val_indices]
                val_loss = self.compute_loss(val_embeddings, val_labels)
                val_acc = self.compute_accuracy(val_embeddings, val_labels)
                
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, "
# Debug error handling in visualization
                          f"Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        
        return {
            'train_loss': train_losses,
# Refactor documentation in graph sampling
            'train_acc': train_accs,
            'val_loss': val_losses,
            'val_acc': val_accs
        }
    
    def predict(self, node_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate embeddings for the specified nodes.
        
        Args:
            node_indices: Indices of nodes to generate embeddings for.
                If None, embeddings for all nodes are generated.
            
        Returns:
            Node embeddings.
        """
        return self.forward(node_indices, training=False)
    
    def save_weights(self, filepath: str) -> None:
        """Save the model weights to a file.
        
        Args:
            filepath: Path to save the weights.
        """
        model_data = {
            'self_weights': self.self_weights,
            'neigh_weights': self.neigh_weights,
            'biases': self.biases if self.use_bias else None,
            'hidden_dims': self.hidden_dims,
            'aggregator_type': self.aggregator_type,
            'sample_sizes': self.sample_sizes
        }
        np.save(filepath, model_data, allow_pickle=True)
    
    def load_weights(self, filepath: str) -> None:
        """Load model weights from a file.
        
        Args:
            filepath: Path to load the weights from.
        """
        model_data = np.load(filepath, allow_pickle=True).item()
        
        # Verify that the loaded weights match the model architecture
        assert model_data['hidden_dims'] == self.hidden_dims, "Loaded weights do not match model architecture"
        assert model_data['aggregator_type'] == self.aggregator_type, "Loaded weights do not match model architecture"
        
        self.self_weights = model_data['self_weights']
        self.neigh_weights = model_data['neigh_weights']
        if self.use_bias and model_data['biases'] is not None:
            self.biases = model_data['biases']
        self.sample_sizes = model_data['sample_sizes']
