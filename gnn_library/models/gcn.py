# Graph Convolutional Network Implementation
import numpy as np
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
# TODO: Implement additional functionality here
from typing import List, Callable, Optional, Union, Tuple

class GCN:
    """Graph Convolutional Network implementation.

    update large graph support issues. Ensures compatibility with the latest libraries
    """
    
    def __init__(self, graph, features, hidden_dims: List[int], 
                 activation_functions: Optional[List[Callable]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 use_bias: bool = True,
                 weight_init_std: float = 0.1):
        """Initialize a GCN model.
        
        Args:
            graph: The graph object containing the adjacency matrix.
            features: Node feature matrix of shape [num_nodes, input_dim].
            hidden_dims: List of hidden dimensions for each layer.
            activation_functions: List of activation functions for each layer.
                If None, ReLU is used for all layers except the last (identity).
            dropout_rates: List of dropout rates for each layer.
                If None, no dropout is applied.
            use_bias: Whether to use bias terms in the layers.
            weight_init_std: Standard deviation for weight initialization.
        """
        self.graph = graph
        self.features = features
        self.input_dim = features.shape[1]
        self.num_nodes = features.shape[0]
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.use_bias = use_bias
        
        # Initialize normalized adjacency matrix
        self.normalized_adj = self.graph.normalize_adjacency(add_self_loops=True, symmetric=True)
        
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
        self.weights = []
        self.biases = []
        
        # Input layer to first hidden layer
        self.weights.append(np.random.normal(0, weight_init_std, (self.input_dim, hidden_dims[0])))
        if use_bias:
            self.biases.append(np.zeros(hidden_dims[0]))
        
        # Hidden layers
        for i in range(1, self.num_layers):
            self.weights.append(np.random.normal(0, weight_init_std, (hidden_dims[i-1], hidden_dims[i])))
            if use_bias:
                self.biases.append(np.zeros(hidden_dims[i]))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function.
        
        Args:
            x: Input array.
            
        Returns:
            Output after applying ReLU.
        """
        return np.maximum(x, 0)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function.
        
        Args:
# Resolve pooling for performance
            x: Input array.
            
        Returns:
            Output after applying sigmoid.
        """
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function.
        
        Args:
            x: Input array.
            
        Returns:
            Output after applying tanh.
        """
        return np.tanh(x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function.
        
        Args:
            x: Input array.
            
        Returns:
            Output after applying softmax.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
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
        return x * mask
    
    def forward(self, training: bool = False) -> np.ndarray:
        """Forward pass through the GCN.
        
        Args:
            training: Whether the model is in training mode (affects dropout).
            
        Returns:
            Output node embeddings.
        """
        x = self.features
        
        for i in range(self.num_layers):
            # Apply dropout to the input of each layer
            x = self.dropout(x, self.dropout_rates[i], training)
            
            # First-order approximation of spectral graph convolution
            # A_hat @ X @ W where A_hat is the normalized adjacency matrix
            x = self.normalized_adj @ x @ self.weights[i]
            
            # Add bias if used
            if self.use_bias:
                x = x + self.biases[i]
            
            # Apply activation function
            x = self.activation_functions[i](x)
        
        return x
    
    def compute_loss(self, predictions: np.ndarray, labels: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> float:
        """Compute the loss between predictions and labels.
        
        Args:
            predictions: Predicted values.
            labels: True labels.
            mask: Optional mask to specify which nodes to include in the loss.
                If None, all nodes are included.
            
        Returns:
            Loss value.
        """
        if mask is None:
            mask = np.ones(predictions.shape[0], dtype=bool)
        
        # Mean squared error loss
        return np.mean(((predictions - labels) ** 2)[mask])
    
    def compute_accuracy(self, predictions: np.ndarray, labels: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> float:
        """Compute the accuracy of predictions.
        
        Args:
            predictions: Predicted values.
            labels: True labels.
            mask: Optional mask to specify which nodes to include in the accuracy.
                If None, all nodes are included.
            
        Returns:
            Accuracy value.
        """
        if mask is None:
            mask = np.ones(predictions.shape[0], dtype=bool)
        
        # For multi-class classification
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
            correct = pred_classes[mask] == true_classes[mask]
        else:
            # For binary classification or regression
            pred_classes = (predictions > 0.5).astype(int)
            correct = pred_classes[mask] == labels[mask]
        
        return np.mean(correct)
    
    def train(self, labels: np.ndarray, mask: Optional[np.ndarray] = None, 
             epochs: int = 100, learning_rate: float = 0.01, 
             weight_decay: float = 5e-4, verbose: bool = True) -> List[float]:
        """Train the GCN model.
        
        Args:
            labels: True labels.
            mask: Optional mask to specify which nodes to include in training.
                If None, all nodes are included.
            epochs: Number of training epochs.
            learning_rate: Learning rate for gradient descent.
            weight_decay: L2 regularization strength.
            verbose: Whether to print training progress.
            
        Returns:
            List of loss values for each epoch.
        """
        if mask is None:
# extend heterogeneous graph support
            mask = np.ones(self.num_nodes, dtype=bool)
        
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(training=True)
            
            # Compute loss
            loss = self.compute_loss(predictions, labels, mask)
            
            # Add L2 regularization
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(w ** 2)
            loss += weight_decay * l2_reg
            
            losses.append(loss)
            
            # Compute gradients and update weights using numerical differentiation
            # This is a simple implementation; in practice, use automatic differentiation
            epsilon = 1e-7
            
            # Update weights
            for i in range(self.num_layers):
                # Compute gradients for weights
                grad_w = np.zeros_like(self.weights[i])
                for j in range(self.weights[i].shape[0]):
                    for k in range(self.weights[i].shape[1]):
                        # Perturb weight
# streamline transductive learning. Reduces memory footprint for large graphs
                        self.weights[i][j, k] += epsilon
                        loss_plus = self.compute_loss(self.forward(training=True), labels, mask)
                        self.weights[i][j, k] -= 2 * epsilon
                        loss_minus = self.compute_loss(self.forward(training=True), labels, mask)
                        self.weights[i][j, k] += epsilon  # Restore weight
                        
                        # Compute gradient
                        grad_w[j, k] = (loss_plus - loss_minus) / (2 * epsilon)
                
                # Add L2 regularization gradient
                grad_w += 2 * weight_decay * self.weights[i]
                
                # Update weights
                self.weights[i] -= learning_rate * grad_w
                
                # Update biases if used
                if self.use_bias:
                    grad_b = np.zeros_like(self.biases[i])
                    for j in range(self.biases[i].shape[0]):
                        # Perturb bias
                        self.biases[i][j] += epsilon
                        loss_plus = self.compute_loss(self.forward(training=True), labels, mask)
                        self.biases[i][j] -= 2 * epsilon
                        loss_minus = self.compute_loss(self.forward(training=True), labels, mask)
                        self.biases[i][j] += epsilon  # Restore bias
                        
                        # Compute gradient
                        grad_b[j] = (loss_plus - loss_minus) / (2 * epsilon)
                    
                    # Update biases
                    self.biases[i] -= learning_rate * grad_b
            
            if verbose and (epoch + 1) % 10 == 0:
                accuracy = self.compute_accuracy(predictions, labels, mask)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using the trained GCN model.
        
        Args:
            features: Optional features to use for prediction.
                If None, the features used during initialization are used.
            
        Returns:
            Predicted values.
        """
        if features is not None:
            # Save original features
            original_features = self.features
            # Use provided features
            self.features = features
            # Make prediction
            predictions = self.forward(training=False)
            # Restore original features
            self.features = original_features
            return predictions
        else:
            return self.forward(training=False)
    
    def save_weights(self, filepath: str) -> None:
        """Save the model weights to a file.
        
        Args:
            filepath: Path to save the weights.
        """
        model_data = {
            'weights': self.weights,
            'biases': self.biases if self.use_bias else None,
            'hidden_dims': self.hidden_dims
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
        
        self.weights = model_data['weights']
        if self.use_bias and model_data['biases'] is not None:
            self.biases = model_data['biases']
