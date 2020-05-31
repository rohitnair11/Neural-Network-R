
sigmoid <- function(x) {
  # Calculate sigmoid of a matrix x
  
  # Inputs:
  # x: a matrix of values
  
  # Output:
  # A matrix with the same size of the input x, where every element x_i is the result of sigmoid(x_i) 
  
  val <- 1 / (1 + exp(-x))
}

sigmoid_derivative <- function(x) {
  # Calculate the derivative of sigmoid function with respect to a matrix x.
  
  # Inputs:
  # x: a matrix of values
  
  # Output:
  # A matrix with the same size of the input x, where every element x_i is the result of the derivative of sigmoid(x_i).
  val <- x * (1 - x)
}

calculate_loss <- function(y_pred, y) {
  # Calculate the loss of predictions and the label.
  
  # Inputs:
  # y_pred: a vector of activations from the last layer of the network.
  # y: a vector of the label of the training samples.
  
  # Output:
  # A number that is the total SSE loss of y_pred and y.
  sum((y - y_pred) ^ 2)
}

calculate_loss_derivative <- function(y_pred, y) {
  # Calculate the loss of predictions and the label. dC/df
  
  # Inputs:
  # y_pred: a vector of activations from the last layer of the network.
  # y: a vector of the label of the training samples.
  
  # Output:
  # A number that is the total SSE loss of y_pred and y.
  -2 * (y - y_pred)
}

calculate_activations <- function(input_matrix, weight_matrix) {
  # Calculate the activations of a layer
  
  # Inputs:
  # input_matrix: a matrix, composed of vectors of inputs. The size of the matrix is (n,m), 
  # where n is the number of samples, and m is the number of the attributes, 
  # or the number of hidden units from last layer.
  # weight_matrix: a matrix, containing the weight for a layer. The dimention of the matrix is (m,q),
  # where q is the number of hidden units for this layer.
  
  # Output:
  # A matrix with the size (n,q), activated by the sigmoid function. 
  sigmoid(input_matrix %*% weight_matrix)
}

calculate_dCdw <- function(in_activations, out_activations, out_dCdf) {
  # Calculate the derivative of loss function with respect to a weight matrix w
  
  # Inputs:
  # in_activations: a matrix of the original input of the layer with weight w.
  # out_activations: a matrix of the original output of the layer with the weight w.
  # out_dCdf: The derivative of the loss function to the out_activations.
  
  # Output:
  # A matrix with the same size of the target matrx w, recording the derivative of loss to w.
  
  val <- t(in_activations) %*% (out_dCdf * sigmoid_derivative(out_activations))
}

calculate_dCdf <- function(weight_matrix, out_activations, out_dCdf) {
  # Calculate the derivative of loss function with respect to an activation output of one layer
  
  # Inputs:
  # weight_matrix: a weight matrix for the current layer
  # out_activations: a matrix of the activation values output from this layer.
  # out_dCdf: The derivative of the loss function to the out_activations of this layer.
  
  # Output:
  # A matrix with the same size of the activation f, recording dC/df_{L-1}, the derivative of loss to 
  # f, the activations of the previous layer.
  val <- (out_dCdf * sigmoid_derivative(out_activations)) %*% t(weight_matrix)
}

neuralnet <- function(x_train, y_train, nodes_layer = 4, n_attributes = 8, learning_rate=0.001, epochs=150) {
  # Implement the neural network.
  
  # Inputs:
  # x_train: The training dataset. A dataframe that has n samples, m attributes.
  # y_train: The labels for training dataset. A dataframe that has n samples, 1 column with the class values.
  # nodes_layer: Integer. In cases of 2-layer neural network, the number of neurons for the first layer is defined here.
  # n_attributes: Integer. Number of attributes.
  # learning_rate: Float. Learning rate of of the neural network.
  # epochs: The number of iterations in training process.
  
  #-------------------------------------------------------------#
  # Data and matrix initialization
  x_train <- as.matrix(x_train)
  y_train <- as.matrix(y_train)
  
  weights_1 <- matrix(rnorm(ncol(x_train) * nodes_layer, mean=0,sd=1), ncol(x_train), nodes_layer)
  weights_2 <- matrix(rnorm(nodes_layer * ncol(y_train), mean=0,sd=1), nodes_layer, ncol(y_train))

  #-------------------------------------------------------------#
  # Training process
  for (i in 1:epochs) {
    #-------------------------------------------------------------#
    # Forward Propagation
    activations_l1 <- calculate_activations(x_train, weights_1)
    activations_op <- calculate_activations(activations_l1, weights_2)
    
    #-------------------------------------------------------------#
    # Calculating training loss
    loss <- calculate_loss(activations_op, y_train)
    dCdf_op <- calculate_loss_derivative(activations_op, y_train)
    
    #-------------------------------------------------------------#
    # Derivative calculation
    gradient_op <- calculate_dCdw(activations_l1, activations_op, dCdf_op)
    
    dCdf_l1 <- calculate_dCdf(weights_2, activations_op, dCdf_op)
    gradient_l1 <- calculate_dCdw(x_train, activations_l1, dCdf_l1)
    
    #-------------------------------------------------------------#
    # Updating weight matrices
    weights_1 <- weights_1 - (learning_rate * gradient_l1)
    weights_2 <- weights_2 - (learning_rate * gradient_op)
    
  }
  #-------------------------------------------------------------#
  # Printing the final training loss
  print(loss)
  
}