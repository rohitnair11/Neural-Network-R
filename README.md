# Neural-Network-R

An Artificial Neural Network (ANN) is created and trained on the `pima-indians-diabetes` dataset using R. Rather than doing backpropagation one instance at a time, the weight updates are calculated for all training instances at once, and summing these to calculate the net change.  
For a given layer in the ANN, a matrix W is used to hold the values of all weights, where W(i,j) is the weight of the edge going from input node i to output node j.  

The following functions are implemented in the code:
* The function `sigmoid` calculates the sigmoid [f(x) = 1/(1+e^-x)] of an input vector x. The function `sigmoid_derivative` calculates the derivative of the sigmoid function with respect to a matrix x.  
* The `calculate_loss` function takes the predicted values and actual values as input and computes the loss of predictions.  
* The `calculate_activations` function is used to calculate the activation vector for a given node, given the input matrix of activations from the prior layer, and the weight matrix. The input is a matrix, where each row corresponds to one training instance and each column corresponds to the activation of a node in the previous layer. The output is a vector, where each item in the vector represents the outputs for one training instance.  
* There are two kinds of gradients calculated to perform back propagation, besides the gradient for the activation function that's already calculated in `sigmoid_derivative()`. In `calculate_dCdw()`, the derivative of the cost with respect to a weight matrix is calculated. In `calculate_dCdf()`, the derivative of the cost with respect to activation values, calculated by `calculate_activations()` is computed.  
* The function `neuralnet` utilizes all the above mentioned function to train the neural network.
