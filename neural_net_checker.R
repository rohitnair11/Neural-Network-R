
rm(list = ls(all = T))
source('./neural_net.R')
# read data from disk, extract train test into separate variables 
all_data <- read.csv('./data/pima-indians-diabetes.csv', stringsAsFactors= T, header = F)
x <- all_data[, 1:8]
y <- all_data[, 9]

set.seed(155)
x_check = matrix( rnorm(9, mean=0,sd=1), 3,3)
f_check = matrix( rnorm(9, mean=0,sd=1), 3,3)
w_check = matrix( rnorm(9, mean=0,sd=1), 3,3)
y_check = matrix( rnorm(9, mean=0,sd=1), 3,3)
y_pred_check = matrix( rnorm(9, mean=0,sd=1), 3,3)

# Sigmoid Implementation
print(sum(sigmoid(x_check)))
# Should be 4.872344
print(sum(sigmoid_derivative(x_check)))
# Should be -3.703403

# Activation Calculation
print(sum(calculate_activations(x_check, w_check)))
# Should be 3.391088

# Loss Calculation 
print(calculate_loss(y_check, y_pred_check))
# Should be 17.56477

# Gradient Calculation
print(sum(calculate_dCdw(f_check,f_check,f_check)))
# Should be -5.867997
print(sum(calculate_dCdf(f_check,f_check,f_check))) # Only required for 2-layer network.
# Should be -15.18125

# Network Training
neuralnet(x,y)
# Should be 22.69416 for 2-layer network; 37.00013 for 1-layer network