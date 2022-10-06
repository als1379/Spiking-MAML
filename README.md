# Spiking-MAML

In this project, we try to train a meta-learning model with Integrate and Fire neurons instead of RELU activation functions.
IF neurons are not continuous, therefore model would be much more efficient. Although, there are difficulties in updating weights and learning because of non-derivability.

For solving our problem with non-derivability, we have two approaches; **surrogate gradient** and **proxy learning**.

## Surrogate Gradient

In this method, we try to use a substitute function for calculating the gradient. We use ArcTan or Sigmoid for substitution.

## Proxy Learning

In this method, we use two networks. There is a non-spiking neural network (without IF neurons) for calculating gradient vector, and There is a spiking neural network for calculating loss function. Non-spiking networks and spiking networks must have the same size. We will update the weights of our spiking neural network with the gradient vector calculated from our non-spiking network. 
