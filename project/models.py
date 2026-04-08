import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    define a feedforward Neural Network
    """
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, retrain_seed=42):
        super(NeuralNet, self).__init__()
        """
        initialize Neural network
        input_dimension: Number of input dimensions
        ouput_dimension: Number of output dimensions
        neurons: Number of hidden layer dimension
        retrain_seed: random seed for initial weights
        """
        # Activation function (hyperbolic tangent)
        activation = nn.Tanh

        self.retrain_seed = retrain_seed

        self.fcs = nn.Sequential(*[nn.Linear(input_dimension, neurons),
                                   activation()])

        self.fch = nn.Sequential(*[
            nn.Sequential(*[
            nn.Linear(neurons, neurons),
            activation()]) for _ in range(n_hidden_layers)])

        self.fce = nn.Linear(neurons, output_dimension)

        # initialize weight with random seed
        self.init_xavier()

    def forward(self, x):
        """
        The forward function performs the set of affine and non-linear transformations defining the network
        """
        return self.fce(self.fch(self.fcs(x)))

    def init_xavier(self,distribution="uniform"):
        """
        initializing the weights of a PyTorch neural network using Xavier initialization
        """
        # set the random seed for generating random number to ensure that the results are repruducible
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            """
            m: a layer or a module in the neural network
            """
            # checks if the module m is an instance of nn.Linear (a linear layer).
            # It also checks if the layer's weights and biases require gradients
            # meaning they will be updated during training.
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                # The gain is calculated for the 'tanh' activation function.
                # The gain is a scaling factor that is used to keep the variance of the outputs of each layer in a neural network under control.
                # Different activation functions require different gains.
                g = nn.init.calculate_gain("tanh")
                # the weights of the linear layer m are initialized using the Xavier uniform distribution, scaled by the calculated gain g.
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(m.weight, gain=g)
                elif distribution == "normal":
                    torch.nn.init.xavier_normal_(m.weight, gain=g)
                else:
                    print("Warning: the input distribution is not implemented! use default distrubtion instead")
                    torch.nn.init.xavier_uniform_(m.weight, gain=g)

                # initialize biases to zero
                m.bias.data.fill_(0)
            return

        # the init_weigts function is applied to all the modules (layers) in the neural network using the self.apply() method
        self.apply(init_weights)

        return
