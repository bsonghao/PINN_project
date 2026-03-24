import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from scipy.stats import qmc


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
        # Activation function
        activation = nn.Tanh

        self.retrain_seed = retrain_seed

        self.fcs = nn.Sequential(*[nn.Linear(input_dimension, neurons),
                                   activation()])

        self.fch = nn.Sequential(*[
            nn.Sequential(*[
            nn.Linear(neurons, neurons),
            activation()]) for _ in range(n_hidden_layers - 1)])

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

class PINN(object):
    """
    Train PINN to solve Burger's equation
    """
    def __init__(self, n_int, n_sb, n_tb, neurons, n_hidden, domain, n_batches, nu):
        """
        n_int: number of collocation points for physics loss
        n_sb: number of collocation points for spatial boundary
        n_tb: number of collocation for temperal boundary
        neurons: number of neurons per hidden layer
        n_hidden: number of hidden layer
        domain: domain extreme
        n_batches: number of batches
        nu: viscocity (parameter in Burger's equation)
        """
        self.n_int = n_int
        self.n_sb = n_sb
        self.n_tb = n_tb
        self.n_batches = n_batches

        self.nu = nu

        self.domain = domain
        self.input_dimension = self.domain.shape[0]
        self.space_dimension = self.input_dimension-1


        self.approximate_solution = NeuralNet(input_dimension=self.input_dimension, output_dimension=1,
                                             n_hidden_layers=n_hidden,
                                             neurons=neurons)

        # Latin hyper cubic generator
        self.LHSeng = qmc.LatinHypercube(d=self.domain.shape[0])

        # mean square loss function
        self.loss = nn.MSELoss()

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    def save_model(self, path="./PINN_burger.th"):
        """save the PINN model after training"""
        torch.save(self.approximate_solution.state_dict(), path)
        return

    def load_model(self, path="./PINN_burger.th"):
        """load the pre-trained PINN model"""
        # Load the saved state dictionary into the model
        self.approximate_solution.load_state_dict(torch.load(path))
        # Set the model to evaluation mode (important for layers like dropout and batchnorm)
        self.approximate_solution.eval()
        return

    def convert(self, sequence):
        """rescale the LHS sampled sequence within the domain bounds"""
        assert (sequence.shape[1] == self.domain.shape[0])
        return sequence * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:,0]
    def initial_condition(self, x):
        """initial condition: u_0(x) = -sin(pi x)"""
        return -torch.sin(np.pi * x)

    def add_temporal_boundary_points(self):
        """
        get collocation point for temporal boundary condition
        """
        input_tb = torch.zeros((self.n_tb,self.input_dimension))

        t0 = self.domain[0, 0]

        input_tb[:,1] = torch.rand(self.n_tb) * (self.domain[1, 1]-self.domain[1, 0]) + self.domain[1, 0]
        input_tb[:,0] = t0

        output_tb = self.initial_condition(input_tb[:,1]).reshape(-1, 1)

        return input_tb, output_tb

    def add_spatial_boundary_points(self):
        """
        get collocation point for spatial boundary condition
        """
        n_sample = int(self.n_sb/2)
        input_sb_1 = torch.zeros((n_sample, self.input_dimension))
        input_sb_2 = torch.zeros_like(input_sb_1)

        input_sb_1[:,0] = torch.rand(n_sample) * (self.domain[0, 1] - self.domain[0, 0]) + self.domain[0, 0]
        input_sb_1[:,1] = self.domain[1, 0]


        input_sb_2[:,0] = torch.rand(n_sample) * (self.domain[0, 1] - self.domain[0, 0]) + self.domain[0, 0]
        input_sb_2[:,1] = self.domain[1, 1]

        input_sb = torch.cat((input_sb_1, input_sb_2), 0)
        output_sb = torch.zeros((input_sb.shape[0],1))

        return input_sb, output_sb

    def add_interior_points(self):
        """
        get collocation point for physics loss
        """
        input_int = self.convert(torch.from_numpy(self.LHSeng.random(self.n_int)).float())
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    def assemble_datasets(self):

        input_sb, output_sb = self.add_spatial_boundary_points()
        input_tb, output_tb = self.add_temporal_boundary_points()
        input_int, output_int = self.add_interior_points()

        training_set_sb = DataLoader(torch.utils.data.TensorDataset(input_sb, output_sb), batch_size=int(self.space_dimension*self.n_sb/self.n_batches), shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=int(self.n_tb/self.n_batches), shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=int(self.n_int/self.n_batches), shuffle=False)

        return training_set_sb, training_set_tb, training_set_int

    def compute_loss(self, training_data, verbose=True):
        """
        compute loss function: Loss = MSE_sb + MSE_tb + MES_int
        """
        def cal_sb_loss():
            """calculate spatial boundary loss"""
            pred = self.approximate_solution(inp_train_sb)
            assert pred.shape == u_train_sb.shape
            return self.loss(pred, u_train_sb)

        def cal_tb_loss():
            """calculate temporal boundary loss"""
            pred = self.approximate_solution(inp_train_tb)
            assert pred.shape == u_train_tb.shape
            return self.loss(pred, u_train_tb)

        def cal_pde_loss():
            """calculate physics loss"""
            inp_train_int.requires_grad = True
            u = self.approximate_solution(inp_train_int)

            # use autograd to evaluate gradients
            grad_u = torch.autograd.grad(u.sum(), inp_train_int, create_graph=True)[0]
            grad_u_t = grad_u[:,0]
            grad_u_x = grad_u[:,1]
            grad_u_xx = torch.autograd.grad(grad_u_x.sum(), inp_train_int, create_graph=True)[0][: ,1]

            # calcuate the PDE residual
            residual = grad_u_t + u.squeeze()*grad_u_x - self.nu*grad_u_xx
            residual = residual.reshape((residual.shape[0], 1))
            assert residual.shape == u_train_int.shape

            return self.loss(residual, u_train_int)

        inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, u_train_int = training_data

        loss_b = (cal_sb_loss()*self.n_sb + cal_tb_loss()*self.n_tb)/(self.n_sb+self.n_tb)
        loss_p = cal_pde_loss()

        loss = loss_b + loss_p

        if verbose:
            print("Total loss: ", round(loss.item(), 4), "| Physics Loss: ", round(loss_p.item(), 4), "| Boundary Loss: ", round(loss_b.item(), 4))

        return loss, loss_b, loss_p

    def fit(self, num_epochs, optimizer, verbose=True):
        """train PINN"""

        history = {"total loss": [], "physics loss": [], "boundary loss": []}

        # loop over epochs
        for epoch in range(num_epochs):
            for j, ((inp_train_sb, u_train_sb), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb, self.training_set_tb, self.training_set_int)):
                if verbose:
                    print(f"########### epoch:{epoch}, batch:{j}:")
                def closure():
                    optimizer.zero_grad()
                    # compute loss
                    training_data = (inp_train_sb, u_train_sb, inp_train_tb, u_train_tb, inp_train_int, u_train_int)
                    loss, loss_p, loss_b = self.compute_loss(training_data, verbose=verbose)
                    # store loss
                    history["total loss"].append(loss.item())
                    history["physics loss"].append(loss_p.item())
                    history["boundary loss"].append(loss_b.item())
                    # back propagation
                    loss.backward()
                    return loss
                # update weights
                optimizer.step(closure=closure)
                break

        return history
