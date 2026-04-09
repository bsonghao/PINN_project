import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import qmc
import numpy as np


class FBPINN_trainer(object):
    """
    this class implement method to train FBPINN networks
    """
    def __init__(self,
                 problem,
                 subdomains,
                 networks,
                 n_sample,
                 n_batches,
                 domain,):
        """
        problem: object
        object that define the problem specific quantities

        subdomains: dict
        dictionary defines subdomains

        network: nn.Module
        input FBPINN network

        n_sample: int
        number of sampled points at each subdomain

        n_batches: int
        number of batches

        domain: list of tuple
        defines the physics domain
        """
        self.problem = problem
        # initialize subdomain object
        self.subdomains = subdomains

        self.input_dimension = len(domain)
        self.space_dimension = self.input_dimension - 1

        # initialize FBPINN ansatz
        self.approximate_solution = networks

        # train on gpus
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.approximate_solution.to(self.device)


        print(f"full model layout:\n{self.approximate_solution}")
        print(f"number of subnets:{len(networks.subnets)}")


        print("### Initial Model parameters:")
        for name, param in self.approximate_solution.named_parameters():
            print(f"name{name}: {param.size()} Fixed: {not param.requires_grad}")

        self.n_sample = n_sample
        self.n_batches = n_batches

        # Latin hyper cubic generator
        self.LHSeng = qmc.LatinHypercube(d=self.input_dimension, seed=12345)

        # sample data points
        self.domain = torch.zeros((len(domain), 2))
        for i, (a, b) in enumerate(domain):
            self.domain[i, 0], self.domain[i, 1] = a , b
        self.assemble_datasets(self.domain)

    def save_model(self, path="./FBPINN_burger.th"):
        """save the PINN model after training"""
        torch.save(self.approximate_solution.state_dict(), path)
        return

    def load_model(self, path="./FBPINN_burger.th"):
        """load the pre-trained PINN model"""
        # Load the saved state dictionary into the model
        self.approximate_solution.load_state_dict(torch.load(path))
        # Set the model to evaluation mode (important for layers like dropout and batchnorm)
        self.approximate_solution.eval()
        return

    def assemble_datasets(self, domain):
        """
        assemble datasets
        """
        input, output =  self.sample_domain(domain)
        self.training_set = DataLoader(torch.utils.data.TensorDataset(input, output), batch_size=int(self.n_sample/self.n_batches), shuffle=False)

        return

    def convert(self, sequence, domain):
        """rescale the LHS sampled sequence within the domain bounds"""
        # convert domain to a torch tensor
        assert (sequence.shape[1] ==  domain.shape[0])
        return sequence *  (domain[:, 1] - domain[:, 0]) + domain[:,0]

    def sample_domain(self, domain):
        """
        sample the whole domain globally
        """
        sequence = torch.from_numpy(self.LHSeng.random(self.n_sample)).float()
        input = self.convert(sequence, domain)
        output = torch.zeros((input.shape[0], 1))

        return input, output

    def compute_loss(self, training_data, verbose=False):
        """
        calculate PDE loss
        """

        inp_train, u_train = training_data
        inp_train.requires_grad = True
        u = self.approximate_solution(inp_train)

        # get gradients (problem specific)
        u, *u_grad = self.problem.get_gradients(inp_train, u)

        # apply constraint_layer
        u, *u_grad = self.problem.boundary_condition(inp_train, u, *u_grad, sd=1.)

        # compute PDE loss
        PDE_loss = self.problem.physics_loss(inp_train, u, *u_grad)

        if verbose:
            print("PDE loss: ", round(PDE_loss.item(), 4))

        return PDE_loss

    def fit(self, num_epochs, optimizer, verbose=True):
        """train FBPINN globally"""
        history = {"loss": []}
        # loop over epochs
        for epoch in range(num_epochs):
            for j, (inp_train, u_train) in enumerate(self.training_set):
                def closure():
                    optimizer.zero_grad()
                    # compute loss
                    training_data = (inp_train, u_train)
                    PDE_loss = self.compute_loss(training_data, verbose=False)
                    # store loss
                    history["loss"].append(PDE_loss.item())
                    # back propagation
                    PDE_loss.backward()
                    return PDE_loss
                # move the training data to gpu
                inp_train = inp_train.to(self.device)
                u_train = u_train.to(self.device)
                optimizer.step(closure=closure)
                # update weights
                if verbose:
                    tmp = history["loss"][-1]
                    print(f"########### epoch:{epoch}, batch:{j}: loss:{tmp:.4f}")
        # move the model back to cpu after the training
        self.approximate_solution.to("cpu")

        return history

    def predict(self,input_x, active_site):
        """make predictions of the model"""
        input_x.requires_grad = True
        input_x = input_x.to("cpu")
        self.approximate_solution.to("cpu")
        out = self.approximate_solution(input_x, active_site, predict_flag=True)
        out, *out_grad = self.problem.get_gradients(input_x, out)
        out, *_ = self.problem.boundary_condition(input_x, out, *out_grad)
        return out
