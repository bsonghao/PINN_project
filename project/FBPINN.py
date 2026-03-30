import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import qmc

from PINN import PINN, NeuralNet
from generate_bound import generate_rectangular_bounds, print_subdomains


def norm_i(X: torch.Tensor, sub: dict) -> torch.Tensor:
    """
    Map X ∈ [x_min, x_max]  →  [0, 1]  (per dimension).
    X : (N, D)
    """
    return (X - sub["center"]) / sub["width"]  + 0.5        # (N, D)



class Unnorm(nn.Module):
    """
    Learnable affine: y_out = scale * y_nn + shift
    One scalar pair shared across all subnets
    (can also be per-subnet).
    """
    def __init__(self, out_dim: int = 1):
        super().__init__()
        #  The scaling parameter is initialized to a tensor of ones, meaning that initially, the output is not scaled.
        #  During training, this parameter will be updated to best fit the training data.
        self.scale = nn.Parameter(torch.ones(out_dim))
        # The shift paramter is initialized to a tensor of zero, meaning that initially, the output is not shifted.
        # During training, this paramter will be updated to best fit the training data.
        self.shift = nn.Parameter(torch.zeros(out_dim))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.scale * y + self.shift


def window_i(X: torch.Tensor, sub: dict) -> torch.Tensor:
    """
    Smooth bump window based on a product of 1-D sigmoids:

        w_i(x) = σ(s·(x - x_min)) · σ(s·(x_max - x))

    where s controls steepness (overlap controls effective width).
    Returns shape (N, 1) — scalar weight per point.
    """
    s     = 4.0 / (2*sub["overlap"] * sub["width"] + 1e-8)    # steepness
    left  = torch.sigmoid( s * (X - sub["core_min"]))        # (N, D)
    right = torch.sigmoid( s * (sub["core_max"]- X))        # (N, D)
    w     = (left * right).prod(dim=-1, keepdim=True)  # product over dims → (N,1)
    return w

def constraint_layer(X, NN):
    """
    Layer that implement hard BC constraint
    f(u) = -sin(pi*x) + tanh(x+1)tanh(x-1)*NN
    """
    output = -torch.sin(np.pi*X[:,1]).reshape(-1,1)

    factor = torch.tanh(X[:,1] + 1.) * torch.tanh(X[:,1] - 1.) * torch.tanh(X[:,0])


    output +=  factor.reshape(-1, 1) * NN
    return output


class FBPINN_ansatz(nn.Module):
    """
    Finite-Basis PINN:

        NN̄(X;θ) = Σᵢ  wᵢ(X) · unnorm( NNᵢ( normᵢ(X); θᵢ ) )

    Parameters
    ----------
    subdomains : list of Subdomain objects
    in_dim     : spatial/temporal input dimension
    out_dim    : output (solution) dimension
    hidden     : hidden units per subnet
    layers     : depth per subnet
    """
    def __init__(
        self,
        subdomains: List[dict],
        in_dim:     int = 1,
        out_dim:    int = 1,
        hidden:     int = 32,
        layers:     int = 3,
    ):
        super().__init__()
        self.subdomains = subdomains
        self.n_sub      = len(subdomains)

        # one small network per subdomain
        self.subnets = nn.ModuleList([
            NeuralNet(in_dim, out_dim, layers, hidden)
            for _ in subdomains
        ])

        # shared unnorm layer  (can be per-subnet: nn.ModuleList of Unnorm)
        self.unnorm = Unnorm(out_dim)

    def forward(self, X:torch.Tensor, active_indices: List, apply_constraint=True) -> torch.Tensor:
        """
        X : (N, D)  — collocation / query points
        Returns (N, out_dim)
        """
        out = torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)

        # Normalised weights
        weights = torch.cat([window_i(X, self.subdomains[i]) for i in active_indices], dim=-1)  # (N, n_sub)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)               # normalise


        # norm = torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)
        for i, site in enumerate(active_indices):
            sub  = self.subdomains[i]
            # norm_i(X)  →  local coords
            x_loc = norm_i(X, sub)                        # (N, D)

            # NN_i  forward pass
            y_loc = self.subnets[site](x_loc)                # (N, out_dim)

            # unnorm
            y_glo = self.unnorm(y_loc)                    # (N, out_dim)

            # accumulate weighted contribution
            out +=  weights[:,i].reshape(-1,1) * y_glo                         # (N, out_dim)

        # implement constraint layer
        if apply_constraint:
            out = constraint_layer(X, out)

        return out



class FBPINN(object):
    """
    this class implement method to train FBPINN networks
    """
    def __init__(self,
                 subdomains,
                 networks,
                 n_sample,
                 n_batches,
                 nu,
                 input_dimension,
                 activation_interval = 0,
                 fixed_after_epochs  = 0,
                 convergence_tol     = 0,
                 convergence_window  = 0,
                 num_per_column      = 0,
                 manager=None,
                 scheduler=None,
                 domain=torch.tensor([[0., 1.],[-1., 1.]])):
        """
        bounds: bounds of subdomains
        overlap: range of overlap in each subdomain
        n_sample: number of sampled points at each subdomain
        neurons: number of neurons at each hidden layer
        n_hidden: number of n_hidden layers
        nu: viscocity paramter for Burger's equation
        """
        # initialize subdomain object
        self.subdomains = subdomains

        self.input_dimension = input_dimension
        self.space_dimension = self.input_dimension - 1

        # initialize FBPINN ansatz
        self.approximate_solution = networks
        # self.approximate_solution =  FBPINN_ansatz(
                                     # subdomains=self.subdomains,
                                     # in_dim=self.input_dimension,
                                     # out_dim=1,
                                     # hidden=neurons,
                                     # layers=n_hidden,
                                 # )
        print("### Initial Model parameters:")
        # initially fixed all subnets parameters
        for idx in range(len(self.subdomains)):
            for param in self.approximate_solution.subnets[idx].parameters():
                param.requires_grad = False
        for name, param in self.approximate_solution.named_parameters():
            print(f"name{name}: {param.size()} Fixed: {not param.requires_grad}")

        self.n_sample = n_sample
        self.n_batches = n_batches

        # Latin hyper cubic generator
        self.LHSeng = qmc.LatinHypercube(d=self.input_dimension)

        # mean square loss function
        self.loss = nn.MSELoss()

        self.assemble_datasets(domain)

        self.nu = nu # viscocity (model parameter)

        if manager is not None:
            self.manager = manager(len(self.approximate_solution.subnets), self.approximate_solution)
        if scheduler is not None:
            self.scheduler = scheduler(self.manager,
                                      activation_interval,
                                      fixed_after_epochs,
                                      convergence_tol,
                                      convergence_window,
                                      num_per_column  )

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
        # input, output =  self.sample_subdomain()
        input, output =  self.sample_domain(torch.tensor([[0, 1],[-1, 1]]))
        self.training_set = DataLoader(torch.utils.data.TensorDataset(input, output), batch_size=int(self.n_sample/self.n_batches), shuffle=False)

        return

    def convert(self, sequence, domain):
        """rescale the LHS sampled sequence within the domain bounds"""
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

    def assemble_subdomain_data(self, active_site, tol=1e-10, verbose=False):
        """
        collect the collocation points within the subdomain of each the active sites (from the scheduler)
        """
        global_input, global_output = next(iter(self.training_set))

        local_data = {}
        mask_all = torch.zeros(self.n_sample, dtype=bool)

        for idx in active_site:
            mask = torch.ones(self.n_sample, dtype=bool)
            sub = self.subdomains[idx]
            for dim in range(self.input_dimension):
                mask &= global_input[:,dim] <= sub["extended_max"][dim]+tol # bitwise and operation
                mask &= global_input[:,dim] >= sub["extended_min"][dim]-tol # bitwise and operation
            mask_all |= mask # bitwise or operation

        active_indices = torch.where(mask_all)
        input  = global_input[active_indices]
        output = torch.zeros((input.shape[0], 1))

        if verbose:
            print(f"training for subdomain {active_site}: {input.shape}")

        return input, output

    def build_subdomain_optimizer(self, optimizer, lr=1e-3):
        """
        Build optimizer that only includes parameters of ACTIVE subdomains.
        Re-called whenever active set changes.
        """
        manager, scheduler = self.manager, self.scheduler
        active_params = []
        for idx in manager.get_active_indices():
            active_params += list(self.approximate_solution.subnets[idx].parameters())
        active_params += self.approximate_solution.unnorm.parameters()

        if not active_params:
            return None

        return optimizer(active_params, lr=lr)

    def fit_subdomain(self, num_epochs, optimizer, verbose=True):
        """
        Full FBPINN training loop with dynamic active/inactive/fixed scheduling.
        """
        if verbose:
            print("\n" + "═" * 60)
            print("  FBPINN Training with Dynamic Subdomain Scheduling")
            print("═" * 60 + "\n")

        # loss_subdomain = []
        global_loss = {"loss": []}

        manager = self.manager
        scheduler = self.scheduler

        for epoch in range(num_epochs):

            # ── Rebuild optimizer for current active set ─────────────────────
            subdomain_optimizer = self.build_subdomain_optimizer(optimizer, lr=1e-3)


            active_indices = manager.get_active_indices()
            # print(active_indices)

            if not active_indices or optimizer is None:
                scheduler.step(epoch, {})
                continue

            training_data = self.assemble_subdomain_data(active_indices)
            # ── Forward pass & loss for each active subdomain ─────────────────
            total_loss  = 0.0
            subdomain_optimizer.zero_grad()
            losses_dict = {}
            loss = self.compute_loss(training_data, active_indices)
            for idx in active_indices:
                losses_dict[idx] = loss.item()
            total_loss = loss


            # ── Backward pass ─────────────────────────────────────────────────
            total_loss.backward()
            subdomain_optimizer.step()

            # store loss data
            global_loss["loss"].append(total_loss.item())

            # ── Update scheduler ──────────────────────────────────────────────
            scheduler.step(epoch, losses_dict)

            # ── Logging ───────────────────────────────────────────────────────
            if verbose:
                print(f"  Epoch {epoch:>4d} | "
                      f"Total Loss: {total_loss.item():.6f} | "
                      f"Active: {manager.get_active_indices()} | "
                      f"Fixed:  {manager.get_fixed_indices()}")
                # for site,value in losses_dict.items():
                    # print(f"subdomain:{site}, loss:{value}")
                # for name, param in self.approximate_solution.named_parameters():
                    # if param.requires_grad:
                        # print(f"active parameter: {name}")
                    # print(f"Parameter: {name}, Fixed: {not param.requires_grad}")
            # os._exit(0)

        print("\n  Training Complete!")
        manager.print_states()
        return global_loss


    def compute_loss(self, training_data, active_indices, verbose=False):
        """
        calculate PDE loss
        """

        inp_train, u_train = training_data
        inp_train.requires_grad = True
        u = self.approximate_solution(inp_train, active_indices)

        # use autograd to evaluate gradients
        grad_u = torch.autograd.grad(u.sum(), inp_train, create_graph=True)[0]
        grad_u_t = grad_u[:,0]
        grad_u_x = grad_u[:,1]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), inp_train, create_graph=True)[0][: ,1]

        # calcuate the PDE residual
        residual = grad_u_t + u.squeeze()*grad_u_x - self.nu*grad_u_xx
        residual = residual.reshape((residual.shape[0], 1))

        assert residual.shape == u_train.shape

        PDE_loss = self.loss(residual, u_train)

        if verbose:
            print("PDE loss: ", round(PDE_loss.item(), 4))

        return PDE_loss

    def fit(self, num_epochs, optimizer, verbose=True):
        """train FBPINN globally"""

        history = {"loss": []}
        active_indices = np.arange(len(self.subdomains))

        # loop over epochs
        for epoch in range(num_epochs):
            for j, (inp_train, u_train) in enumerate(self.training_set):
                if verbose:
                    print(f"########### epoch:{epoch}, batch:{j}:")
                def closure():
                    optimizer.zero_grad()
                    # compute loss
                    training_data = (inp_train, u_train)
                    PDE_loss = self.compute_loss(training_data, active_indices, verbose=verbose)
                    # store loss
                    history["loss"].append(PDE_loss.item())
                    # back propagation
                    PDE_loss.backward()
                    return PDE_loss
                # update weights
                optimizer.step(closure=closure)

        # store loss function data to json file
        import pandas as pd
        df = pd.DataFrame(history)
        df.sort_index().to_json()
        # df.to_json("FBPINN_loss_function.json")

        return history


if __name__ == '__main__':
    # ── Build 2*3 block overlapping 2-D subdomains on [0, 1]x[-1, 1]

    subdomains =  generate_rectangular_bounds(
                           dimensions = 2,
                           num_subdomains = [2, 3],
                           ranges = [(0, 1), (-1 ,1)],
                           overlap = 0.2,
                           overlap_mode = "absolute")

    print_subdomains(subdomains, dimensions=2)

    # subdomains = [
        # Subdomain(
            # x_min=torch.tensor([lo]),
            # x_max=torch.tensor([hi]),
            # overlap=0.2,
        # )
        # for lo, hi in bounds
    # ]

    model = FBPINN_ansatz(
        subdomains=subdomains,
        in_dim=2,
        out_dim=1,
        hidden=32,
        layers=3,
    )

    print(model)


    # ── create a 2D equal space grid points
    x = np.linspace(0,1,100)
    y = np.linspace(-1, 1, 100)
    x,y = np.meshgrid(x, y)
    inputs = torch.from_numpy(np.stack([x.flatten(), y.flatten()], axis=-1)).float()
    inputs.requires_grad = True
    u_pred = model(inputs)                               # (N, 1)

    # ── Derivative via autograd (needed for PDE residual)
    du_dX = torch.autograd.grad(
        u_pred, inputs,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]

    print("u_pred shape :", u_pred.shape)           # (200, 1)
    print("du/dX  shape :", du_dX.shape)            # (200, 1)
    print("Params         :", sum(p.numel() for p in model.parameters()))

    # plot window function
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 6, figsize=(60, 8), dpi=150)

    for i, sub in enumerate(subdomains):
        w = window_i(inputs, sub).reshape((100,100))
        img = axs[i].pcolormesh(x, y, w.detach().numpy(), cmap="rainbow")
        cbar = fig.colorbar(img, ax=axs[i])


    plt.legend()
    plt.savefig("plot_domain.png")
    plt.show()
