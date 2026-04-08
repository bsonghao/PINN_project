import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call, stack_module_state
from typing import List, Tuple
import numpy as np
from copy import deepcopy


from models import NeuralNet
from generate_bound import generate_rectangular_bounds, print_subdomains


class norm(nn.Module):
    """
    x_out = (x-x_mean)/x_std
    One scalar pair shared across all subnets
    (can also be per-subnet).
    """
    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def forward(self, X: torch.tensor) -> torch.tensor:
        """
        Map X ∈ [x_min, x_max]  →  [-1, 1]  (per dimension).
        X : (N, D)
        """
        return (X - self.sub["center"]) / self.sub["width"]  * 2.0     # (N, D)


class unnorm(nn.Module):
    """
    y_out = y_nn * scale + shift
    One scalar pair shared across all subnets
    (can also be per-subnet).
    """
    def __init__(self, unnorm_para):
        super(unnorm, self).__init__()
        self.scale, self.shift= unnorm_para


    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.scale+ self.shift        # (N, D)

class window_i(nn.Module):
    """
    window function for FBPINN
    """
    def __init__(self, sub):
        super().__init__()
        self.sub = sub

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Smooth bump window based on a product of 1-D sigmoids:

            w_i(x) = σ(s·(x - x_min)) · σ(s·(x_max - x))

        where s controls steepness (overlap controls effective width).
        Returns shape (N, 1) — scalar weight per point.
        """
        sub = self.sub
        s     = 4.0 / (2*sub["overlap"] * sub["width"] + 1e-8)    # steepness
        left  = torch.sigmoid( s * (X - sub["core_min"]))        # (N, D)
        right = torch.sigmoid( s * (sub["core_max"]- X))        # (N, D)
        w     = (left * right).prod(dim=-1, keepdim=True)  # product over dims → (N,1)
        return w


class constraint_layer(nn.Module):
    """layer the inforce the hard constraints for BC"""
    def __init__(self):
        super(constraint_layer, self).__init__()

    def forward(self, X:torch.tensor, out:torch.tensor) -> torch.tensor:
        """
        Layer that implement hard BC constraint
        f(u) = -sin(pi*x) + tanh(x+1)tanh(x-1)*tanh(t) * NN
        """
        output = -torch.sin(np.pi*X[:,1]).reshape(-1,1)

        factor = torch.tanh(X[:,1] + 1.) * torch.tanh(X[:,1] - 1.) * torch.tanh(X[:,0])


        output +=  factor.reshape(-1, 1) * out

        return output

class FBPINN(nn.Module):
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
        unnorm_para: tuple,
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
            NeuralNet(
                input_dimension  = in_dim,
                output_dimension = out_dim,
                n_hidden_layers  = layers,
                neurons          = hidden,
            )
            for _ in subdomains
        ])

        # window function layer
        self.window = nn.ModuleList([
            window_i(sub)
            for sub in subdomains
        ])

        # unnorm layer  (can be per-subnet: nn.ModuleList of Unnorm)
        self.norm = nn.ModuleList([
              norm(sub)
              for sub in subdomains
        ])

        # unnorm layer
        self.unnorm = unnorm(unnorm_para)

        # constraint layer
        self.constraint_layer = constraint_layer()


    def forward(self, X:torch.Tensor, active_indices, apply_constraint=True) -> torch.Tensor:
        """
        X : (N, D)  — collocation / query points
        Returns (N, out_dim)
        """
        out = torch.zeros(X.shape[0], 1, device=X.device, dtype=X.dtype)

        # Normalised weights
        weights = torch.cat([self.window[i](X) for i in range(self.n_sub)], dim=-1)  # (N, n_sub)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)               # normalise


        # norm = torch.zeros((X.shape[0], 1), device=X.device, dtype=X.dtype)
        for i in range(self.n_sub):
            # norm_i(X)  →  local coords
            x_loc = self.norm[i](X)                        # (N, D)

            # NN_i  forward pass
            y_loc = self.subnets[i](x_loc)                # (N, out_dim)

            # unnorm
            y_glo = self.unnorm(y_loc)                    # (N, out_dim)

            # accumulate weighted contribution
            out +=  weights[:,i].reshape(-1,1) * y_glo                         # (N, out_dim)

        # implement constraint layer
        if apply_constraint:
            out = self.constraint_layer(X, out)

        return out


class FBPINN_fast(nn.Module):
    """
    Finite Basis Physics-Informed Neural Network.

    One NeuralNet per subdomain. Forward pass is parallelized using
    torch.vmap over stacked subdomain parameters.

    Args:
    subdomains : list of Subdomain objects
    in_dim     : spatial/temporal input dimension
    out_dim    : output (solution) dimension
    hidden     : hidden units per subnet
    layers     : depth per subnet
    """

    def __init__(self,
                 subdomains: List[dict],
                 unnorm_para: tuple,
                 in_dim:     int = 1,
                 out_dim:    int = 1,
                 hidden:     int = 32,
                 layers:     int = 3,
                 ):
        super().__init__()

        self.subdomains = subdomains
        self.n_subdomains  = len(subdomains)
        self.output_dim    = out_dim

        # vectorize lower bounds of subdomain
        self.low_core = torch.stack([sub["core_min"] for sub in subdomains], dim=0)  # (n_sub, in_dim)
        self.hi_core  = torch.stack([sub["core_max"] for sub in subdomains], dim=0)  # (n_sub, in_dim)
        self.low_ext  = torch.stack([sub["extended_min"] for sub in subdomains], dim=0)  # (n_sub, in_dim)
        self.hi_ext   = torch.stack([sub["extended_max"] for sub in subdomains], dim=0)  # (n_sub, in_dim)

        # ── Create one NeuralNet per subdomain ──────────────────────
        # Give each a different seed so they start with different weights
        # one small network per subdomain
        self.subnets = nn.ModuleList([
        nn.Sequential(*[
            NeuralNet(
                input_dimension  = in_dim,
                output_dimension = out_dim,
                n_hidden_layers  = layers,
                neurons          = hidden,
            ),
            unnorm(unnorm_para)
            ])
            for _ in subdomains
        ])

        # window function layer
        self.window = nn.ModuleList([
            window_i(sub)
            for sub in subdomains
        ])

        # unnorm layer  (can be per-subnet: nn.ModuleList of Unnorm)
        self.norm = nn.ModuleList([
              norm(sub)
              for sub in subdomains
        ])

        # constraint layer
        self.constraint_layer = constraint_layer()

        # ── Stack parameters for vmap ───────────────────────────────
        # params_stacked[key] has shape (n_subdomains, ...)
        # buffers_stacked[key] has shape (n_subdomains, ...)
        self._stack_params()

        # Keep a reference model (stateless) for functional_call
        self._base_model = deepcopy(self.subnets[0])
        # Zero out the base model parameters (we'll inject via functional_call)
        for p in self._base_model.parameters():
            p.requires_grad_(False)

    # ── Parameter stacking ────────────────────────────────────────────

    def _stack_params(self):
        """
         Stack per-subdomain parameters into batched tensors so vmap can
        process all subdomains in a single vectorized call.

        After stacking:
            self.params  : dict[str -> Tensor(n_subdomains, *param_shape)]
            self.buffers : dict[str -> Tensor(n_subdomains, *buffer_shape)]
        """
        params, buffers = stack_module_state(list(self.subnets))
        # Register stacked params as a ParameterDict so they are
        # trainable and show up in self.parameters()
        self.stacked_params = nn.ParameterDict({
            # '.' is illegal in ParameterDict keys — replace with '__'
            k.replace(".", "__"): nn.Parameter(v)
            for k, v in params.items()
        })
        # Buffers (e.g., BatchNorm stats) — none here but kept for generality
        self._stacked_buffers_raw = buffers

    def _get_params_buffers(self):
        """
        Reconstruct the original key format (with '.') from stacked params.
        """
        params = {
            k.replace("__", "."): v
            for k, v in self.stacked_params.items()
        }
        buffers = self._stacked_buffers_raw   # empty dict for our NeuralNet
        return params, buffers

    # ── Stateless single-subnet forward ──────────────────────────────

    @staticmethod
    def _stateless_forward(params: dict,
                           buffers: dict,
                           base_model: nn.Module,
                           x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate `base_model` on `x` using externally supplied `params`.
        This is the function that vmap will vectorize over.

        Args:
            params      : dict of parameter tensors (un-batched, for ONE subnet)
            buffers     : dict of buffer  tensors  (un-batched, for ONE subnet)
            base_model  : stateless reference model
            x           : input tensor of shape (n_points, input_dim)

        Returns:
            output tensor of shape (n_points, output_dim)
        """
        return functional_call(base_model, (params, buffers), (x,))

    # -- vectorize single domain norm and window function
    @staticmethod
    def window_vectorize(X:torch.Tensor,
                         lo_core:torch.Tensor,
                         hi_core:torch.Tensor,
                         lo_ext:torch.Tensor,
                         hi_ext:torch.Tensor) -> torch.Tensor:
        """
        evaluate window function for a single domain

        Args:
        x        : input tensor
        lo_core  : lower bound of the core domain
        hi_core  : upper bound of the core domain
        lo_ext   : lower bound of the extended domain
        hi_ext   : upper bound of the extended domain
        """
        overlap = torch.max(hi_ext - hi_core, lo_core - lo_ext)
        width = hi_ext - lo_ext
        s     = 4.0 / (2 * overlap * width + 1e-8)    # steepness
        left  = torch.sigmoid( s * (X - lo_core))        # (N, D)
        right = torch.sigmoid( s * (hi_core - X))        # (N, D)
        w     = (left * right).prod(dim=-1, keepdim=True)  # product over dims → (N,1)
        return w

    @staticmethod
    def norm_vectorize(X:torch.Tensor,
                       lo_ext:torch.Tensor,
                       hi_ext:torch.Tensor) -> torch.Tensor:
        """
        normalize input for a single domain

        Args:
        x : input tensor
        low_ext  : lower bound of the extended domain
        hi_ext   : upper bound of the extended domain

        Map X ∈ [x_min, x_max]  →  [-1, 1]  (per dimension).
        X : (N, D)
        """
        center = (lo_ext + hi_ext) / 2.
        width = (hi_ext - lo_ext) / 2.
        return (X - center) / width       # (N, D)
    # ── vmap-powered parallel forward ────────────────────────────────

    def _vmapped_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate ALL subnets on the SAME input x in parallel using vmap.

        Args:
            x   : shape (n_points, input_dim)

        Returns:
            outputs : shape (n_subdomains, n_points, output_dim)
        """
        params, buffers = self._get_params_buffers()
        base_model      = self._base_model

        # vmap over the leading (subdomain) dimension of each param tensor
        # in_dims=0 for params/buffers; None for base_model and x (shared)
        batched_forward = vmap(
            self._stateless_forward,
            in_dims  = (0, 0, None, None),   # batch over params & buffers
            out_dims = 0                      # stack outputs along dim 0
        )

        return batched_forward(params, buffers, base_model, x)
        # shape: (n_subdomains, n_points, output_dim)



    # ── Public forward ────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, active_indices, apply_constraint=True) -> torch.Tensor:
        """
        Full FBPINN forward pass.

        Strategy:
          1. For each subdomain, normalize x to [-1,1] within that subdomain.
          2. Evaluate all subnets in parallel via vmap.
          3. Multiply each subnet output by its window function.
          4. Sum windowed contributions → assembled solution u(x).
          5. add constaints for boundary condition

        Args:
            x : (n_points, input_dim)

        Returns:
            u : (n_points, output_dim)
        """
        # ── Step 1: build normalized inputs for ALL subdomains ────────
        # x_normalized: (n_subdomains, n_points, input_dim)
        x_normalized = vmap(self.norm_vectorize,in_dims=(None, 0, 0),out_dims=0)(x, self.low_ext, self.hi_ext)
        # shape: (n_subdomains, n_points, input_dim)

        # ── Step 2: vmap over subdomains AND their inputs ─────────────
        params, buffers = self._get_params_buffers()
        base_model      = self._base_model

        # Now vmap batches over (params, buffers, x_normalized) simultaneously
        batched_forward = vmap(
            self._stateless_forward,
            in_dims  = (0, 0, None, 0),   # 0=subdomain batch for params, buffers, x_norm
            out_dims = 0
        )

        subnet_outputs = batched_forward(
            params, buffers, base_model, x_normalized
        )
        # shape: (n_subdomains, n_points, output_dim)


        # ── Step 3 & 4: window and sum ────────────────────────────────
        # Normalised weights
        weights = vmap(self.window_vectorize, in_dims=(None, 0, 0, 0, 0), out_dims=0)(x, self.low_core, self.hi_core, self.low_ext, self.hi_ext)
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)  # (n_sub, N, output_dim)
        u = (subnet_outputs * weights).sum(dim=0, keepdim=False) # (n_points, output_dim)

        # ── Step 5: add constraint layer  ────────────────────────────────
        if apply_constraint:
            u = self.constraint_layer(x, u)


        return u




if __name__ == '__main__':
    # ── Build 2*3 block overlapping 2-D subdomains on [0, 1]x[-1, 1]

    subdomains =  generate_rectangular_bounds(
                           dimensions = 2,
                           num_subdomains = [3, 6],
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
    model = FBPINN(
        subdomains=subdomains,
        unnorm_para = (3, 5),
        in_dim=2,
        out_dim=1,
        hidden=32,
        layers=3,
    )
    model_fast = FBPINN_fast(
        subdomains=subdomains,
        unnorm_para = (3, 5),
        in_dim=2,
        out_dim=1,
        hidden=32,
        layers=3,
    )

    # print(model)
    # print(model_fast)
    # os._exit(0)
    # ── create a 2D equal space grid points
    x = np.linspace(0,1,100)
    y = np.linspace(-1, 1, 100)
    x,y = np.meshgrid(x, y)
    inputs = torch.from_numpy(np.stack([x.flatten(), y.flatten()], axis=-1)).float()
    inputs.requires_grad = True
    u_pred = model(inputs, np.arange(18), apply_constraint=True)                               # (N, 1)
    u_pred_fast = model_fast(inputs, np.arange(18), apply_constraint=True)
    print(torch.mean((u_pred-u_pred_fast)**2))
    # assert torch.allclose(weights, weights_fast)
    # assert torch.allclose(u_pred, u_pred_fast)

    # ── Derivative via autograd (needed for PDE residual)
    du_dX = torch.autograd.grad(
        u_pred, inputs,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
    )[0]

    du_dX_fast = torch.autograd.grad(
        u_pred_fast, inputs,
        grad_outputs=torch.ones_like(u_pred_fast),
        create_graph=True,
    )[0]

    print(torch.mean((du_dX-du_dX_fast)**2))

    # assert torch.allclose(du_dX, du_dX_fast)

    print("u_pred shape :", u_pred.shape)           # (200, 1)
    print("du/dX  shape :", du_dX.shape)            # (200, 1)
    print("Params         :", sum(p.numel() for p in model.parameters()))

    # plot window function
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(subdomains), figsize=(60, 8), dpi=150)

    for i, sub in enumerate(subdomains):
        window = window_i(sub)
        w = window(inputs).reshape((100,100))
        img = axs[i].pcolormesh(x, y, w.detach().numpy(), cmap="rainbow")
        axs[i].set_xlabel("t")
        axs[i].set_ylabel("x")
        cbar = fig.colorbar(img, ax=axs[i])


    plt.legend()
    plt.savefig("plot_domain.png")
    plt.show()
