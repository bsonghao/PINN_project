from enum import Enum
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from PINN import NeuralNet

# ─────────────────────────────────────────────────────────────
# Subdomain State Enum
# ─────────────────────────────────────────────────────────────
class SubdomainState(Enum):
    INACTIVE = "inactive"   # Not yet participating in training
    ACTIVE   = "active"     # Currently being trained
    FIXED    = "fixed"      # Converged and frozen

# ─────────────────────────────────────────────────────────────
# Subdomain State Manager
# ─────────────────────────────────────────────────────────────
class SubdomainStateManager:
    """
    Manages the Active / Inactive / Fixed state for each subdomain.
    Controls parameter freezing and gradient flow accordingly.
    """

    def __init__(self, num_subdomains: int, model:nn.Module):
        self.num_subdomains = num_subdomains
        self.model          = model
        # self.subnets        = model.subnets

        # All subdomains start as INACTIVE
        self.states: List[SubdomainState] = [
            SubdomainState.INACTIVE for _ in range(num_subdomains)
        ]

        # Track loss history per subdomain for convergence detection
        self.loss_history: Dict[int, List[float]] = {
            i: [] for i in range(num_subdomains)
        }

    # ── State Transitions ──────────────────────────────────────────────────

    def activate(self, subdomain_idx: int):
        """Move a subdomain from INACTIVE → ACTIVE."""
        if self.states[subdomain_idx] == SubdomainState.INACTIVE:
            self.states[subdomain_idx] = SubdomainState.ACTIVE
            self._unfreeze(subdomain_idx)
            print(f"  [Scheduler] Subdomain {subdomain_idx}: INACTIVE → ACTIVE")

    def fix(self, subdomain_idx: int):
        """Move a subdomain from ACTIVE → FIXED (freeze its weights)."""
        if self.states[subdomain_idx] == SubdomainState.ACTIVE:
            self.states[subdomain_idx] = SubdomainState.FIXED
            self._freeze(subdomain_idx)
            print(f"  [Scheduler] Subdomain {subdomain_idx}: ACTIVE → FIXED")

    def reactivate(self, subdomain_idx: int):
        """Move a subdomain from FIXED → ACTIVE (unfreeze for fine-tuning)."""
        if self.states[subdomain_idx] == SubdomainState.FIXED:
            self.states[subdomain_idx] = SubdomainState.ACTIVE
            self._unfreeze(subdomain_idx)
            print(f"  [Scheduler] Subdomain {subdomain_idx}: FIXED → ACTIVE")

    # ── Parameter Freezing ─────────────────────────────────────────────────

    def _freeze(self, subdomain_idx: int):
        """Freeze all parameters of a subdomain network."""
        for param in self.model.subnets[subdomain_idx].parameters():
            param.requires_grad = False

    def _unfreeze(self, subdomain_idx: int):
        """Unfreeze all parameters of a subdomain network."""
        for param in self.model.subnets[subdomain_idx].parameters():
            param.requires_grad = True
    # ── Getters ────────────────────────────────────────────────────────────

    def get_active_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.states)
                if s == SubdomainState.ACTIVE]

    def get_inactive_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.states)
                if s == SubdomainState.INACTIVE]

    def get_fixed_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.states)
                if s == SubdomainState.FIXED]

    def record_loss(self, subdomain_idx: int, loss_val: float):
        self.loss_history[subdomain_idx].append(loss_val)

    def print_states(self):
        print(f"\n  {'─'*50}")
        print(f"  {'Subdomain':<12} {'State':<12} {'Loss History (last 3)'}")
        print(f"  {'─'*50}")
        for i, state in enumerate(self.states):
            recent = self.loss_history[i][-3:]
            recent_str = [f"{l:.4f}" for l in recent]
            print(f"  {i:<12} {state.value:<12} {recent_str}")
        print(f"  {'─'*50}\n")


# ─────────────────────────────────────────────────────────────
# Base Scheduler
# ─────────────────────────────────────────────────────────────
class BaseScheduler:
    """Base class for all FBPINN training schedulers."""

    def __init__(self, manager: SubdomainStateManager):
        self.manager = manager

    def step(self, epoch: int, losses: Dict[int, float]):
        """Called every epoch to update states. Override in subclasses."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# Strategy 1: Sequential Scheduler
# Activates subdomains one by one, left to right.
# ─────────────────────────────────────────────────────────────
class SequentialScheduler(BaseScheduler):
    """
    Activates subdomains sequentially, one per `activation_interval` epochs.
    Useful for problems with a clear spatial/temporal ordering.
    """

    def __init__(
        self,
        manager: SubdomainStateManager,
        activation_interval: int = 20, # specifies the number of epochs between the activation of successive subdomains
        fixed_after_epochs: int    = 60, # defines the number of epochs that must be completed after a subdomain is activated before it can be fixed
        convergence_tol: float   = 1e-4, # parameter represents the tolerance level for determining convergence of the loss function.
        convergence_window: int  = 5 # parameter specifies the number of most recent epochs to consider when evaluating convergence.
    ):
        super().__init__(manager)
        self.activation_interval = activation_interval
        self.fixed_after_epochs    = fixed_after_epochs
        self.convergence_tol     = convergence_tol
        self.convergence_window  = convergence_window
        self.activation_epoch    = {}  # Track when each subdomain was activated

    def step(self, epoch: int, losses: Dict[int, float]):
        m = self.manager

        # Record losses for active subdomains
        for idx in m.get_active_indices():
            if idx in losses:
                m.record_loss(idx, losses[idx])

        # Activate the next subdomain at regular intervals
        inactive = m.get_inactive_indices()
        if inactive and (epoch % self.activation_interval == 0):
            next_idx = inactive[0]
            m.activate(next_idx)
            self.activation_epoch[next_idx] = epoch

        # Check active subdomains for convergence → fix them
        for idx in m.get_active_indices():
            if self._has_converged(idx) or self._max_epochs_reached(idx, epoch):
                m.fix(idx)

    def _has_converged(self, idx: int) -> bool:
        history = self.manager.loss_history[idx]
        if len(history) < self.convergence_window:
            return False
        recent = history[-self.convergence_window:]
        return (max(recent) - min(recent)) < self.convergence_tol

    def _max_epochs_reached(self, idx: int, current_epoch: int) -> bool:
        activated_at = self.activation_epoch.get(idx, current_epoch)
        return (current_epoch - activated_at) >= self.fixed_after_epochs

# ─────────────────────────────────────────────────────────────
# Strategy 2: column Scheduler
# Activates subdomains column by column, left to right.
# ─────────────────────────────────────────────────────────────
class ColumnScheduler(BaseScheduler):
    """
    Activates subdomains sequentially, one per `activation_interval` epochs.
    Useful for problems with a clear spatial/temporal ordering.
    """

    def __init__(
        self,
        manager: SubdomainStateManager,
        activation_interval: int = 20, # specifies the number of epochs between the activation of successive subdomains
        fixed_after_epochs: int    = 60, # defines the number of epochs that must be completed after a subdomain is activated before it can be fixed
        convergence_tol: float   = 1e-4, # parameter represents the tolerance level for determining convergence of the loss function.
        convergence_window: int  = 5, # parameter specifies the number of most recent epochs to consider when evaluating convergence.
        num_per_column: int = 1 # number of subdomains per column
    ):
        super().__init__(manager)
        self.activation_interval = activation_interval
        self.fixed_after_epochs    = fixed_after_epochs
        self.convergence_tol     = convergence_tol
        self.convergence_window  = convergence_window
        self.activation_epoch    = {}  # Track when each subdomain was activated
        self.num_per_column = num_per_column

    def step(self, epoch: int, losses: Dict[int, float]):
        m = self.manager
        num_per_column = self.num_per_column

        # Record losses for active subdomains
        for idx in m.get_active_indices():
            if idx in losses:
                m.record_loss(idx, losses[idx])

        # Activate the next subdomain at regular intervals
        inactive = m.get_inactive_indices()
        if inactive and (epoch % self.activation_interval == 0):
            next_idx = inactive[:num_per_column]
            for i in range(num_per_column):
                m.activate(next_idx[i])
                self.activation_epoch[next_idx[i]] = epoch

        # Check active subdomains for convergence → fix them
        for idx in m.get_active_indices():
            if self._has_converged(idx) or self._max_epochs_reached(idx, epoch):
                m.fix(idx)

    def _has_converged(self, idx: int) -> bool:
        history = self.manager.loss_history[idx]
        if len(history) < self.convergence_window:
            return False
        recent = history[-self.convergence_window:]
        return (max(recent) - min(recent)) < self.convergence_tol

    def _max_epochs_reached(self, idx: int, current_epoch: int) -> bool:
        activated_at = self.activation_epoch.get(idx, current_epoch)
        return (current_epoch - activated_at) >= self.fixed_after_epochs





if __name__ == "__main__":

    def build_fbpinn_optimizer(
        subnets: List[nn.Module],
        manager: SubdomainStateManager,
        lr: float = 1e-3
    ) -> optim.Optimizer:
        """
        Build optimizer that only includes parameters of ACTIVE subdomains.
        Re-called whenever active set changes.
        """
        active_params = []
        for idx in manager.get_active_indices():
            active_params += list(subnets[idx].parameters())

        if not active_params:
            return None

        return optim.Adam(active_params, lr=lr)


    def compute_subdomain_loss(
        subnet: nn.Module,
        x_collocation: torch.Tensor,
        x_boundary: torch.Tensor,
        u_boundary: torch.Tensor,
        physics_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Compute combined physics + boundary loss for a single subdomain.
        """
        # ── Physics loss (residual) ──────────────────────────────────────────
        x_phys = x_collocation.clone().requires_grad_(True)
        u_pred = subnet(x_phys)
        du_dx  = torch.autograd.grad(
            u_pred, x_phys,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0]
        # Example PDE residual: du/dx - f(x) = 0
        residual     = du_dx - torch.sin(x_phys)
        physics_loss = (residual ** 2).mean()

        # ── Boundary loss ────────────────────────────────────────────────────
        u_bc      = subnet(x_boundary)
        bc_loss   = ((u_bc - u_boundary) ** 2).mean()

        return bc_loss + physics_weight * physics_loss


    def train_fbpinn(
        subnets: List[nn.Module],
        manager: SubdomainStateManager,
        scheduler: BaseScheduler,
        subdomains_data: List[Dict],
        num_epochs: int = 200,
        lr: float       = 1e-3,
        log_interval: int = 10
    ):
        """
        Full FBPINN training loop with dynamic active/inactive/fixed scheduling.
        """
        print("\n" + "═" * 60)
        print("  FBPINN Training with Dynamic Subdomain Scheduling")
        print("═" * 60 + "\n")

        for epoch in range(num_epochs):

            # ── Rebuild optimizer for current active set ─────────────────────
            optimizer = build_fbpinn_optimizer(subnets, manager, lr=lr)

            active_indices = manager.get_active_indices()

            if not active_indices or optimizer is None:
                scheduler.step(epoch, {})
                continue

            # ── Forward pass & loss for each active subdomain ─────────────────
            optimizer.zero_grad()
            total_loss  = 0.0
            losses_dict = {}

            for idx in active_indices:
                data     = subdomains_data[idx]
                x_phys   = data["x_collocation"]
                x_bc     = data["x_boundary"]
                u_bc     = data["u_boundary"]

                loss = compute_subdomain_loss(
                    subnets[idx], x_phys, x_bc, u_bc
                )
                losses_dict[idx] = loss.item()
                total_loss      += loss

            # ── Backward pass ─────────────────────────────────────────────────
            total_loss.backward()
            optimizer.step()

            # ── Update scheduler ──────────────────────────────────────────────
            scheduler.step(epoch, losses_dict)

            # ── Logging ───────────────────────────────────────────────────────
            if epoch % log_interval == 0:
                print(f"  Epoch {epoch:>4d} | "
                      f"Total Loss: {total_loss.item():.6f} | "
                      f"Active: {manager.get_active_indices()} | "
                      f"Fixed:  {manager.get_fixed_indices()}")

        print("\n  Training Complete!")
        manager.print_states()

    N = 4  # Number of subdomains

    # ── Build subnets ──────────────────────────────────────────────────────
    subnets = [NeuralNet(input_dimension=1, output_dimension=1, n_hidden_layers=1, neurons=20) for _ in range(N)]

    # ── Dummy subdomain data ───────────────────────────────────────────────
    subdomains_data = []
    for i in range(N):
        lo = i * 0.25
        hi = lo + 0.25
        subdomains_data.append({
            "x_collocation": torch.linspace(lo, hi, 50).unsqueeze(1),
            "x_boundary"   : torch.tensor([[lo], [hi]]),
            "u_boundary"   : torch.zeros(2, 1),
        })

    # ── Choose a Scheduler ─────────────────────────────────────────────────
    manager   = SubdomainStateManager(N, subnets)

    # Swap any of these three strategies:
    # scheduler = SequentialScheduler(
        # manager,
        # activation_interval = 20,
        # fixed_after_epochs    = 50,
        # convergence_tol     = 1e-4
    # )

    scheduler = ColumnScheduler(
        manager,
        activation_interval = 20,
        fixed_after_epochs    = 50,
        convergence_tol     = 1e-4,
        num_per_column      = 1
    )


    manager = scheduler.manager

    # scheduler = ParallelScheduler(manager, convergence_tol=1e-4, min_epochs=30)
    # scheduler = WaveScheduler(manager, wave_size=2, convergence_tol=1e-4)

    # ── Train ──────────────────────────────────────────────────────────────
    train_fbpinn(
        subnets, manager, scheduler,
        subdomains_data,
        num_epochs   = 200,
        lr           = 1e-3,
        log_interval = 10
    )
