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
        problem: object that define the problem specific quantities
        subdomains: object defines subdomains
        network: input FBPINN network
        n_sample: number of sampled points at each subdomain
        n_batches: number of batches
        """
        self.problem = problem
        # initialize subdomain object
        self.subdomains = subdomains

        self.input_dimension = input_dimension
        self.space_dimension = self.input_dimension - 1

        # initialize FBPINN ansatz
        self.approximate_solution = networks

        # train on gpus
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.approximate_solution.to(self.device)


        print(f"full model layout:\n{self.approximate_solution}")
        print(f"number of subnets:{len(networks.subnets)}")

        # os._exit(0)
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
        self.LHSeng = qmc.LatinHypercube(d=self.input_dimension, seed=12345)

        # mean square loss function
        self.loss = nn.MSELoss()
        # sample data points
        self.assemble_datasets(domain)

        # self.nu = nu # viscocity (model parameter)

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

    def assemble_subdomain_data(self, active_sites, tol=1e-10, verbose=False):
        """
        collect the collocation points within the subdomain of each the active sites (from the scheduler)
        """

        global_input, global_output = next(iter(self.training_set))

        mask_all = torch.zeros(self.n_sample, dtype=bool)

        for idx in active_sites:
            mask = torch.ones(self.n_sample, dtype=bool)
            sub = self.subdomains[idx]
            for dim in range(self.input_dimension):
                mask &= global_input[:,dim] <= sub["extended_max"][dim]+tol # bitwise and operation
                mask &= global_input[:,dim] >= sub["extended_min"][dim]-tol # bitwise and operation
            mask_all |= mask

        active_indices = torch.where(mask_all)
        input  = global_input[active_indices]
        output = torch.zeros((input.shape[0], 1))
        return input, output

    def build_subdomain_optimizer(self, optimizer, lr=1e-4):
        """
        Build optimizer that only includes parameters of ACTIVE subdomains.
        Re-called whenever active set changes.
        """
        manager, scheduler = self.manager, self.scheduler
        active_params = []
        for idx in manager.get_active_indices():
            active_params += list(self.approximate_solution.subnets[idx].parameters())

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

        # turn to taining mode
        self.approximate_solution.train()
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
        # active all parameters
        for idx in range(len(self.subdomains)):
            for param in self.approximate_solution.subnets[idx].parameters():
                param.requires_grad = True

        history = {"loss": []}
        active_indices = np.arange(len(self.subdomains))

        # loop over epochs
        for epoch in range(num_epochs):
            for j, (inp_train, u_train) in enumerate(self.training_set):
                def closure():
                    optimizer.zero_grad()
                    # compute loss
                    training_data = (inp_train, u_train)
                    PDE_loss = self.compute_loss(training_data, active_indices, verbose=False)
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
