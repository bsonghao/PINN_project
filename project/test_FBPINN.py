import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from FBPINN import FBPINN, FBPINN_ansatz
from exact_solution import exact_solution
from generate_bound import generate_rectangular_bounds, print_subdomains
from plot_result import plotting_FBPINN
from schedule import SequentialScheduler, ColumnScheduler, SubdomainStateManager

def main():
    # input parameter for PINN

    # collocation parameters
    n_sample = 60000
    input_dimension = 2

    # NN parameters
    neurons, n_hidden = 16, 2 # 16 neurons per hidden layer and 2 hidden layers (4 layers in total)

    # define physics domain
    domain = [(0., 1.), (-1., 1.)] # time domain x space domain
    # parameters to define subdomains
    num_grid = [3, 2]
    overlap = 0.2

    # parameters for scheduler
    activation_interval = 20000
    fixed_after_epochs  = 40000
    convergence_tol     = 1e-8
    convergence_window  = 5
    num_per_column      = num_grid[0]
    # ── Build 2*3 block overlapping 2-D subdomains on [0, 1]x[-1, 1]

    subdomains =  generate_rectangular_bounds(
                           dimensions = 2,
                           num_subdomains = num_grid,
                           ranges = domain,
                           overlap = overlap,
                           overlap_mode = "absolute")

    print_subdomains(subdomains, dimensions=2)

    # os._exit(0)

    # build FBPINN ansatz
    network = FBPINN_ansatz(
                          subdomains = subdomains,
                          in_dim=2,
                          out_dim=1,
                          hidden = neurons,
                          layers = n_hidden,)

    # setup scheculer
    # ── Choose a Scheduler ─────────────────────────────────────────────────
    # manager   = SubdomainStateManager(len(subdomains), network.subnets)

    # Swap any of these three strategies:
    # scheduler = SequentialScheduler(
        # manager,
        # activation_interval = 2,
        # fix_after_epochs    = 2,
        # convergence_tol     = 1e-4
    # )

    # scheduler = ColumnScheduler(
            # manager,
            # activation_interval = 20,
            # fix_after_epochs    = 40,
            # convergence_tol     = 1e-8,
            # num_per_column      = num_grid[1]
        # )

    # viscosity parameter
    nu = 0.01 / np.pi

    # parameters for training
    # batch parameters
    n_batches = 1
    # number of epochs
    n_epochs = 1000


    # FBPINN training
    model = FBPINN(subdomains,
                   network,
                   n_sample,
                   n_batches,
                   nu,
                   input_dimension,
                   activation_interval,
                   fixed_after_epochs,
                   convergence_tol,
                   convergence_window,
                   num_per_column,
                   SubdomainStateManager,
                   ColumnScheduler)

    # os._exit(0)

    if True:
        # LBFGS optimizer
        # optimizer_LBFGS = optim.LBFGS(model.approximate_solution.parameters(),
                                  # lr=float(0.5),
                                  # max_iter=50000,
                                  # max_eval=50000,
                                  # history_size=150,
                                  # line_search_fn="strong_wolfe",
                                  # tolerance_change=1.0 * np.finfo(float).eps)
        # ADAM optimizer
        # optimizer_ADAM = optim.Adam(model.approximate_solution.parameters(),
                                    # lr=float(0.001))
        # loss = model.fit(num_epochs=n_epochs,
                    # optimizer=optimizer_ADAM,
                    # verbose=True)
        loss = model.fit_subdomain(num_epochs=n_epochs,
                                   optimizer= optim.Adam,
                                   verbose=True)
        # save the model after training
        model.save_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
    else:
        model.load_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
        loss = pd.read_json("FBPINN_loss_function.json").sort_index().T
        # print(loss)
        # os._exit(0)

    # plot results
    plotting_FBPINN(model, loss, domain, num_grid=400)


    return

if __name__ == '__main__':
    main()
