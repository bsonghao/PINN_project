import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time

from FBPINN import FBPINN, FBPINN_fast
from FBPINN_trainer import FBPINN_trainer
from exact_solution import exact_solution
from generate_bound import generate_rectangular_bounds, print_subdomains
from plot_result import plotting_FBPINN
from schedule import SequentialScheduler, ColumnScheduler, SubdomainStateManager
from problems import Burgers2D

def main():
    # input parameter for PINN

    # collocation parameters
    n_sample = 40000
    input_dimension = 2

    # NN parameters
    neurons, n_hidden = 16, 2 # 16 neurons per hidden layer and 2 hidden layers (4 layers in total)
    unnorm_para = (1, 0)

    # define physics domain
    domain = [(0., 1.), (-1., 1.)] # time domain x space domain
    # parameters to define subdomains
    num_grid = [3, 2]
    overlap = 0.1

    # viscosity parameter
    nu = 0.01 / np.pi

    # parameters for training
    # batch parameters
    n_batches = 1
    # number of epochs
    n_epochs = 50000

    # setup problem
    problem = Burgers2D(nu)


    # parameters for scheduler
    global_flag = True
    activation_interval = 200
    fixed_after_epochs  = 200
    convergence_tol     = 1e-8
    convergence_window  = 5
    num_per_column      = num_grid[1]
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
    if False:
        network = FBPINN(
                     subdomains = subdomains,
                     in_dim=2,
                     out_dim=1,
                     hidden = neurons,
                     layers = n_hidden,
                     unnorm_para = unnorm_para,
                     )
    else:
        # build FBPINN ansatz with parallelized FBPINN ansatz using vmap
        network = FBPINN_fast(
             subdomains = subdomains,
             in_dim=2,
             out_dim=1,
             hidden = neurons,
             layers = n_hidden,
             unnorm_para = unnorm_para,
             )

    # FBPINN training
    model = FBPINN_trainer(
                   problem,
                   subdomains,
                   network,
                   n_sample,
                   n_batches,
                   input_dimension,
                   activation_interval,
                   fixed_after_epochs,
                   convergence_tol,
                   convergence_window,
                   num_per_column,
                   SubdomainStateManager,
                   ColumnScheduler)
    print(f"model layers,\n{model.approximate_solution}")
    # os._exit(0)

    if True:
        if global_flag:
            # LBFGS optimizer
            # optimizer_LBFGS = optim.LBFGS(model.approximate_solution.parameters(),
                                      # lr=float(0.5),
                                      # max_iter=50000,
                                      # max_eval=50000,
                                      # history_size=150,
                                      # line_search_fn="strong_wolfe",
                                      # tolerance_change=1.0 * np.finfo(float).eps)
            # ADAM optimizer
            optimizer_ADAM = optim.Adam(model.approximate_solution.parameters(),
                                        lr=float(0.001))
            start_time = time.perf_counter()
            loss = model.fit(num_epochs=n_epochs,
                        optimizer=optimizer_ADAM,
                        verbose=True)
            end_time = time.perf_counter()
            print(f"It takes {end_time-start_time:.3f} s to train FBPINN for {n_epochs:d} epoches")
            # store loss function data
            df = pd.DataFrame(loss)
            df.to_json("global_FBPINN_loss_function.json")
        else:
            loss = model.fit_subdomain(num_epochs=n_epochs,
                                   optimizer= optim.Adam,
                                   verbose=True)
            # store loss function data
            df = pd.DataFrame(loss)
            df.to_json("local_FBPINN_loss_function.json")
        # save the model after training
        model.save_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
    else:
        model.load_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
        loss = pd.read_json("global_FBPINN_loss_function.json").sort_index()
        # print(loss)
        # os._exit(0)

    # plot results
    # model.approximate_solution.eval()
    plotting_FBPINN(model, loss, domain, num_grid=400)


    return

if __name__ == '__main__':
    main()
