import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from FBPINN import FBPINN
from exact_solution import exact_solution
from generate_bound import generate_rectangular_bounds, print_subdomains
from plot_result import plotting_FBPINN

def main():
    # input parameter for PINN

    # collocation parameters
    n_sample = 10000

    # NN parameters
    neurons, n_hidden = 20, 3 # 20 neurons per hidden layer and 7 hidden layers (9 layers in total)

    # define physics domain
    domain = [(0., 1.), (-1., 1.)] # time domain x space domain
    # parameters to define subdomains
    num_grid = [2, 3]
    overlap = 0.1

    # ── Build 2*3 block overlapping 2-D subdomains on [0, 1]x[-1, 1]

    subdomains =  generate_rectangular_bounds(
                           dimensions = 2,
                           num_subdomains = num_grid,
                           ranges = domain,
                           overlap = overlap,
                           overlap_mode = "absolute")

    print_subdomains(subdomains, dimensions=2)

    # viscosity parameter
    nu = 0.01 / np.pi

    # parameters for training
    # batch parameters
    n_batches = 1
    # number of epochs
    n_epochs = 10


    # FBPINN training
    model = FBPINN(subdomains, n_sample, neurons, n_hidden, n_batches, nu)

    # plot collocation points
    data=  iter(model.training_set)
    input, output = next(data)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(16, 8), dpi=150)
    # plt.scatter(input[:, 0].detach().numpy(), input[:, 1].detach().numpy(), label="collocation Points", alpha=.5)
    # plt.xlabel("t",fontsize=40)
    # plt.ylabel("x",fontsize=40)
    # plt.legend(fontsize=20)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.savefig("FBPINN_collocation_points.png")
    # plt.show()
    # os._exit(0)
    if True:
        # LBFGS optimizer
        optimizer_LBFGS = optim.LBFGS(model.approximate_solution.parameters(),
                                  lr=float(0.5),
                                  max_iter=50000,
                                  max_eval=50000,
                                  history_size=150,
                                  line_search_fn="strong_wolfe",
                                  tolerance_change=1.0 * np.finfo(float).eps)
        # ADAM optimizer
        optimizer_ADAM = optim.Adam(model.approximate_solution.parameters(),
                                    lr=float(0.001))
        loss = model.fit(num_epochs=n_epochs,
                    optimizer=optimizer_LBFGS,
                    verbose=True)
        # save the model after training
        model.save_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
    else:
        model.load_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/FBPINN_burger.th")
        loss = pd.read_json("FBPINN_loss_function.json")

    # plot results
    plotting_FBPINN(model, loss, domain)


    return

if __name__ == '__main__':
    main()
