import torch
import torch.optim as optim
import numpy as np

from PINN import PINN
from exact_solution import exact_solution
from plot_result import plotting

def main():
    # input parameter for PINN

    # collocation parameters
    n_int, n_sb = 10000, 100
    n_tb = 200-n_sb
    # NN parameters
    neurons, n_hidden = 20, 9

    # physics domain
    domain = torch.tensor([[0., 1.], # time domain
                           [-1., 1.]]) # space domain
    # viscosity parameter
    nu = 0.01 / np.pi

    # parameters for training
    # batch parameters
    n_batches = 1
    # number of epochs
    n_epochs = 1000


    # PINN training
    model = PINN(n_int, n_sb, n_tb, neurons, n_hidden, domain, n_batches, nu)
    if False:
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
        hist = model.fit(num_epochs=n_epochs,
                    optimizer=optimizer_LBFGS,
                    verbose=True)
        # save the model after training
        model.save_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/PINN_burger.th")
    else:
        model.load_model(path="/Users/pauliebao/AI_for_chemistry/PINN_problem/PINN_project/results/PINN_burger.th")

    # plot results
    plotting(model, domain)

    return

if __name__ == '__main__':
    main()
