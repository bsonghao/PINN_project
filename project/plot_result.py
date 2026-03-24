import numpy as np
import torch as torch
import matplotlib.pyplot as plt
from exact_solution import exact_solution

def plotting(model, domain, num_grid=1000):
    """
    plot the results
    """
    # plot sampling points

    # Plot the input training points
    input_sb_, output_sb_ = model.add_spatial_boundary_points()
    input_tb_, output_tb_ = model.add_temporal_boundary_points()
    input_int_, output_int_ = model.add_interior_points()

    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(input_sb_[:, 0].detach().numpy(), input_sb_[:, 1].detach().numpy(), label="spatial boundary Points")
    plt.scatter(input_int_[:, 0].detach().numpy(), input_int_[:, 1].detach().numpy(), label="Interior Points")
    plt.scatter(input_tb_[:, 0].detach().numpy(), input_tb_[:, 1].detach().numpy(), label="temporal boundary Points")
    plt.xlabel("t",fontsize=40)
    plt.ylabel("x",fontsize=40)
    plt.legend(fontsize=40)
    plt.savefig("Collocation_points.png")
    plt.show()

    #plot heatmap for the full solution
    t = np.linspace(domain[0, 0], domain[0, 1], num_grid)
    x = np.linspace(domain[1, 0], domain[1, 1], num_grid)
    t, x = np.meshgrid(t, x)

    inputs = torch.from_numpy(np.stack([t.flatten(), x.flatten()], axis=-1)).float()
    output = model.approximate_solution(inputs).reshape((num_grid, num_grid))
    exact_output = exact_solution(inputs).reshape((num_grid, num_grid))

    fig, axs = plt.subplots(1, 3, figsize=(40, 8), dpi=150)

    im1 = axs[0].pcolormesh(t, x, exact_output.detach().numpy(), cmap="rainbow")
    axs[0].set_xlabel("t",fontsize=40)
    axs[0].set_ylabel("x",fontsize=40)
    cbar = fig.colorbar(im1, ax=axs[0])
    #axs[0].grid(True, which="both", ls=":")

    im2 = axs[1].pcolormesh(t, x, output.detach().numpy(), cmap="rainbow")
    axs[1].set_xlabel("t", fontsize=40)
    axs[1].set_ylabel("x", fontsize=40)
    plt.colorbar(im2, ax=axs[1])
    #axs[1].grid(True, which="both", ls=":")

    im3 = axs[2].pcolormesh(t, x, abs(exact_output.detach().numpy()-output.detach().numpy()), cmap="rainbow")
    axs[2].set_xlabel("t", fontsize=40)
    axs[2].set_ylabel("x", fontsize=40)
    plt.colorbar(im3, ax=axs[2])
    #axs[2].grid(True, which="both", ls=":")

    axs[0].set_title("Exact Solution", fontsize=40)
    axs[1].set_title("Approximate Solution", fontsize=40)
    axs[2].set_title("Absolute error", fontsize=40)
    plt.savefig("full_solution.png")
    plt.show()

    # plot solution at t=0.25,0.50,0.75
    t_cross_sections = [0.25, 0.5, 0.75]
    fig, axs = plt.subplots(1, len(t_cross_sections), figsize=(40, 8), dpi=150)

    for i, t_cs in enumerate(t_cross_sections):
        axs[i].set_xlabel("t", fontsize=40)
        axs[i].set_ylabel("u(x,t)", fontsize=40)
        input_dimension = domain.shape[0]
        x = torch.zeros((num_grid, input_dimension))
        x[:,0] = t_cs
        x[:,1] = torch.linspace(domain[1, 0], domain[1, 1], num_grid)
        u_pinn = model.approximate_solution(x).reshape(-1, )
        u_exact = exact_solution(x).reshape(-1,)
        axs[i].plot(x[:,1].detach().numpy(), u_pinn.detach().numpy(), label="prediction", color="red", linestyle="--", linewidth=10, alpha=.5)
        axs[i].plot(x[:,1].detach().numpy(), u_exact.detach().numpy(), label="exact", color="blue", linestyle="-", linewidth=10, alpha=.5)
        axs[i].set_title(f"t={t_cs:.2f}", fontsize=40)
        axs[i].legend(fontsize=40)


    plt.savefig("time_cross_section.png")
    plt.show()

    err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
    print("relative L2 Error Norm: ", err.item())

    return
