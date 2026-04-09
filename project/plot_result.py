import numpy as np
import torch as torch
import matplotlib.pyplot as plt
from exact_solution import exact_solution

def plotting_PINN(model, loss, domain, num_grid=400):
    """
    plot the results for PINN
    """

    # plot loss function
    plt.figure(figsize=(16,8))
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(loss["total loss"]) + 1), loss["total loss"], label="total Loss")
    plt.plot(np.arange(1, len(loss["physics loss"]) + 1), loss["physics loss"], label="physics Loss")
    plt.plot(np.arange(1, len(loss["boundary loss"]) + 1), loss["boundary loss"], label="boundary Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=20)
    plt.title("Plot of loss function", fontsize=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("loss_function.png")
    plt.show()
    # plot sampling points

    # Plot the input training points
    # (input_sb_, u_train_sb), (input_tb_, u_train_tb), (input_int_, u_train_int) = zip(model.training_set_sb, model.training_set_tb, model.training_set_int)
    data_sb, data_tb, data_int = iter(model.training_set_sb), iter(model.training_set_tb), iter(model.training_set_int)
    input_sb_, output_sb_ = next(data_sb)
    input_tb_, output_tb_ = next(data_tb)
    input_int_, output_int_ = next(data_int)

    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(input_sb_[:, 0].detach().numpy(), input_sb_[:, 1].detach().numpy(), label="spatial boundary Points", alpha=.5)
    plt.scatter(input_int_[:, 0].detach().numpy(), input_int_[:, 1].detach().numpy(), label="Interior Points", alpha=.5)
    plt.scatter(input_tb_[:, 0].detach().numpy(), input_tb_[:, 1].detach().numpy(), label="temporal boundary Points", alpha=.5)
    plt.xlabel("t",fontsize=40)
    plt.ylabel("x",fontsize=40)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
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
    axs[0].tick_params(labelsize=20)
    cbar = fig.colorbar(im1, ax=axs[0])
    #axs[0].grid(True, which="both", ls=":")

    im2 = axs[1].pcolormesh(t, x, output.detach().numpy(), cmap="rainbow")
    axs[1].set_xlabel("t", fontsize=40)
    axs[1].set_ylabel("x", fontsize=40)
    axs[1].tick_params(labelsize=20)
    plt.colorbar(im2, ax=axs[1])
    #axs[1].grid(True, which="both", ls=":")

    im3 = axs[2].pcolormesh(t, x, abs(exact_output.detach().numpy()-output.detach().numpy()), cmap="rainbow")
    axs[2].set_xlabel("t", fontsize=40)
    axs[2].set_ylabel("x", fontsize=40)
    axs[2].tick_params(labelsize=20)
    plt.colorbar(im3, ax=axs[2])
    #axs[2].grid(True, which="both", ls=":")

    axs[0].set_title("Exact Solution", fontsize=40)
    axs[1].set_title("Approximate Solution", fontsize=40)
    axs[2].set_title("Absolute error", fontsize=40)
    plt.savefig("full_solution.png")
    plt.show()
    print(abs(exact_output-output).max())

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
        axs[i].legend(fontsize=20)
        axs[i].tick_params(labelsize=20)
        print(abs(u_pinn-u_exact).max())


    plt.savefig("time_cross_section.png")
    plt.show()

    err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
    print("relative L2 Error Norm: ", err.item())

    return

def plotting_FBPINN(model, loss, domain, num_grid=1000):
    """
    plot the results for PINN
    """

    # plot loss function
    plt.figure(figsize=(16,8))
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(loss["loss"]) + 1), loss["loss"], label="PDE Loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(fontsize=20)
    plt.title("Plot of loss function", fontsize=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("FBPINN_loss_function.png")
    plt.show()
    # plot sampling points

    # Plot the input training points
    data=  iter(model.training_set)
    input, output = next(data)


    plt.figure(figsize=(16, 8), dpi=150)
    plt.scatter(input[:, 0].detach().numpy(), input[:, 1].detach().numpy(), label="collocation Points", alpha=.5)
    plt.xlabel("t",fontsize=40)
    plt.ylabel("x",fontsize=40)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("FBPINN_collocation_points.png")
    plt.show()

    #plot heatmap for the full solution
    t_lo, t_hi = domain[0]
    x_lo, x_hi = domain[1]
    t = np.linspace(t_lo, t_hi, num_grid)
    x = np.linspace(x_lo, x_hi, num_grid)
    t, x = np.meshgrid(t, x)
    active_sites = np.arange(len(model.subdomains))
    inputs = torch.from_numpy(np.stack([t.flatten(), x.flatten()], axis=-1)).float()
    output = model.predict(inputs, active_sites).reshape((num_grid, num_grid))
    exact_output = exact_solution(inputs).reshape((num_grid, num_grid))

    fig, axs = plt.subplots(1, 3, figsize=(40, 8), dpi=150)

    im1 = axs[0].pcolormesh(t, x, exact_output.detach().numpy(), cmap="rainbow")
    axs[0].set_xlabel("t",fontsize=40)
    axs[0].set_ylabel("x",fontsize=40)
    axs[0].tick_params(labelsize=20)
    cbar = fig.colorbar(im1, ax=axs[0])
    #axs[0].grid(True, which="both", ls=":")

    im2 = axs[1].pcolormesh(t, x, output.detach().numpy(), cmap="rainbow")
    axs[1].set_xlabel("t", fontsize=40)
    axs[1].set_ylabel("x", fontsize=40)
    axs[1].tick_params(labelsize=20)
    plt.colorbar(im2, ax=axs[1])
    #axs[1].grid(True, which="both", ls=":")

    im3 = axs[2].pcolormesh(t, x, abs(exact_output.detach().numpy()-output.detach().numpy()), cmap="rainbow")
    axs[2].set_xlabel("t", fontsize=40)
    axs[2].set_ylabel("x", fontsize=40)
    axs[2].tick_params(labelsize=20)
    plt.colorbar(im3, ax=axs[2])
    #axs[2].grid(True, which="both", ls=":")

    axs[0].set_title("Exact Solution", fontsize=40)
    axs[1].set_title("Approximate Solution", fontsize=40)
    axs[2].set_title("Absolute error", fontsize=40)
    plt.savefig("FBPINN_full_solution.png")
    plt.show()
    # print(abs(exact_output-output).max())

    # plot solution at t=0.25,0.50,0.75
    t_cross_sections = [0.25, 0.5, 0.75]
    fig, axs = plt.subplots(1, len(t_cross_sections), figsize=(40, 8), dpi=150)

    for i, t_cs in enumerate(t_cross_sections):
        axs[i].set_xlabel("t", fontsize=40)
        axs[i].set_ylabel("u(x,t)", fontsize=40)
        input_dimension = len(domain)
        x = torch.zeros((num_grid, input_dimension))
        x[:,0] = t_cs
        x_lo, x_hi = domain[1]
        x[:,1] = torch.linspace(x_lo, x_hi, num_grid)
        u_pinn = model.predict(x, active_sites).reshape(-1, )
        u_exact = exact_solution(x).reshape(-1,)
        axs[i].plot(x[:,1].detach().numpy(), u_pinn.detach().numpy(), label="prediction", color="red", linestyle=":", linewidth=10, alpha=.5)
        axs[i].plot(x[:,1].detach().numpy(), u_exact.detach().numpy(), label="exact", color="blue", linestyle="-", linewidth=10, alpha=.5)
        axs[i].set_title(f"t={t_cs:.2f}", fontsize=40)
        axs[i].legend(fontsize=20)
        axs[i].tick_params(labelsize=20)
        # print(abs(u_pinn-u_exact).max())


    plt.savefig("FBPINN_time_cross_section.png")
    plt.show()

    err = (torch.mean((output - exact_output) ** 2) / torch.mean(exact_output ** 2)) ** 0.5
    print("relative L2 Error Norm: ", err.item())

    return
