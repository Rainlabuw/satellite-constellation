import numpy as np
from dist_auction_algo_josh import Auction
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from scipy.ndimage import convolve
from methods import *

def run_perturbed_auction(n, m, pert_scale, eps):
    # Create first auction with random benefit matrix and 
    # get prices for this to use in initialization
    graph = rand_connected_graph(n)
    auction = Auction(n, m, graph=graph, eps=0.01)
    auction.run_auction()
    prices_init = auction.agents[0].public_prices

    #Perturb the benefits
    perturbed_benefits = auction.benefits + np.random.normal(scale=pert_scale, size=(n, m))
    perturbed_benefits = np.clip(perturbed_benefits, 0, 1)

    #Run seeded auction
    seeded_auction = Auction(n, m, graph=graph, benefits=perturbed_benefits, prices=prices_init, eps=eps)
    seeded_benefit = seeded_auction.run_auction()

    unseeded_auction = Auction(n, m, graph=graph, benefits=perturbed_benefits, eps=eps)
    unseeded_benefit = unseeded_auction.run_auction()

    dist = calc_distance_btwn_solutions(seeded_auction.agents, auction.agents)

    return seeded_benefit, unseeded_benefit, seeded_auction.n_iterations, unseeded_auction.n_iterations, dist

def run_2var_test():
    # Define the range of n and perturbation scale
    n_range = np.arange(10, 100, 10)
    pert_scale_range = np.arange(0.001, 0.101, 0.01)
    m_mult_range = np.arange(1, 2, 0.1)

    # Create a meshgrid of n and perturbation scale
    n_mesh, m_mult_mesh = np.meshgrid(n_range, m_mult_range)

    # Initialize arrays to store the results
    seeded_benefit = np.zeros_like(n_mesh, dtype='float')
    unseeded_benefit = np.zeros_like(n_mesh, dtype='float')
    seeded_iterations = np.zeros_like(n_mesh, dtype='float')
    unseeded_iterations = np.zeros_like(n_mesh, dtype='float')

    # Loop through the meshgrid and call run_perturbed_auction on each combination of n and perturbation scale
    for i in range(n_mesh.shape[0]):
        for j in range(n_mesh.shape[1]):
            print(f"{i}/{n_mesh.shape[0]}, {j}/{n_mesh.shape[1]}", end='\r')
            sb, usb, si, usi, dst = run_perturbed_auction(n_mesh[i,j], int(n_mesh[i,j]*m_mult_mesh[i,j]), 0.05, 0.01)
            seeded_benefit[i,j] = sb
            unseeded_benefit[i,j] = usb
            seeded_iterations[i,j] = si
            unseeded_iterations[i,j] = usi

    # Define the size of the window by which to smooth the data
    window_size = 3  # for a 3x3 kernel

    # Create a normalized 2D kernel (so the sum of the kernel is 1)
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)

    seeded_benefit = convolve(seeded_benefit, kernel, mode='reflect')
    unseeded_benefit = convolve(unseeded_benefit, kernel, mode='reflect')
    seeded_iterations = convolve(seeded_iterations, kernel, mode='reflect')
    unseeded_iterations = convolve(unseeded_iterations, kernel, mode='reflect')

    # Plot the results
    # Function to plot a single 3D surface
    def plot_3d_surface(ax, x, y, z, color, label):
        surf = ax.plot_surface(x, y, z, color=color, edgecolor='k', label=label)
        return surf

    # Create a new figure for the 3D plot
    fig = plt.figure()

    # Set up the axes for the first subplot
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    seeded_surf = plot_3d_surface(ax1, n_mesh, m_mult_mesh, seeded_benefit, 'blue', 'Seeded')
    unseeded_surf = plot_3d_surface(ax1, n_mesh, m_mult_mesh, unseeded_benefit, 'red', 'Unseeded')
    ax1.set_xlabel('n')
    ax1.set_ylabel('m multiplier')
    ax1.set_zlabel('Benefit')
    ax1.set_title('Seeded vs Unseeded Benefit')

    # Set up the axes for the second subplot
    ax2 = fig.add_subplot(2, 1, 2, projection='3d')
    seeded_surf = plot_3d_surface(ax2, n_mesh, m_mult_mesh, seeded_iterations, 'blue', 'Seeded')
    unseeded_surf = plot_3d_surface(ax2, n_mesh, m_mult_mesh, unseeded_iterations, 'red', 'Unseeded')
    ax2.set_xlabel('n')
    ax2.set_ylabel('m multiplier')
    ax2.set_zlabel('Iterations')
    ax2.set_title('Seeded vs Unseeded Iterations')

    # Create legend manually
    legend_elements = [Patch(facecolor='blue', edgecolor='k', label='Seeded'),
                    Patch(facecolor='red', edgecolor='k', label='Unseeded')]

    # Create a legend for the first subplot
    ax1.legend(handles=legend_elements)

    # Show plot
    # plt.tight_layout()
    plt.show()
    plt.savefig("mesh.png")

def run_1var_test():
    eps = 0.01
    n = 100
    m = "variable"
    pert_scale = 0.05

    independent_variable = np.arange(1,2,0.05)
    # independent_variable = np.arange(10, 100, 5)
    ind_label = "m mult"

    fig, axes = plt.subplots(nrows=3, ncols=1)
    
    fig.suptitle(f"Speedup Testing for n={n}, m={m}, eps={eps}, pert_scale={pert_scale}")
    sbs = []
    usbs = []
    sis = []
    usis = []
    dsts = []
    for i, ind in enumerate(independent_variable):
        print(f"{i}/{len(independent_variable)}", end='\r')
        sb, usb, si, usi, dst = run_perturbed_auction(n, int(n*ind), pert_scale, eps)
        sbs.append(sb)
        usbs.append(usb)
        sis.append(si)
        usis.append(usi)
        dsts.append(dst/n)
    
    # Define the size of the window by which to smooth the data
    window_size = 3  # for a 3x1 kernel

    # Create a normalized 1D kernel (so the sum of the kernel is 1)
    kernel = np.ones(window_size) / (window_size)

    seeded_benefit = convolve(np.array(sbs), kernel, mode='reflect')
    unseeded_benefit = convolve(np.array(usbs), kernel, mode='reflect')
    seeded_iterations = convolve(np.array(sis), kernel, mode='reflect')
    unseeded_iterations = convolve(np.array(usis), kernel, mode='reflect')
    distances = convolve(np.array(dsts), kernel, mode='reflect')

    axes[0].plot(independent_variable, seeded_benefit, label="Seeded Benefit")
    axes[0].plot(independent_variable, unseeded_benefit, label="Unseeded Benefit")
    axes[0].set_xlabel(ind_label)
    axes[0].set_ylabel("Benefit")
    axes[0].legend()

    axes[1].plot(independent_variable, distances, label="Distance btwn solutions")
    axes[1].set_xlabel(ind_label)
    axes[1].set_ylabel("% Of Agents changed")
    axes[1].legend()

    axes[2].plot(independent_variable, seeded_iterations, label="Seeded Iterations")
    axes[2].plot(independent_variable, unseeded_iterations, label="Unseeded Iterations")
    axes[2].set_xlabel(ind_label)
    axes[2].set_ylabel("Iterations")
    axes[2].legend()

    plt.savefig("test.png")
    plt.show()

if __name__ == "__main__":
    run_1var_test()