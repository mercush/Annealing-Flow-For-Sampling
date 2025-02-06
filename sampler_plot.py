import matplotlib.pyplot as plt
import os
import numpy as np

def plot_samples(Xtraj, Type, d, c, plot_directory=None, index=None):
    fig = plt.figure(figsize=(8.5, 8.5))
    fsize = 35
    Xtmp = Xtraj.cpu().numpy()
    circle = False
    if Type == 'truncated':
        circle = True
    if d == 2:
        ax = fig.add_subplot(111)
        Xtmp = Xtraj.cpu().numpy()
        print(f"Length of Xtmp: {len(Xtmp)}")
        ax.scatter(Xtmp[:, 0], Xtmp[:, 1], s=0.5)
        if Type == 'indicator':
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
        else:
            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
        if circle:
            theta = np.linspace(0, 2*np.pi, 2000)
            x = c * np.cos(theta)
            y = c * np.sin(theta)
            ax.plot(x, y, color = 'red', linestyle = '--')
    else:
        ax = fig.add_subplot(111, projection='3d')
        Xtmp = Xtraj.cpu().numpy()
        ax.scatter(Xtmp[:, 0], Xtmp[:, 1], Xtmp[:, 2], s=0.5)
        if Type == 'indicator':
            ax.set_xlim([-10, 10])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
        elif Type == 'funnel':
            ax.set_xlim([-5, 5])
            ax.set_ylim([-10, 10])
            ax.set_zlim([-10, 10])
        else:
            ax.set_xlim([-15, 15])
            ax.set_ylim([-15, 15])
            ax.set_zlim([-15, 15])
    if circle:
        distances = np.linalg.norm(Xtmp, axis=1)
        within_sphere = np.sum(distances <= c)
        proportion = within_sphere / len(Xtmp)
        print('Proportion of points within c: ', proportion)
        distances2 = np.linalg.norm(Xtmp, axis=1)
        within_sphere2 = np.sum(distances2 <= c+2)
        proportion2 = within_sphere2 / len(Xtmp)
        print('Proportion of points within c+2: ', proportion2)

    ax.set_title(f'Annealing Flow', fontsize=fsize)
    ax.tick_params(axis='both', which='major', labelsize=26)

    fig.tight_layout()
    if d >= 3:
        if Type == 'funnel':
            filename = f'd={d}_{Type}_Annealing.png'
        else:
            filename = f'd={d}_{Type}_Annealing.png'
    else:
        if Type == 'funnel':
            filename = f'd={d}_{Type}_Annealing.png'

        else:
            filename = f'd={d}_{Type}_c={c}_Annealing_01.png'

    plt.savefig(os.path.join(plot_directory, filename))
    plt.savefig(os.path.join(plot_directory, filename.replace('png', 'pdf')))



    plt.close()
    if Type == 'funnel':
        fig2 = plt.figure(figsize=(8.5, 8.5))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(Xtmp[:, 0], Xtmp[:, 1], s=0.5)
        ax2.set_xlim([-5, 5])
        ax2.set_ylim([-10, 10])

        ax2.set_title('Annealing Flow', fontsize=fsize)
        ax2.tick_params(axis='both', which='major', labelsize=26)
        filename2 = f'd={d}_{Type}_Annealing_12.png'
        plt.savefig(os.path.join(plot_directory, filename2))
        plt.savefig(os.path.join(plot_directory, filename2.replace('png', 'pdf')))

    elif Type == 'exponential' and d != 1:
        fig2 = plt.figure(figsize=(8.5, 8.5))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(Xtmp[:, 0], Xtmp[:, 1], s=0.5)
        ax2.set_xlim([-15, 15])
        ax2.set_ylim([-15, 15])
        ax2.set_title('Annealing Flow', fontsize=fsize)
        ax2.tick_params(axis='both', which='major', labelsize=26)
        filename2 = f'd={d}_{Type}_Annealing_12.png'
        plt.savefig(os.path.join(plot_directory, filename2))
        plt.savefig(os.path.join(plot_directory, filename2.replace('png', 'pdf')))
        fig3 = plt.figure(figsize=(8.5, 8.5))
        ax3 = fig3.add_subplot(111)
        def f(x):
            normalization_constant = 1/(2 * np.sqrt(2 * np.pi) * np.exp(0.5*c**2))
            return normalization_constant * np.exp(c*np.abs(x)) * np.exp(-0.5 * x**2)
        hist = ax3.hist(Xtmp[:, 0], bins=300, density=True, alpha=0.7, label='Samples')
        x_true = np.linspace(-15, 15, 800)
        true_density = f(x_true)
        ax3.plot(x_true, true_density, color='orange', linewidth=2, label='True Density')
        ax3.legend(loc='upper right', fontsize=22)
        ax3.set_ylim(0, 0.45)
        ax3.set_title('Annealing Flow', fontsize=fsize)
        ax3.tick_params(axis='both', which='major', labelsize=26)
        filename3 = f'd={d}_{Type}_Annealing_1.png'
        plt.savefig(os.path.join(plot_directory, filename3))
        plt.savefig(os.path.join(plot_directory, filename3.replace('png', 'pdf')))
        plt.close()
    plt.close()