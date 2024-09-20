import os
import re
import argparse
import yaml
import math
import matplotlib.pyplot as plt
import scipy
import pickle
import numpy as np 
import torch
import torch.nn as nn
import torchdiffeq as tdeq
import time
from PIL import Image
from argparse import Namespace
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from Annealing_Flow import gen_data, get_e_ls, divergence_approx, divergence_bf, FCnet, ODEFunc,\
    CNF, default_CNF_structure, FlowNet_forward, l2_norm_sqr, move_over_blocks, push_samples_forward,\
        on_off, load_prev_CNFs, loop_data_loader, add_diffuse

# python /Users/a59611/code/gen/AnnealingFlow/Annealing_Flow.py --JKO_config /Users/a59611/code/gen/AnnealingFlow/Annealing_Flow_exp.yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpu = torch.cuda.device_count()
mult_gpu = False if num_gpu < 2 else True

def plot_samples(Xtraj, d, plot_directory=None, index=None):
    fig = plt.figure(figsize=(8.5, 8.5))
    fsize = 35
    Xtmp = Xtraj.cpu().numpy()
    if d == 1:
        def f(x):
            normalization_constant = 1/(2 * np.sqrt(2 * np.pi) * np.exp(0.5*c**2))
            return normalization_constant * np.exp(c*np.abs(x)) * np.exp(-0.5 * x**2)
        ax = fig.add_subplot(111)
        hist = ax.hist(Xtraj.cpu().numpy(), bins=300, density=True, alpha=0.7, label='Samples')
        x_true = np.linspace(-15, 15, 800)
        true_density = f(x_true)
        ax.plot(x_true, true_density, color='orange', linewidth=2, label='True Density')
        ax.legend(loc='upper right', fontsize=22)  # Increased legend font size
        ax.set_ylim(0, 0.45)

    elif d == 2:
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

    ax.set_title(f'AI-Sampler', fontsize=fsize)
    
    ax.tick_params(axis='both', which='major', labelsize=26)

    fig.tight_layout()
    if d >= 3:
        if Type == 'funnel':
            filename = f'd={d}_{Type}_sigma_{sigma}_Annealing.png'
        else:
            filename = f'd={d}_{Type}_Annealing.png'
    else:
        if Type == 'funnel':
            filename = f'd={d}_{Type}_sigma_{sigma}_Annealing.png'

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
        filename2 = f'd={d}_{Type}_sigma_{sigma}_Annealing_12.png'
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
    plt.close()

def get_beta(block_id):
    total_num = block_idxes[-1]
    beta = block_id/total_num
    return beta

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--AnnealingFlow_config', default = '/Samplers.yaml', type=str, help='Path to the YAML file')

args_parsed = parser.parse_args()
with open(args_parsed.AnnealingFlow_config, 'r') as file:
    args_yaml = yaml.safe_load(file)

if __name__ == '__main__':
    c = args_yaml['data']['c']
    Xdim_flow = args_yaml['data']['Xdim_flow']
    master_dir = '/storage/home/hcoda1/3/dwu381/scratch/Flow/AnnealingFlow_Final_Version'
    d = args_yaml['data']['Xdim_flow']
    c = args_yaml['data']['c']
    Type = args_yaml['data']['type']
    samplers_trained_path = os.path.join(master_dir, f'samplers_trained/d={Xdim_flow}_{Type}_c={c}')
    if os.path.exists(samplers_trained_path):
        samples_dir = os.path.join(master_dir, f'samplers_trained/d={Xdim_flow}_{Type}_c={c}_4')
    else:
        print("Warning: The 'samplers_trained' directory does not exist. Users must first run Annealing_Flow.py to train the samplers, and then use this code for fast sampling.")
        print("Program terminated.")
        exit()
    circle = False
    if Type == 'truncated':
        circle = True
    all_entries = os.listdir(samples_dir)
    block_pattern = re.compile(r'block(\d+)')
    block_numbers = []
    for entry in all_entries:
        match = block_pattern.match(entry)
        if match:
            block_numbers.append(int(match.group(1)))
    if block_numbers:
        max_block_number = max(block_numbers)
        # Create a list ranging from 1 to the highest block number
        block_idxes = list(range(1, max_block_number + 1))
        print(f"Number of blocks: {max_block_number}")
        print(f"Block list: {block_idxes}")
    else:
        print("No blocks found.")
    for block_id in block_idxes:
        folder_suffix = args_yaml['eval']['folder_suffix']
        dir = os.path.join(samples_dir,f'{folder_suffix}')
        
        prefix = 'block' 
        common_name = f'{prefix}{block_id}'
        filepath = os.path.join(dir, common_name + '.pth') # Store the block in .pth file
        self = Namespace() 

        nte = args_yaml['sampling']['nsamples']
        xte = gen_data(nte, args_yaml['data']['Xdim_flow'])
        xte = xte.float().to(device)
        self.X_test = xte
        vfield_config = Namespace(hid_dims = args_yaml['CNF']['hid_dims'], Xdim_flow=args_yaml['data']['Xdim_flow'])
        self.CNF = default_CNF_structure(config = vfield_config)
        self.ls_args_CNF = []  # self is a Namespace
        common_args_CNF = Namespace(
            int_mtd = 'rk4',
            num_e = 1, 
            num_int_pts = 1,
            fix_e_ls = True, 
            use_NeuralODE = True, 
            rtol = 1e-5,
            atol = 1e-5)
        S = args_yaml['data']['S']
        hk_blocks = 0.10
        hk_b = 1
        hk_ls = np.array([hk_b/S] * S)
        self.ls_args_CNF = []
        for i in range(S):
            args_CNF_now = Namespace(**vars(common_args_CNF))  
            hk_sub = hk_ls[i]
            args_CNF_now.Tk_1 = 0 if i == 0 else np.sum(hk_ls[:i])
            args_CNF_now.Tk = args_CNF_now.Tk_1 + hk_sub
            self.ls_args_CNF.append(args_CNF_now)  

        total_params = sum(p.numel() for p in self.CNF.parameters())
        #self.block_now = len(self.ls_args_CNF)
        self.block_now = len(self.ls_args_CNF)
        self_ls_prev = []
        if block_id > 1:
            load_configs = Namespace(block_id = block_id, master_dir = samples_dir, vfield_config = vfield_config,\
                prefix = prefix)
            self_ls_prev = load_prev_CNFs(load_configs)
            assert len(self_ls_prev) == block_id - 1
            self.CNF.load_state_dict(self_ls_prev[-1].CNF.state_dict())
        if os.path.exists(filepath):
            checkpt = torch.load(filepath, map_location=torch.device('cpu'))
            self.CNF.load_state_dict(checkpt['model'])
            self.loss_at_block = checkpt['loss_at_block']
        else:
            self.loss_at_block = []
        self.CNF = torch.nn.DataParallel(self.CNF)

    on_off(self, on = True)
    move_configs = Namespace(block_id = block_id, self_ls_prev = self_ls_prev)
    start_time = time.time()
    Z_traj = move_over_blocks(self, move_configs, nte = nte)
    end_time = time.time()
    print(f"Time taken for sampling {nte} points: {end_time - start_time} seconds")
    plot_dir = os.path.join(master_dir, 'plot_results')
    os.makedirs(plot_dir, exist_ok=True)
    plot_samples(Z_traj[-1], d= Xdim_flow, plot_directory = plot_dir)

    filename_data = os.path.join(samples_dir, f'Xtest_{nte}.pkl')
    def count_modes(samples, threshold=7):
        binary_samples = (samples > threshold).astype(int)
        decimal_samples = np.sum(binary_samples * (2 ** np.arange(samples.shape[1])), axis=1)
        unique_modes, mode_counts = np.unique(decimal_samples, return_counts=True)
        total_samples = len(samples)
        mode_proportions = mode_counts / total_samples
        return len(unique_modes), mode_proportions
    Z_samples = Z_traj[-1].cpu().numpy() if isinstance(Z_traj[-1], torch.Tensor) else Z_traj[-1]
    num_modes, proportions = count_modes(Z_samples)
    print(f'When calculating the number of modes, we recommend first sampling over 20,000 points, as there are 1,024 modes in total. Sampling fewer points may not cover all of them.')
    print(f"Number of modes: {num_modes}")