""" Plot_PE_PINN_Results.py

Description:
   Code to map unstructured data onto 2D grid using interpolation capabilities of scipy\n
        For more information, please check our paper at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110

"""

import os
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Input interface for python.
parser = argparse.ArgumentParser(description='''
        Plot parameter estimation PINN results for elasticity imaging\n
        For more information, please check our paper at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110'''
                                 )

# Argument for interpolation resolution along horizontal axis
parser.add_argument('-nxp', '--numxplot',
                    help='Num Node in X for ploting final results (default 200)', type=int, nargs=1, default=[40])
parser.add_argument('-op', '--outputpath', help='Output path (default ./output)',
                    type=str, nargs=1, default=['output'])

parser.add_argument('-ip', '--inputpath', help='Input domain data path (default ./input_domain_data)',
                    type=str, nargs=1, default=['input_domain_data'])
args = parser.parse_args()


# loading the ground_truth data
# loading the training files
X_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'X_locations.txt')).reshape(-1, 1)
Y_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Y_locations.txt')).reshape(-1, 1)
Sxx_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Sxx.txt')).reshape(-1, 1)
Syy_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Syy.txt')).reshape(-1, 1)
Sxy_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Sxy.txt')).reshape(-1, 1)
Youngs_GroundTruth = np.loadtxt(os.path.join(
    args.inputpath[0], 'Youngs.txt')).reshape(-1, 1)
Poissons_GroundTruth = np.loadtxt(os.path.join(
    args.inputpath[0], 'Poissons.txt')).reshape(-1, 1)

# loading the PINN-estimated results
Sxx_pred = np.loadtxt(os.path.join(
    args.outputpath[0], 'Sxx_pred.txt')).reshape(-1, 1)
Syy_pred = np.loadtxt(os.path.join(
    args.outputpath[0], 'Syy_pred.txt')).reshape(-1, 1)
Sxy_pred = np.loadtxt(os.path.join(
    args.outputpath[0], 'Sxy_pred.txt')).reshape(-1, 1)
Youngs_pred = np.loadtxt(os.path.join(
    args.outputpath[0], 'Youngs_pred.txt')).reshape(-1, 1)
Poissons_pred = np.loadtxt(os.path.join(
    args.outputpath[0], 'Poissons_pred.txt')).reshape(-1, 1)


XMIN = np.min(X_Training)
XMAX = np.max(X_Training)
YMIN = np.min(Y_Training)
YMAX = np.max(Y_Training)


l0 = np.mean([XMAX-XMIN, YMAX-YMIN])

Xmesh_plot = np.linspace(XMIN, XMAX, args.numxplot[0]).reshape((-1, 1))
Ymesh_plot = np.linspace(YMIN, YMAX, int(np.round(
    args.numxplot[0]*YMAX/XMAX))).reshape((-1, 1))

X_plot, Y_plot = np.meshgrid(Xmesh_plot, Ymesh_plot)


def make_structured(oneD_data):
    # Convert 1D data to structured map using a meshgrid

    # Use the scipy griddata function to perform linear interpolation in 2D
    data_output = griddata(np.column_stack(((1/l0)*X_Training, (1/l0)*Y_Training)),
                           oneD_data, np.column_stack(((1/l0)*X_plot.reshape(-1, 1), (1/l0)*Y_plot.reshape(-1, 1))))

    # Reshape the output into rectangular shape
    structured_data_output = data_output.reshape(
        len(Ymesh_plot), len(Xmesh_plot))
    return structured_data_output


Sxx_GT_grid = make_structured(Sxx_Training)
Syy_GT_grid = make_structured(Syy_Training)
Sxy_GT_grid = make_structured(Sxy_Training)
Youngs_GT_grid = make_structured(Youngs_GroundTruth)
Poissons_GT_grid = make_structured(Poissons_GroundTruth)


Sxx_pred_grid = make_structured(Sxx_pred)
Syy_pred_grid = make_structured(Syy_pred)
Sxy_pred_grid = make_structured(Sxy_pred)
Youngs_pred_grid = make_structured(Youngs_pred)
Poissons_pred_grid = make_structured(Poissons_pred)


def plot_sbs_all():
    # Create a 2*5 panel to plot and compare stress and material parameter maps for ground truth and PINN estimation

    f, axes = plt.subplots(2, 5, sharey=True, figsize=(20, 10))

    # First column: normal x stress
    input_img = Sxx_GT_grid
    output_img = Sxx_pred_grid
    data_range = np.max(input_img)-np.min(input_img)

    ax1_sub = axes[0, 0].pcolor(input_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[0, 0].set_aspect('equal', 'box')
    axes[0, 0].set_title("$\sigma_{xx}$ (Pa)", fontsize=20)
    axes[0, 0].xaxis.set_visible(False)
    axes[0, 0].set_ylabel('Ground truth', fontsize=20)
    axes[0, 0].set_yticklabels([])
    axes[0, 0].set_xticklabels([])
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    f.colorbar(ax1_sub, ax=axes[0, 0], fraction=0.046, pad=0.04)

    ax2_sub = axes[1, 0].pcolor(output_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[1, 0].set_aspect('equal', 'box')
    axes[1, 0].xaxis.set_visible(False)
    axes[1, 0].set_ylabel('PINN estimation', fontsize=20)
    axes[1, 0].set_yticklabels([])
    axes[1, 0].set_xticklabels([])
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    f.colorbar(ax2_sub, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Second column: normal y stress
    input_img = Syy_GT_grid
    output_img = Syy_pred_grid
    data_range = np.max(input_img)-np.min(input_img)

    ax1_sub = axes[0, 1].pcolor(input_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[0, 1].set_aspect('equal', 'box')
    axes[0, 1].set_title("$\sigma_{yy}$ (Pa)", fontsize=20)
    axes[0, 1].yaxis.set_visible(False)
    axes[0, 1].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 1], fraction=0.046, pad=0.04)

    ax2_sub = axes[1, 1].pcolor(output_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[1, 1].set_aspect('equal', 'box')
    axes[1, 1].yaxis.set_visible(False)
    axes[1, 1].xaxis.set_visible(False)
    f.colorbar(ax2_sub, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Third column: shear xy stress
    input_img = Sxy_GT_grid
    output_img = Sxy_pred_grid
    data_range = np.max(input_img)-np.min(input_img)

    ax1_sub = axes[0, 2].pcolor(input_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[0, 2].set_aspect('equal', 'box')
    axes[0, 2].set_title("$\sigma_{xy}$ (Pa)", fontsize=20)
    axes[0, 2].yaxis.set_visible(False)
    axes[0, 2].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 2], fraction=0.046, pad=0.04)

    ax2_sub = axes[1, 2].pcolor(output_img, cmap='jet', vmin=np.min(input_img) - 0.1*(data_range),
                                vmax=np.max(input_img) + 0.1*(data_range))
    axes[1, 2].set_aspect('equal', 'box')
    axes[1, 2].yaxis.set_visible(False)
    axes[1, 2].xaxis.set_visible(False)
    f.colorbar(ax2_sub, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Fourth column: Poisson's ratio
    input_img = Poissons_GT_grid
    output_img = Poissons_pred_grid

    ax1_sub = axes[0, 3].pcolor(input_img, cmap='jet', vmin=0.3,
                                vmax=0.5)
    axes[0, 3].set_aspect('equal', 'box')
    axes[0, 3].set_title("$\u03BD_{xy}$", fontsize=20)
    axes[0, 3].yaxis.set_visible(False)
    axes[0, 3].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 3], fraction=0.046, pad=0.04)

    ax2_sub = axes[1, 3].pcolor(output_img, cmap='jet', vmin=0.3,
                                vmax=0.5)
    axes[1, 3].set_aspect('equal', 'box')
    axes[1, 3].yaxis.set_visible(False)
    axes[1, 3].xaxis.set_visible(False)
    f.colorbar(ax2_sub, ax=axes[1, 3], fraction=0.046, pad=0.04)

    # Fifth column: Young's Modulus
    input_img = Youngs_GT_grid/1000
    output_img = Youngs_pred_grid/1000

    ax1_sub = axes[0, 4].pcolor(input_img, cmap='jet', vmin=1,
                                vmax=6)
    axes[0, 4].set_aspect('equal', 'box')
    axes[0, 4].set_title("$E$ (kPa)", fontsize=20)
    axes[0, 4].yaxis.set_visible(False)
    axes[0, 4].xaxis.set_visible(False)
    f.colorbar(ax1_sub, ax=axes[0, 4], fraction=0.046, pad=0.04)

    ax2_sub = axes[1, 4].pcolor(output_img, cmap='jet', vmin=1,
                                vmax=6)
    axes[1, 4].set_aspect('equal', 'box')
    axes[1, 4].yaxis.set_visible(False)
    axes[1, 4].xaxis.set_visible(False)
    f.colorbar(ax2_sub, ax=axes[1, 4], fraction=0.046, pad=0.04)

    plt.savefig(os.path.join(
        args.outputpath[0], "PINN_GroundTruth_comparison.png"))


plot_sbs_all()
