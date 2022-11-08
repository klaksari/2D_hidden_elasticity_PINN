""" PE_PINN.py

Description:
   Code written using SciANN for spatial discovery of elastic modulus and Poisson's ratio in elasticity imaging\n
        For more information, please check our paper at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110

Modified from code developed by Ehsan Haghighat: 
https://www.sciann.com/, https://github.com/sciann/sciann-applications/tree/master/SciANN-SolidMechanics
https://www.sciencedirect.com/science/article/pii/S0045782521000773

"""

import os
import sys
import time
import numpy as np
from sciann.utils.math import diff
from sciann import SciModel, Functional, Parameter
from sciann import Data, Tie
from sciann import Variable, Field

import matplotlib.pyplot as plt
import argparse


# Input interface for python.
parser = argparse.ArgumentParser(description='''
        Code written using SciANN for spatial discovery of elastic modulus and Poisson's ratio in elasticity imaging\n
        For more information, please check our paper at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4203110'''
                                 )

# Define number of data points.
parser.add_argument(
    '-l', '--layers', help='Num layers and neurons (default 6 layers each 100 neurons [100, 100, 100, 100, 100, 100])', type=int, nargs='+', default=[100]*6)
parser.add_argument('-af', '--actf', help='Activation function (default tanh)',
                    type=str, nargs=1, default=['tanh'])
parser.add_argument('-rn', '--resnet', help='Constructs a resnet architecture (default False)',
                    type=bool, nargs=1, default=[False])

parser.add_argument('-bs', '--batchsize',
                    help='Batch size for Adam optimizer (default 32)', type=int, nargs=1, default=[100])
parser.add_argument(
    '-e', '--epochs', help='Maximum number of epochs (default 2000)', type=int, nargs=1, default=[1])
parser.add_argument('-lr', '--learningrate',
                    help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument(
    '-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

parser.add_argument('--shuffle', help='Shuffle data for training (default True)',
                    type=bool, nargs=1, default=[True])
parser.add_argument('--stopafter', help='Patience argument from Keras (default 200000)',
                    type=int, nargs=1, default=[200000])
parser.add_argument('--dtype', help='Data type for weights and biases (default float32)',
                    type=str, nargs=1, default=['float32'])
parser.add_argument('--gpu', help='Use GPU if available (default False)',
                    type=bool, nargs=1, default=[True])
parser.add_argument('-op', '--outputpath', help='Output path (default ./output)',
                    type=str, nargs=1, default=['output'])

parser.add_argument('-ip', '--inputpath', help='Input domain data path (default ./input_domain_data)',
                    type=str, nargs=1, default=['input_domain_data'])
parser.add_argument('-of', '--outputprefix',
                    help='Output path (default res**)', type=str, nargs=1, default=['res'])


args = parser.parse_args()

if not args.gpu[0]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# current file name.
current_file_name = os.path.basename(__file__).split(".")[0]

# Prepare training data
# loading the training files
X_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'X_locations.txt')).reshape(-1, 1)
Y_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Y_locations.txt')).reshape(-1, 1)


Exx_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Exx.txt')).reshape(-1, 1)
Eyy_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Eyy.txt')).reshape(-1, 1)
Exy_Training = np.loadtxt(os.path.join(
    args.inputpath[0], 'Exy.txt')).reshape(-1, 1)
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


XMIN = np.min(X_Training)
XMAX = np.max(X_Training)
YMIN = np.min(Y_Training)
YMAX = np.max(Y_Training)

# Extract boundary indices
XTOL, YTOL = np.array([XMAX-XMIN, YMAX-YMIN])*1e-6
left_ids = np.where(abs(X_Training - XMIN) < XTOL)[0]
right_ids = np.where(abs(X_Training - XMAX) < XTOL)[0]
bot_ids = np.where(abs(Y_Training - YMIN) < YTOL)[0]
top_ids = np.where(abs(Y_Training - YMAX) < YTOL)[0]
leftright_ids = np.unique(np.concatenate([left_ids, right_ids]))
topbottom_ids = np.unique(np.concatenate([bot_ids, top_ids]))

# Compute characteristic scales
Syy_Max = np.max(Syy_Training[top_ids])
sigma0 = Syy_Max
l0 = np.mean([XMAX-XMIN, YMAX-YMIN])


# Define body force functions in x and y. In the current study, they were assumed to be zero.
def bodyfx(xx):
    x, y = xx[0], xx[1]
    frc = 0
    return frc


def bodyfy(xx):
    x, y = xx[0], xx[1]
    frc = 0
    return frc


# Function for plotting loss in a semilogx graph
def cust_semilogx(AX, X, Y, xlabel, ylabel):
    if X is None:
        im = AX.semilogy(Y)
    else:
        im = AX.semilogy(X, Y)
    if xlabel is not None:
        AX.set_xlabel(xlabel)
    if ylabel is not None:
        AX.set_ylabel(ylabel)


def train():
    # define output folder.
    if not os.path.isdir(args.outputpath[0]):
        os.mkdir(args.outputpath[0])

    output_file_name = os.path.join(args.outputpath[0], args.outputprefix[0])
    fname = output_file_name + \
        "_{}_".format(args.actf[0]) + "x".join([str(x) for x in args.layers])

    # Neural Network Setup.
    x = Variable("x", dtype=args.dtype[0])
    y = Variable("y", dtype=args.dtype[0])
    Exx = Variable("Exx", dtype=args.dtype[0])
    Eyy = Variable("Eyy", dtype=args.dtype[0])
    Exy = Variable("Exy", dtype=args.dtype[0])

    # Two networks for stress terms and Lamé parameters:
    Sxx, Syy, Sxy = Functional(
        ["Sxx", "Syy", "Sxy"],
        [x, y], args.layers, args.actf[0], res_net=args.resnet[0])

    lame1, lame2 = Functional(["lame1", "lame2"], [
        x, y], args.layers, args.actf[0], res_net=args.resnet[0])

    C11 = (2*lame2 + lame1)
    C12 = lame1
    C33 = 2*lame2

    d1 = Data(Sxx)
    d2 = Data(Syy)

    c1 = Tie(Sxx, Exx*C11 + Eyy*C12)
    c2 = Tie(Syy, Eyy*C11 + Exx*C12)
    c3 = Tie(Sxy, Exy*C33)

    Lx = diff(Sxx, x) + diff(Sxy, y)
    Ly = diff(Sxy, x) + diff(Syy, y)

    # Define the optimization model (set of inputs and constraints)
    # Add load_weights_from= argument if you want to load from previous weights

    # The main model takes spatial coordinates and strains as input variables,
    # has two data-driven targets (d1, d2), three constitutive equations (c1, c2, c3), and two momentum balance equations (Lx, Ly)

    model = SciModel(
        inputs=[x, y, Exx, Eyy, Exy],
        targets=[d1, d2, c1, c2, c3, Lx, Ly],
        loss_func="mse"
    )
    with open("{}_summary".format(fname), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))

    # Because the formulation is non-dimensional, we have to non-dimensionalize the domain data before using in PINN
    input_data = [(1/l0)*X_Training, (1/l0)*Y_Training,
                  Exx_Training, Eyy_Training, Exy_Training]

    data_d1 = (1/sigma0)*Sxx_Training
    data_d2 = (1/sigma0)*Syy_Training

    data_c1 = 'zeros'
    data_c2 = 'zeros'
    data_c3 = 'zeros'

    data_Lx = (l0/sigma0)*bodyfx(input_data)
    data_Ly = (l0/sigma0)*bodyfy(input_data)

    # The normal stress values in x and y should be satisfied by PINN outputs at the lateral and top-bottom boundaries, respectively.
    # The rest of the targets (physical equations) are met over the entire domain:
    target_data = [(leftright_ids, data_d1),
                   (topbottom_ids, data_d2),
                   data_c1,
                   data_c2,
                   data_c3,
                   data_Lx, data_Ly]

    # Train the model
    training_time = time.time()
    history = model.train(
        x_true=input_data,
        y_true=target_data,
        epochs=args.epochs[0],
        batch_size=args.batchsize[0],
        shuffle=args.shuffle[0],
        learning_rate=args.learningrate[0],
        stop_after=args.stopafter[0],
        verbose=args.verbose[0],
    )
    training_time = time.time() - training_time

    for loss in history.history:
        np.savetxt(fname+"_{}".format("_".join(loss.split("/"))),
                   np.array(history.history[loss]).reshape(-1, 1))

    time_steps = np.linspace(0, training_time, len(history.history["loss"]))
    np.savetxt(fname+"_Time", time_steps.reshape(-1, 1))

    # Return network outputs at input data coordinates
    input_plot = input_data

    # Each non-dimensional output needs to be converted to dimensional form using the corresponding characteristic scale
    lame1_pred = sigma0*lame1.eval(model, input_plot)
    lame2_pred = sigma0*lame2.eval(model, input_plot)

    # Convert Lamé parameters to Young's (elastic) modulus and Poisson's ratio
    Youngs_pred = lame2_pred * \
        (3*lame1_pred + 2*lame2_pred)/(lame1_pred+lame2_pred)
    Poissons_pred = lame1_pred/(2*(lame1_pred+lame2_pred))

    # Because this specific problem was plane stress loading and our PINN formulation is written in plane strain form, a further conversion step is required:
    Youngs_pred = Youngs_pred/(1-Poissons_pred**2)
    Poissons_pred = Poissons_pred/(1-Poissons_pred)

    # Finally convert stresses to dimensional form:
    Sxx_pred = sigma0*Sxx.eval(model, input_plot)
    Syy_pred = sigma0*Syy.eval(model, input_plot)
    Sxy_pred = sigma0*Sxy.eval(model, input_plot)

    # Save outputs as 1D vectors at collocation points
    np.savetxt(os.path.join(
        args.outputpath[0], 'Sxx_pred.txt'), Sxx_pred, delimiter=', ', fmt='%1.6e')
    np.savetxt(os.path.join(
        args.outputpath[0], 'Syy_pred.txt'), Syy_pred, delimiter=', ', fmt='%1.6e')
    np.savetxt(os.path.join(
        args.outputpath[0], 'Sxy_pred.txt'), Sxy_pred, delimiter=', ', fmt='%1.6e')
    np.savetxt(os.path.join(
        args.outputpath[0], 'Youngs_pred.txt'), Youngs_pred, delimiter=', ', fmt='%1.6e')
    np.savetxt(os.path.join(
        args.outputpath[0], 'Poissons_pred.txt'), Poissons_pred, delimiter=', ', fmt='%1.6e')


# Plot loss terms
def plot():
    # Plot the loss evolution vs. epochs
    output_file_name = os.path.join(args.outputpath[0], args.outputprefix[0])
    fname = output_file_name + \
        "_{}_".format(args.actf[0]) + "x".join([str(x) for x in args.layers])

    loss = np.loadtxt(fname+"_loss")
    time = np.loadtxt(fname+"_Time")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=300)
    cust_semilogx(ax[0], None, loss/loss[0], "epochs", "L/L0")
    cust_semilogx(ax[1], time, loss/loss[0], "time(s)", None)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15,
                        top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("{}_loss.png".format(output_file_name))


train()
plot()
