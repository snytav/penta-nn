import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot, cm


def get_XY_layer(pde,x_space,y_space,t_space,nt,psy_trial):
    nx = x_space.shape[0]
    ny = y_space.shape[0]
    surface2 = np.zeros((ny, nx))
    t0 = t_space[nt]
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            net_outt = pde.forward(torch.Tensor([x, y,t0]))
            surface2[i][j] = psy_trial([x, y,t0], net_outt)

    return surface2

import matplotlib.pyplot as plt

def plot_3Dsurface(x_space,y_space,surface2,x_name,y_name,name):
    # fig = plt.figure()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x_space, y_space)
    surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 3)

    fig.colorbar(surf, ax=ax)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name);
    plt.title(name)
    plt.savefig(name+'.png')

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, surface2,cmap='coolwarm')
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title(name)
    ax.set_xlabel(y_name)
    ax.set_ylabel(x_name)
    plt.savefig(name + '_contourf_2D_.png')


def make_surface_from_function(x,v,t,func):

    p = np.zeros((x.shape[0],v.shape[0]))
    for i, xi in enumerate(x):
        for k, yi in enumerate(v):
            p[i][k] = func(torch.Tensor([xi, yi, t])).item()

    return p

def get_analytic_and_trial_solution_2D(x_space,y_space,analytic_solution,psy_trial,pde):
    surface = np.zeros((x_space.shape[0],y_space.shape[0]))
    surface2 = np.zeros((x_space.shape[0], y_space.shape[0]))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = analytic_solution([x, y])

    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            net_outt = pde.forward(torch.Tensor([x, y,0.0]))
            surface2[i][j] = psy_trial([x, y], net_outt)

    return surface,surface2
