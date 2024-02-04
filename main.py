

# Commented out IPython magic to ensure Python compatibility.
import autograd.numpy as np
from autograd import grad, jacobian
import autograd.numpy.random as npr

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline

from visual_PDE import plot_3Dsurface
from convection_basic import linear_convection_solve

c = 1.0
Lx = 1.0

nx = 40
ny = 5
Lt = ny*0.025


u,u2D = linear_convection_solve(c,Lx,nx+1,Lt,ny)

dx = Lx / nx
dy = Lt / ny
x_space = np.linspace(0, Lx, nx)
y_space = np.linspace(0, Lt, ny)


def analytic_solution(x):
    ix = int(np.where(x_space == x[0])[0])
    iy = int(np.where(y_space == x[1])[0])
    ansol = u2D[iy][ix]
    # if not isinstance(t,np.float64):
    #     qq = 0
    return ansol



# def analytic_solution(x):
#
#
#     return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * \
#     		np.sin(np.pi * x[0]) * (np.exp(np.pi * x[1]) - np.exp(-np.pi * x[1]))
surface = np.zeros((nx, ny))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        tt = analytic_solution([x, y])
        surface[i][j] = tt

plot_3Dsurface(x_space,y_space,surface.T,'X','Y','Analytical solution')
# surf = ax.plot_surface(X, Y, surface.T, rstride=1, cstride=1, cmap=cm.viridis,
#         linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 2)
#
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$');

def f(x):
    return 0.

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])


def neural_network_x(x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

def A(x):
    return analytic_solution(x)


def psy_trial(x, net_out):
    return A(x) + x[0] * (Lx - x[0]) * x[1] * (Lt - x[1]) * net_out


def loss_function(W, x, y):
    loss_sum = 0.

    for i_x,xi in enumerate(x):
        for i_y,yi in enumerate(y):

            input_point = np.array([xi, yi])

            net_out = neural_network(W, input_point)[0]
            loss_sum += (u2D[i_y][i_x] - net_out)**2
    return loss_sum

#TODO: check A(x) function that it gives the necessary form of boundary conditions
#TODO^ check the derivatives and sqr value at each point with the exact value
#TODO: it cannot be the same value but they must converge with epoch
def loss_function1(W, x, y):
    loss_sum = 0.

    for xi in x:
        for yi in y:

            input_point = np.array([xi, yi])

            net_out = neural_network(W, input_point)[0]

            net_out_jacobian = jacobian(neural_network_x)(input_point)
            net_out_hessian = jacobian(jacobian(neural_network_x))(input_point)

            psy_t = psy_trial(input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial)(input_point, net_out)
            psy_t_hessian = jacobian(jacobian(psy_trial))(input_point, net_out)

            gradient_of_trial_dx = psy_t_jacobian[0]
            gradient_of_trial_dy = psy_t_jacobian[1]

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            func = f(input_point) # right part function

            err_sqr = ((gradient_of_trial_dx + gradient_of_trial_dy) - func)**2
            loss_sum += err_sqr

    return loss_sum

W = [npr.randn(2, 10), npr.randn(10, 1)]
lmb = 0.01

print(neural_network(W, np.array([1, 1])))
loss = 1e3
i = 0
while loss > 5e-2:
    loss_grad =  grad(loss_function)(W, x_space, y_space)
    loss_grad1 =  grad(loss_function1)(W, x_space, y_space)

    W[0] = W[0] - lmb * loss_grad1[0]
    W[1] = W[1] - lmb * loss_grad1[1]
    loss = loss_function1(W, x_space, y_space)

    print(i,loss_function(W, x_space, y_space),loss)
    i = i + 1

surface2 = np.zeros((nx, ny))
surface = np.zeros((nx, ny))

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        surface[i][j] = analytic_solution([x, y])

for i, x in enumerate(x_space):
    for j, y in enumerate(y_space):
        net_outt = neural_network(W, [x, y])[0]
        surface2[i][j] = psy_trial([x, y], net_outt)


print(surface[2])
print(surface2[2])

# plot_3Dsurface(x_space,y_space,surface.T,'X','Y','Analytical solution')

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(x_space, y_space)
# surf = ax.plot_surface(X, Y, surface.T, rstride=1, cstride=1, cmap=cm.viridis,
#         linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 3)
#
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$');


plot_3Dsurface(x_space,y_space,surface2.T,'X','Y','NN solution')
qq = 0

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# X, Y = np.meshgrid(x_space, y_space)
# surf = ax.plot_surface(X, Y, surface2.T, rstride=1, cstride=1, cmap=cm.viridis,
#         linewidth=0, antialiased=False)

# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 3)
#
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$');

