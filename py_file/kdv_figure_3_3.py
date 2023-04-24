# Importing the necessary packages
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import autograd.numpy.random as npr

import autograd.numpy as np
from Lie_NN.lieneuralsolver import NNSolver  #Note the location of the folder
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

# Set picture parameters that can be modified
plt.rcParams["figure.figsize"] = (6,4)
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 10


def f(t, y):
    # kdv equation
    a = 2
    b = 6
    X = y[0]  #u1
    Y = y[1]  #u2
    Z = y[2]  #u3
    return [Y, Z, a * Y - b * X * Y]

# Training Points t is \xi
t = np.linspace(-3, 3, 250).reshape(-1, 1)

# Initial value (boundary -3)
y0_list = [0.05586144, 0.07676177, 0.10236137]
# Numerical solver
sol = solve_ivp(f, [t.min(), t.max()], y0_list,
                t_eval=t.ravel(), method='Radau', rtol=1e-5)
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$u$', fontsize=10)
plt.plot(sol.t, sol.y[0],  label='$u_1$')
plt.plot(sol.t, sol.y[1],   label='$u_2$')
plt.plot(sol.t, sol.y[2],   label='$u_3$')
plt.legend()
plt.show()  #Draw graph

############################# Solver for the proposed method
nn = NNSolver(f, t, y0_list, n_hidden=30)  #Number of neurons in the hidden layer 30
print(nn)

nn.reset_weights()
nn.train(maxiter=2000, iprint=100)   # Maximum number of iterations
nn.loss_func(nn.params_list)
print(nn.loss_func(nn.params_list))  # Print the optimal parameters
print(nn.params_list[0])             # Print the optimal parameters of network 1
nn.plot_loss()                       # Number of iterations and loss function image

############################ The output of the constructed solution in the training set
y_pred_list, dydt_pred_list = nn.predict()
plt.figure(figsize= (8, 6))
plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.plot(t, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.legend(loc = "best")
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.xlim((-3, 3.1))
plt.show()

############################# Calculate the mean square error on the training set
np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(3)])
print(np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(3)]))

############################# Output of the constructed solution on the test set
t_test = np.linspace(-3, 3.5, 250).reshape(-1,1)
sol = solve_ivp(f, [t_test.min(), t_test.max()], y0_list, method='Radau', rtol=1e-5)
y_pred_list, dydt_pred_list = nn.predict(t=t_test)
plt.figure(figsize=(8, 6))
plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.plot(t_test, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.axvline(x = 3 , linestyle = '--')
plt.legend(loc = 'best')
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.show()



