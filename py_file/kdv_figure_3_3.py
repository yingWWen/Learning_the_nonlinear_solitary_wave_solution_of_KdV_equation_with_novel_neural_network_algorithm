from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import autograd.numpy.random as npr

import autograd.numpy as np
from Lie_NN.lieneuralsolver import NNSolver
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

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

    X = y[0]
    Y = y[1]
    Z = y[2]

    return [Y, Z, a * Y - b * X * Y]


t = np.linspace(-3, 3, 250).reshape(-1, 1)



y0_list = [0.05586144, 0.07676177, 0.10236137]
sol = solve_ivp(f, [t.min(), t.max()], y0_list,
                t_eval=t.ravel(), method='Radau', rtol=1e-5)

bwith = 0.5
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.scatter(sol.t, sol.y[0], s=2, marker='_', alpha=0.5,  label='$u_1$')
plt.scatter(sol.t, sol.y[1], s=2,  marker='o', alpha=0.5, label='$u_2$')
plt.scatter(sol.t, sol.y[2], s=2, marker = '+', alpha=0.5,  label = '$u_3$')
plt.grid(which='major')

plt.tick_params(top=False, bottom=True, left=True, right=False)
plt.tick_params(which='both', direction='in', width=0.5, length=2)

plt.axis([-3, 3, -1, 1])
plt.xticks(np.arange(-3, 4, step=1))
plt.yticks(np.arange(-1.5, 1.5, step=0.25))


plt.xlabel('$t$', fontsize=10)
plt.ylabel('$u$', fontsize=10)
plt.legend()
plt.show()


nn = NNSolver(f, t, y0_list, n_hidden=30)
print(nn)

nn.reset_weights()

nn.train(maxiter=2000, iprint=100)

nn.loss_func(nn.params_list)

print(nn.loss_func(nn.params_list))
print(nn.params_list[0])


y_pred_list, dydt_pred_list = nn.predict()
plt.figure(figsize= (8, 6))
bwith = 0.5
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.grid(which='major')


plt.tick_params(top=False, bottom=True, left=True, right=False)

plt.tick_params(which='both', direction='in', width=0.5, length=2)
plt.axis([-3, 3, -1, 1])
plt.xticks(np.arange(-3, 4, step=1))
plt.yticks(np.arange(-0.5, 1.5, step=0.3))
plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.plot(t, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.legend(loc = "best")
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.xlim((-3, 3.1))
plt.show()



np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(3)])
print(np.mean([sqrt(mean_squared_error(sol.y[i], y_pred_list[i])) for i in range(3)]))


t_test = np.linspace(-3, 3.5, 250).reshape(-1,1)
sol = solve_ivp(f, [t_test.min(), t_test.max()], y0_list, method='Radau', rtol=1e-5)
y_pred_list, dydt_pred_list = nn.predict(t=t_test)
plt.figure(figsize=(8, 6))

bwith = 0.5
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.grid(which='major')

plt.tick_params(top=False, bottom=True, left=True, right=False)

plt.tick_params(which='both', direction='in', width=0.5, length=2)
plt.axis([-3, 3.3, -1, 1])
plt.xticks(np.arange(-3, 3.5, step=1))
plt.yticks(np.arange(-0.5, 1.5, step=0.3))



plt.plot(sol.t, sol.y[0], label='Exact solution - $u$')
plt.plot(t_test, y_pred_list[0], 'o', label='Neural solution - $\hat{u}_1$')
plt.axvline(x = 3 , linestyle = '--')
plt.legend(loc = 'best')
plt.xlabel(r'$\xi$', fontsize=10)
plt.ylabel('Values', fontsize=10)
plt.show()

def plot_loss_compare(model, maxiter=2000, iprint=100):
    model.reset_weights()
    save_init_paras_list = model.params_list
    save_init_flattened_params = model.flattened_params

    methods = ['BFGS']
    plt.figure(figsize=(8, 6))
    for method in methods:
        model.params_list = save_init_paras_list
        model.flattened_params = save_init_flattened_params

        start_time = time.time()
        model.train(method=method, maxiter=maxiter, iprint=iprint)
        duration = time.time() - start_time

        plt.semilogy(range(len(model.loss)), model.loss, label=method + ' (time = {}s'.format(round(duration, 2)) + ')')

    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.show()
plot_loss_compare(nn)


