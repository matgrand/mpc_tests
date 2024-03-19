import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *

SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True

if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

#initial state: [angle, angular velocity]
if SP: x0 = np.array([np.random.rand()*0.5 + np.pi, 0]) # [rad, rad/s] # SINGLE PENDULUM
if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
if CDP: raise NotImplementedError('Cart double pendulum not implemented')
# Time
dt = 0.01 # [s] time step
simT = 1 # [s] simulation time

if SP: INPUT_SIZE = int(10 * simT)  # number of control inputs
if DP: INPUT_SIZE = int(100 * simT)  # number of control inputs


ITERATIONS = 1000 #1000

print(f'input size: {INPUT_SIZE}')
print(f'iterations: {ITERATIONS}')

###############################
# MODEL PREDICTIVE CONTROL
###############################
# cost function
if SP:
    kt  = 60 # kinetic energy weight MIN
    kv  = -100 # potential energy weight MAX
    keu = 2 # control expanded input weight MIN
    costs = [[],[],[]] # costs to plot later
    labels = ['T', 'V', 'u']
    def cost(x, eu, append=False):
        '''Cost function'''
        n = len(x) # number of time steps
        weights = np.linspace(0, 1, n)**3 # weights for the cost function
        t = kinetic_energy(x) # kinetic energy
        v = potential_energy(x) # potential energy
        te = kt * t * weights
        ve = kv * v * weights 
        eu = keu * eu**2 * np.linspace(0, 1, len(eu)) # weight for the control input
        # debug, append the energies
        if append: costs[0].append(te), costs[1].append(-ve), costs[2].append(eu)
        final_cost = (np.sum(te) + np.sum(ve) + np.sum(eu)) / n # total cost
        return final_cost
if DP:
    kt  = 60 # kinetic energy weight MIN
    kv  = -100 # potential energy weight MAX
    keu = 2 # control expanded input weight MIN
    costs = [[],[],[]] # costs to plot later
    labels = ['T', 'V', 'u']
    def cost(x, eu, append=False):
        '''Cost function'''
        n = len(x) # number of time steps
        weights = np.linspace(0, 1, n)**3 # weights for the cost function
        t = kinetic_energy(x) # kinetic energy
        v = potential_energy(x) # potential energy
        te = kt * t * weights
        ve = kv * v * weights 
        eu = keu * eu**2 * np.linspace(0, 1, len(eu)) # weight for the control input
        # debug, append the energies
        if append: costs[0].append(te), costs[1].append(-ve), costs[2].append(eu)
        return (np.sum(te) + np.sum(ve) + np.sum(eu)) / n # total cost

# "SOLVER"
# optimize the control input to minimize the cost function
u = np.zeros(INPUT_SIZE) # control input
#perturbations for each control input, bigger changes for earlier control inputs
if SP: pert = np.linspace(3e-1, 3e-2, INPUT_SIZE) 
if DP: pert = np.linspace(5, 5, INPUT_SIZE)
pd = 1 #0.999 # perturbation decay, 1 -> no decay
print(f'perturbation: {pd} -> {pd**ITERATIONS}')
if SP: lr = 3e-2 # learning rate for the gradient descent
if DP: lr = 1e-7 # learning rate for the gradient descent
# initialize the best cost and control input
best_J = np.inf
best_u = np.zeros_like(u)
# gradient descent
for i in tqdm(range(ITERATIONS), ncols=50):
    assert u.shape == (INPUT_SIZE,), f'u.shape: {u.shape}, INPUT_SIZE: {INPUT_SIZE}'
    x,t,eu = simulate(x0, simT, dt, u) # simulate the pendulum
    J = cost(x, eu, append=True) # calculate the cost
    if J < best_J: best_J, best_u = J, u
    # calculate the gradient
    Jgrad = np.zeros(INPUT_SIZE) # initialize the gradient 
    for j in range(INPUT_SIZE):
        up = np.copy(u)
        up[j] += pert[j] * pd**i # perturb the control input
        xp, tp, eup = simulate(x0, simT, dt, up) # simulate the pendulum
        Jgrad[j] = (cost(xp, eup) - J) # calculate the gradient
    u -= Jgrad*lr # update the control input
    if i%7 == 0: print(f'cost: {J:.2f}', end='\r')
u = best_u
print(f'iteration {i+1}/{ITERATIONS}, cost: {best_J:.2f}')

# SIMULATION 
################################################################################################
# Simulate the pendulum
x,t,eu = simulate(x0, simT, dt, u) # simulate the pendulum

J = cost(x, eu) # calculate the cost
print(f'cost: {J:.2f}')
print(f'u: {u}')

# calculate the energies
T = kinetic_energy(x) # kinetic energy
V = potential_energy(x) # potential energy 
################################################################################################

##  PLOTTING

# plot the state and energies
if SP:
    plot_single(x, t, eu, T, V, figsize=(10,8))
    a1 = animate_pendulum(x, eu, dt, l, figsize=(4,4))
    a2 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
if DP:
    plot_double(x, t, eu, T, V, figsize=(10,8))
    a1 = animate_double_pendulum(x, eu, dt, l1, l2, figsize=(4,4))
    a2 = animate_costs(np.array(costs), labels=labels, figsize=(6,4))

plt.show()
################################################################################################