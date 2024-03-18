import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *

# single pendolum
from single_pendolum import *

# Initial conditions
x0 = np.random.rand()*0.5 + np.pi # [rad] intial angle (0 is up)
dx0 = 0
# Time
dt = 0.01 # [s] time step
simT = 1 # [s] simulation time

INPUT_SIZE = int(20 * simT)  # number of control inputs

###############################
# MODEL PREDICTIVE CONTROL
###############################
# cost function
kt  = 60 # kinetic energy weight
kv  = -100 # potential energy weight
keu = 2 # control expanded input weight
costs = [[],[],[]] # costs to plot later
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
    return np.sum(te) + np.sum(ve) + np.sum(eu) # total cost

# "SOLVER"
# optimize the control input to minimize the cost function
ITERATIONS = 1000#1000
u = np.zeros(INPUT_SIZE) # control input
#perturbations for each control input, bigger changes for earlier control inputs
pert = np.linspace(3e-2, 3e-4, INPUT_SIZE) 
pd = 0.999 # perturbation decay
print(f'perturbation: {pd} -> {pd**ITERATIONS}')
lr = 1e-1 # learning rate for the gradient descent
# initialize the best cost and control input
best_J = np.inf
best_u = np.zeros_like(u)
# gradient descent
for i in tqdm(range(ITERATIONS), ncols=50):
    assert u.shape == (INPUT_SIZE,), f'u.shape: {u.shape}, INPUT_SIZE: {INPUT_SIZE}'
    x,t,eu = simulate(x0, dx0, simT, dt, u) # simulate the pendulum
    J = cost(x, eu, append=True) # calculate the cost
    if J < best_J: best_J, best_u = J, u
    # calculate the gradient
    Jgrad = np.zeros(INPUT_SIZE) # initialize the gradient 
    for j in range(INPUT_SIZE):
        up = np.copy(u)
        up[j] += pert[j] * pd**i # perturb the control input
        xp, tp, eup = simulate(x0, dx0, simT, dt, up) # simulate the pendulum
        Jgrad[j] = (cost(xp, eup) - J) # calculate the gradient
    u -= Jgrad*lr # update the control input
    if i%7 == 0: print(f'cost: {J:.2f}', end='\r')
u = best_u
print(f'iteration {i+1}/{ITERATIONS}, cost: {best_J:.2f}')

# SIMULATION 
################################################################################################
# Simulate the pendulum
x,t,eu = simulate(x0, dx0, simT, dt, u) # simulate the pendulum

J = cost(x, eu) # calculate the cost
print(f'cost: {J:.2f}')
print(f'u: {u}')

# calculate the energies
T = kinetic_energy(x) # kinetic energy
V = potential_energy(x) # potential energy 
################################################################################################

##  PLOTTING
# plot the state and energies
plot_single(x, t, eu, T, V, figsize=(12,10))
#animations
a1 = animate_pendulum(x, eu, dt, l, figsize=(4,4))
a2 = animate_costs(np.array(costs), labels=['T', 'V', 'u'])
plt.show()
################################################################################################