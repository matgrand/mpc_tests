import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *

# single pendolum
from single_pendolum import *

# Initial conditions
x0 = np.pi + np.random.rand()*0.1 # [rad] intial angle (0 is up)
dx0 = 0
# Time
dt = 0.01 # [s] time step
simT = 1 # [s] simulation time
fps = 60 # [Hz] frames per second

INPUT_SIZE = int(8 * simT)  # number of control inputs

###############################
# MODEL PREDICTIVE CONTROL
###############################
# cost function
kt  = 100 # kinetic energy weight
kv  = -100 # potential energy weight
keu = 5 # control expanded input weight
costs = [[],[],[]]


def cost(x, eu, append=False):
    '''Cost function'''
    n = len(x) # number of time steps
    weights = np.linspace(0, 1, n)#**2 # weights for the cost function
    t = kinetic_energy(x) # kinetic energy
    v = potential_energy(x) # potential energy
    te = kt * t * weights
    ve = kv * v * weights 

    eu = keu * eu**2 * np.linspace(0, 1, len(eu)) # weight for the control input

    # debug, append the energies
    if append: costs[0].append(te), costs[1].append(ve), costs[2].append(eu)

    return np.sum(te) + np.sum(ve) + np.sum(eu) # total cost

# GRADIENT DESCENT
# optimize the control input to minimize the cost function
ITERATIONS = 3000#1000
# u = np.zeros(INPUT_SIZE) # control input
u = np.random.rand(INPUT_SIZE)*INPUT_CLIP - INPUT_CLIP/2 # control input
pert = 1e-3 # perturbation of the control input for the gradient, will be updated
ss = np.linspace(0.1, 0.003, len(u)) # step size for the gradient
# ss = np.ones(len(u))*3e-3 # step size for the gradient

u_time_weight = 5*np.linspace(1, 0, INPUT_SIZE)**2 # weight for the control input

best_J = np.inf
best_u = np.zeros_like(u)

for i in tqdm(range(ITERATIONS), ncols=50):
    assert u.shape == (INPUT_SIZE,), f'u.shape: {u.shape}, INPUT_SIZE: {INPUT_SIZE}'
    x,t,eu = simulate(x0, dx0, simT, dt, u) # simulate the pendulum
    J = cost(x, eu, append=True) # calculate the cost
    if J < best_J: best_J, best_u = J, u
    # calculate the gradient
    Jgrad = np.zeros(len(u)) # initialize the gradient 
    for j in range(len(u)):
        up = np.copy(u)
        up[j] += pert*u_time_weight[j] # perturb the control input
        xp, tp, eup = simulate(x0, dx0, simT, dt, up) # simulate the pendulum
        Jgrad[j] = (cost(xp, eup) - J)/ss[j] # calculate the gradient
    # update the control input
    u = u - Jgrad*ss[j] # update the control input
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
a1 = animate_pendulum(x, eu, dt, fps, l, figsize=(4,4))
a2 = animate_costs(np.array(costs), figsize=(8,6))
plt.show()
################################################################################################