import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu #expand input function
 
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
if SP: x0 = np.array([0.2 + np.pi 
                      , 0]) # [rad, rad/s] # SINGLE PENDULUM
if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
if CDP: raise NotImplementedError('Cart double pendulum not implemented')
# Time
dt = 0.01 # [s] time step
simT = 2 # [s] simulation time

if SP: INPUT_SIZE = int(16)  # number of control inputs
if DP: INPUT_SIZE = int(16)  # number of control inputs


ITERATIONS = 300 #1000
CLIP = True


print(f'input size: {INPUT_SIZE}')
print(f'iterations: {ITERATIONS}')

###############################
# MODEL PREDICTIVE CONTROL
###############################
# cost function
if SP:
    kt  = 60 #60 # kinetic energy weight MIN
    kv  = -100 #-100 # potential energy weight MAX
    keu = 2 #2 # control expanded input weight MIN
    costs = [[],[],[]] # costs to plot later
    labels = ['T', 'V', 'u']
    def cost(x, eu, append=False):
        '''Cost function'''
        n = len(x) # number of time steps
        t = kinetic_energy(x) # kinetic energy
        v = potential_energy(x) # potential energy
        te = kt * t
        ve = kv * v 
        eu = keu * eu**2 * np.linspace(0, 1, len(eu))#**2 # weight for the control input
        # debug, append the energies
        if append: costs[0].append(te), costs[1].append(-ve), costs[2].append(eu)
        final_cost = np.sum(te) + np.sum(ve) + np.sum(eu)
        return final_cost / n
if DP:
    kt  = 60 #60 # kinetic energy weight MIN
    kv  = -100 #-100 # potential energy weight MAX
    keu = 2 #2 # control expanded input weight MIN
    costs = [[],[],[]] # costs to plot later
    labels = ['T', 'V', 'u']
    def cost(x, eu, append=False):
        '''Cost function'''
        n = len(x) # number of time steps
        t = kinetic_energy(x) # kinetic energy
        v = potential_energy(x) # potential energy
        te = kt * t
        ve = kv * v 
        eu = keu * eu**2 * np.linspace(0, 1, len(eu))#**2 # weight for the control input
        # debug, append the energies
        if append: costs[0].append(te), costs[1].append(-ve), costs[2].append(eu)
        final_cost = np.sum(te) + np.sum(ve) + np.sum(eu)
        return final_cost / n

# "SOLVER"
# optimize the control input to minimize the cost function
u = np.zeros(INPUT_SIZE) # control input
#perturbations for each control input, bigger changes for earlier control inputs
pd = 1 #0.999 # perturbation decay, 1 -> no decay
print(f'perturbation: {pd} -> {pd**ITERATIONS}')
if SP: lr = .1 # learning rate for the gradient descent
if DP: lr = 1e-7 # learning rate for the gradient descent

nt = int(simT/dt) # number of time steps

xs = np.zeros((ITERATIONS, nt, len(x0))) # state vectors
us, Ts, Vs = [np.zeros((ITERATIONS, nt)) for _ in range(3)] # control inputs, kinetic and potential energies

def grad(u, c):
    '''Calculate the gradient'''
    d = np.zeros(INPUT_SIZE) # initialize the gradient 
    for j in range(INPUT_SIZE):
        up = np.copy(u)
        up[j] += lr * pd**i # perturb the control input
        xp, _, eup = simulate(x0, simT, dt, up, expu, CLIP) # simulate the pendulum
        d[j] = (cost(xp, eup) - c) # calculate the gradient
    return d

# GRADIENT DESCENT

# first iteration
x,t,eu = simulate(x0, simT, dt, u, expu, CLIP) # simulate the pendulum
xs[0], us[0], Ts[0], Vs[0] = x, eu, kinetic_energy(x), potential_energy(x)
J = cost(x, eu, append=True) # calculate the cost
prev_J = J

for i in tqdm(range(1,ITERATIONS), ncols=60):
    # calculate the gradient
    Jgrad = grad(u, J) 
    new_u = u - Jgrad*lr # update the control input
    # simulate the pendulum
    x,t,eu = simulate(x0, simT, dt, new_u, expu, CLIP) 
    # save the state and control input
    xs[i], us[i], Ts[i], Vs[i] = x, eu, kinetic_energy(x), potential_energy(x) 
    # calculate the cost
    new_J = cost(x, eu, append=True) 

    if new_J < J: # decreasing cost
        u, J = new_u, new_J # update the control input and cost 
        lr *= 1.1 # increase the learning rate
    else: # increasing cost
        lr *= 0.95 # decrease the learning rate
        if lr < 1e-10: print(f'learning rate too small, breaking...'); break

    if i%1 == 0: print(f'cost: {J:.2f}, lr: {lr:.1e}', end='\r')
# u = best_u
print(f'iteration {i+1}/{ITERATIONS}, cost: {J:.2f}')

# SIMULATION 
################################################################################################
# Simulate the pendulum
J = cost(x, eu) # calculate the cost
print(f'cost: {J:.2f}')
print(f'u: {u}')

x,t,eu = simulate(x0, simT, dt, u, expu, CLIP) # simulate the pendulum


# calculate the energies
T = kinetic_energy(x) # kinetic energy
V = potential_energy(x) # potential energy 
################################################################################################

##  PLOTTING

# plot the state and energies
if SP:
    a2 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
    xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
    print(f'xs1: {xs1.shape}, xs2: {xs2.shape}, us: {us.shape}, Ts: {Ts.shape}, Vs: {Vs.shape}')
    to_plot = np.array([xs1, xs2, us, Ts, Vs])
    a3 = general_multiplot_anim(to_plot, t, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
    # plot_single(x, t, eu, T, V, figsize=(10,8))
    #extended simulation
    x,t,eu = simulate(x0, simT, dt, u, expu, CLIP, continue_for=2*simT) # simulate the pendulum
    ap1 = animate_pendulum(x, eu, dt, l, figsize=(4,4), title='dt=0.01')
    # dt = 0.0001
    # x,t,eu = simulate(x0, simT, dt, u, expu, CLIP, continue_for=2*simT) # simulate the pendulum
    # ap2= animate_pendulum(x, eu, dt, l, figsize=(4,4), title='dt=0.0001')
    # dt=0.1
    # ap3 = animate_pendulum(x, eu, dt, l, figsize=(4,4), title='dt=0.1')
    # x,t,eu = simulate(x0, simT, dt, u, expu, CLIP, continue_for=2*simT) # simulate the pendulum

if DP:
    plot_double(x, t, eu, T, V, figsize=(10,8))
    a1 = animate_double_pendulum(x, eu, dt, l1, l2, figsize=(4,4))
    a2 = animate_costs(np.array(costs), labels=labels, figsize=(6,4))

plt.show()
################################################################################################