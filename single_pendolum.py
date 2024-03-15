import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm
from plotting import animate_pendulum, C
np.random.seed(42)

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.3 # [kg/s] damping coefficient

INPUT_MAX = 50 # maximum control input

# Initial conditions
x0 = np.pi # [rad] initial angle
dx0 = 0
# Time
dt = 0.01 # [s] time step
simT = 1.5 # [s] simulation time
fps = 60 # [Hz] frames per second

nt = int(simT/dt) # number of time steps
INPUT_SIZE = int(8 * simT)  # number of control inputs

# calculate the dynamics using symbolic math
t = sp.symbols('t')
θ = sp.symbols('θ', cls=sp.Function)(t) # angle (state)
u = sp.symbols('u', cls=sp.Function)(t) # control input
dθ = θ.diff(t) # angular velocity
ddθ = dθ.diff(t) # angular acceleration

# let's use the Lagrangian
x, y = l*sp.sin(θ), l*sp.cos(θ) # position of the pendulum
dx, dy = x.diff(t), y.diff(t) # velocity of the pendulum
T = 1/2 * m * (dx**2 + dy**2) # Kinetic energy
min_V = -m*g*l # minimum potential energy, to set the zero level
V = m*g*y - min_V # Potential energy
L = T - V # Lagrangian

# Euler-Lagrange equation
leq = L.diff(θ) - (L.diff(dθ)).diff(t) -μ*dθ + u # Euler-Lagrange equation

# Solve the differential equation
ddθ = sp.solve([leq], [ddθ], simplify=False)[0] # solve the differential equation for the acceleration

# lambdify to use with numpy
fddθ = sp.lambdify((θ, dθ, u), ddθ, 'numpy') 
fT = sp.lambdify((θ, dθ), T, 'numpy')
fV = sp.lambdify((θ, dθ), V, 'numpy') 
del t, θ, dθ, leq, T, V, L, x, y, u # delete the symbolic variables

def expand_input(iu, nto): # iu: input, nto: number of time steps in the output
    '''Expand the control input to match the state vector'''
    liu = len(iu) # length of the compressed input
    ou = np.zeros((nt)) # expanded control input
    it = np.linspace(0, simT, liu) # input time
    ot = np.linspace(0, simT, nto) # expanded time
    ii = 0 # index for the compressed input
    for i in range(nto):
        ia, ib = it[ii], it[ii+1] # time interval for the compressed input
        a, b = iu[ii], iu[ii+1] # control input interval
        ou[i] = a + (ot[i] - ia)*(b - a)/(ib - ia) # linear interpolation
        if ot[i] > it[ii+1]: ii += 1 # update the index
    return ou

# def expand_input(iu, nto): # iu: input, nto: number of time steps in the output
#     ''' Use input as Fourier coefficients and create a control input'''
#     liu = len(iu) # length of the compressed input
#     MAX_FREQ = 100 # maximum frequency [Hz]
#     ou = np.zeros((nt)) # expanded control input
#     freqs = np.linspace(0, MAX_FREQ, liu) # frequencies
#     ot = np.linspace(0, simT, nto) # expanded time
#     for i in range(liu):
#         ou += iu[i]*np.sin(2*np.pi*freqs[i]*ot)
#     return ou

def step(x, u, dt): 
    '''Integrate the differential equation using the Euler method'''
    θ, dθ = x # split the state vector
    dθ = dθ + fddθ(θ, dθ, u)[0]*dt # integrate the acceleration
    θ = θ + dθ*dt # integrate the velocity
    return np.array([θ, dθ]) # return the new state vector

#simulate a run
def simulate(x0, dx0, simT, dt, u):
    '''Simulate the pendulum'''
    t = np.arange(0, simT, dt)
    x = np.zeros((len(t), 2)) # [θ, dθ] -> state vector
    eu = expand_input(u, len(t))
    eu = np.clip(eu, -INPUT_MAX, INPUT_MAX) # clip the control input
    x[0] = [x0, dx0] # initial conditions
    for i in range(1, len(t)):
        x[i] = step(x[i-1], eu[i], dt)
    return x, t 

###############################
# MODEL PREDICTIVE CONTROL
###############################
# cost function
kt  = 100 # kinetic energy weight
kv  = -100 # potential energy weight
ku = 0 # control input weight
def cost(x, u):
    '''Cost function'''
    n = len(x) # number of time steps
    weights = np.linspace(0, 1, n)**2 # weights for the cost function
    # weights = np.clip(weights, 0, weights[n//3]) # clip the weights
    t = fT(*x.T) # kinetic energy
    v = fV(*x.T) # potential energy
    te = np.sum(t*weights) 
    ve = np.sum(v*weights) 
    # u = expand_input(u, n)
    # uw = np.sum((u*weights)**2) # control input weight

    return kt*te + kv*ve # cost function

# GRADIENT DESCENT
# optimize the control input to minimize the cost function
iterations = 3000
# u = np.zeros(INPUT_SIZE) # control input
u = np.random.rand(INPUT_SIZE)*INPUT_MAX*2 - INPUT_MAX # control input
pert = 1e-3 # perturbation of the control input for the gradient, will be updated
ss = np.linspace(0.1, 0.003, len(u)) # step size for the gradient
# ss = np.ones(len(u))*0.01 # step size for the gradient

best_J = np.inf
best_u = np.zeros_like(u)

for i in tqdm(range(iterations), ncols=50):
    assert u.shape == (INPUT_SIZE,), f'u.shape: {u.shape}, INPUT_SIZE: {INPUT_SIZE}'
    x,t = simulate(x0, dx0, simT, dt, u) # simulate the pendulum
    J = cost(x, u) # calculate the cost
    if J < best_J: best_J, best_u = J, u
    # calculate the gradient
    Jgrad = np.zeros(len(u)) # initialize the gradient 
    for j in range(len(u)):
        up = np.copy(u)
        up[j] += pert # perturb the control input
        xp, tp = simulate(x0, dx0, simT, dt, up) # simulate the pendulum
        Jgrad[j] = (cost(xp, up) - J)/ss[j] # calculate the gradient
    # update the control input
    u = u - Jgrad*ss[j] # update the control input
    print(f'cost: {J:.2f}', end='\r')
u = best_u
print(f'iteration {i+1}/{iterations}, cost: {best_J:.2f}')

# SIMULATION 
################################################################################################

# Simulate the pendulum
# u = np.zeros(int(simT/dt)) # control input
x,t = simulate(x0, dx0, simT, dt, u) # simulate the pendulum

J = cost(x, u) # calculate the cost
print(f'cost: {J:.2f}')
print(f'u: {u}')

# calculate the energies
T = fT(*x.T) # kinetic energy
V = fV(*x.T) # potential energy 
################################################################################################

# plot the state and energies
u = expand_input(u, len(t))
fig, ax = plt.subplots(4, 1, figsize=(12,10)) #figsize=(18,12))
ax[0].plot(t, x[:,0], label='θ, angle', color=C)
ax[0].set_ylabel('Angle [rad]')
ax[0].grid(True)
ax[1].plot(t, x[:,1], label='dθ, angular velocity', color=C)
ax[1].set_ylabel('Angular velocity [rad/s]')
ax[1].grid(True)
ax[2].plot(t, T, label='T, kinetic energy', color='red')
ax[2].plot(t, V, label='V, potential energy', color='blue')
ax[2].set_ylabel('Energy [J]')
# ax[2].set_yscale('log')
ax[2].legend()
ax[2].plot(t, T+V, '--',label='T+V, total energy', color='black')
ax[2].legend(), ax[2].grid(True)
ax[3].plot(t, u, label='u, control input', color=C)
ax[3].set_ylabel('Control input')
ax[3].grid(True)
plt.tight_layout()

animate_pendulum(x, u, dt, fps, l, figsize=(4,4))

cost(x,u)

plt.show()
