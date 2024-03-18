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

INPUT_CLIP = 50 # maximum control input

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
def kinetic_energy(x): return fT(*x.T)
def potential_energy(x): return fV(*x.T)

del t, θ, dθ, leq, T, V, L, x, y, u # delete the symbolic variables

def expand_input(iu, simT, dt, nto): 
    '''
    Expand the control input to match the state vector
    iu: input, simT: simulation time, nto: number of time steps in the output
    '''
    liu = len(iu) # length of the compressed input
    ou = np.zeros((int(simT/dt))) # expanded control input
    it = np.linspace(0, simT, liu) # input time
    ot = np.linspace(0, simT, nto) # expanded time
    ii = 0 # index for the compressed input
    for i in range(nto):
        ia, ib = it[ii], it[ii+1] # time interval for the compressed input
        a, b = iu[ii], iu[ii+1] # control input interval
        ou[i] = a + (ot[i] - ia)*(b - a)/(ib - ia) # linear interpolation
        if ot[i] > it[ii+1]: ii += 1 # update the index
    return ou

# def expand_input(iu, simT, dt, nto): # iu: input, nto: number of time steps in the output
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
    eu = expand_input(u, simT, dt, len(t))
    eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x[0] = [x0, dx0] # initial conditions
    for i in range(1, len(t)):
        x[i] = step(x[i-1], eu[i], dt)
    return x, t, eu
