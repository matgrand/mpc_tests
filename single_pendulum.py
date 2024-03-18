import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
np.random.seed(42)

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.8 # [kg/s] damping coefficient

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

# def expand_input(iu, t, ne): 
#     '''
#     input iu is an approxximation of the control input
#     Expand the control input to match the state vector
#     iu: compressed input, t: simulation time, ne: number expanded control inputs
#     '''
#     nc = len(iu) # number of compressed control inputs
#     ou = np.zeros((ne)) # expanded control input
#     ct = np.linspace(0, t, nc) # compressed time
#     et = np.linspace(0, t, ne) # expanded time
#     ii = 0 # index for the compressed input
#     for i in range(ne):
#         ia, ib = ct[ii], ct[ii+1] # time interval for the compressed input
#         a, b = iu[ii], iu[ii+1] # control input interval
#         ou[i] = a + (et[i] - ia)*(b - a)/(ib - ia) # linear interpolation
#         if et[i] > ct[ii+1]: ii += 1 # update the index
#     return ou

def expand_input(iu, t, ne): 
    '''
    input is defined as a sequence of additions to the first control input
    Expand the control input to match the state vector
    iu: input, t: simulation time, ne: number expanded control inputs
    '''
    nc = len(iu) # length of the compressed input
    ou = np.zeros((ne)) # expanded control input
    ct = np.linspace(0, t, nc) # input time
    et = np.linspace(0, t, ne) # expanded time
    ii = 0 # index for the compressed input
    cumulated = iu[0] # cumulated control input
    for i in range(ne):
        dtc = ct[ii+1] - ct[ii] # time interval for the compressed input
        dti = et[i] - ct[ii] # time interval for the expanded input
        ou[i] = cumulated + iu[ii+1]*dti/dtc # linear interpolation
        if et[i] > ct[ii+1]: 
            ii += 1 # update the index
            cumulated += iu[ii] # update the cumulated control input
    return ou

def step(x, u, dt): 
    '''Integrate the differential equation using the Euler method'''
    θ, dθ = x # split the state vector
    dθ = dθ + fddθ(θ, dθ, u)[0]*dt # integrate the acceleration
    θ = θ + dθ*dt # integrate the velocity
    return np.array([θ, dθ]) # return the new state vector

#simulate a run
def simulate(x0, simT, dt, u):
    '''Simulate the pendulum'''
    n = int(simT/dt) # number of time steps
    t = np.linspace(0, simT, n) # time vector
    x = np.zeros((n, 2)) # [θ, dθ] -> state vector
    eu = expand_input(u, simT, n) # expand the control input
    eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x[0] = x0 # initial conditions
    for i in range(1, n):
        x[i] = step(x[i-1], eu[i], dt)
    return x, t, eu
