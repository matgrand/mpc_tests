import numpy as np
import sympy as sp
# from inputs import addittive_resample as expand_input
from inputs import frequency_resample as expand_input

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.8 # [kg/s] damping coefficient

INPUT_CLIP = 30 # maximum control input

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

def step(x, u, dt): 
    '''Integrate the differential equation using the Euler method'''
    θ, dθ = x # split the state vector
    dθ = dθ + fddθ(θ, dθ, u)[0]*dt # integrate the acceleration
    θ = θ + dθ*dt # integrate the velocity
    return np.array([θ, dθ]) # return the new state vector

#simulate a run
def simulate(x0, simT, dt, u, clip=True, continue_for=0):
    '''Simulate the pendulum'''
    n = int(simT/dt) # number of time steps
    t = np.linspace(0, simT, n) # time vector
    x = np.zeros((n, 2)) # [θ, dθ] -> state vector
    eu = expand_input(u, simT, n) # expand the control input
    if clip: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x[0] = x0 # initial conditions
    for i in range(1, n):
        x[i] = step(x[i-1], eu[i], dt)
    
    if continue_for > 0:
        #extend the simulation with u=0
        ne = int(continue_for/dt)
        te = np.linspace(0, continue_for, ne)
        x = np.vstack([x, np.zeros((ne, 2))])
        for i in range(n, n+ne):
            x[i] = step(x[i-1], 0, dt)
        t = np.hstack([t, te])
        eu = np.hstack([eu, np.zeros(ne)])
        

    return x, t, eu


