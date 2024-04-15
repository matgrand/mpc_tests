import numpy as np; π = np.pi
import sympy as sp

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.4 # [kg/s] damping coefficient
WRAP_AROUND = True # wrap the angle to [-π, π]

if µ < 0: print('Warning: the damping coefficient is negative')

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
    if WRAP_AROUND: θ = (θ+π) % (2*π) - π # normalize the angle to [-π, π]
    return np.array([θ, dθ]) # new state vector

if __name__ == '__main__':

    # initial conditions
    x0 = np.array([0.1, 0]) # initial conditions
    t = np.linspace(0, 10, 10000) # time vector
    u = 0*np.sin(t) # control input
    x = np.zeros((len(t), 2)) # state vector
    x[0] = x0 # initial conditions

    # simulate the pendulum
    for i in range(1, len(t)): x[i] = step(x[i-1], u[i], t[1] - t[0])

    # plot the results
    from plotting import *

    ap1 = animate_pendulum(x, u, t[1]-t[0], l, 60, (6,6))
    plt.show()
