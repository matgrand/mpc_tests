import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm
from plotting import animate_pendulum, C

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.7 # [kg/s] damping coefficient

# Initial conditions
x0 = np.pi/2
dx0 = 0
# Time
dt = 0.001
simT = 20 # [s] simulation time
fps = 60 # [Hz] frames per second

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

def step(x, u, dt): 
    '''Integrate the differential equation using the Euler method'''
    θ, dθ = x # split the state vector
    dθ = dθ + fddθ(θ, dθ, u)[0]*dt # integrate the acceleration
    θ = θ + dθ*dt # integrate the velocity
    return np.array([θ, dθ]) # return the new state vector

###############################
# MODEL PREDICTIVE CONTROL
###############################

# cost function
J = None






# numerical integration
t = np.arange(0, simT, dt)
x = np.zeros((len(t), 2)) # [θ, dθ] -> state vector
u = np.zeros(len(t)) # control input
x[0] = [x0, dx0] # initial conditions

for i in tqdm(range(1, len(t))):
    ui = 10*np.sin(2*np.pi*1.5*t[i]) # control input
    x[i] = step(x[i-1], ui, dt)
    u[i] = ui 

T = fT(*x.T) # kinetic energy
V = fV(*x.T) # potential energy





# plot the state and energies
fig, ax = plt.subplots(4, 1, figsize=(18,12))
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


animate_pendulum(x, u, dt, fps, l)

plt.show()
