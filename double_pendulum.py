
import numpy as np; π = np.pi
import sympy as sp
from plotting import animate_double_pendulum, plot_double
import matplotlib.pyplot as plt

###################################################################################################################
l1 = 1  # First arm
l2 = 1  # Second arm
g = 9.81  # gravity
μ1 = 0 #.7  # friction coefficient first joint
μ2 = 0 #.7  # friction coefficient second joint
m1 = 1  # mass of the first pendulum
m2 = 1  # mass of the second pendulum

# use lagrangian mechanics to derive the equations of motion
# define the symbolic variables
t = sp.symbols('t')
θ1, θ2, u = sp.symbols('θ1 θ2 u', cls=sp.Function)
#define as functions of time
θ1, θ2 = θ1(t), θ2(t) # angles of the joints
u = u(t) # control input
dθ1, dθ2 = θ1.diff(t), θ2.diff(t) # angular velocities of the joints
ddθ1, ddθ2 = dθ1.diff(t), dθ2.diff(t) # angular accelerations of the joints

#define position of all the masses
x1, y1 = l1*sp.sin(θ1), l1*sp.cos(θ1) # position of the first pendulum
x2, y2 = x1 + l2*sp.sin(θ2), y1 + l2*sp.cos(θ2) # position of the second pendulum
dx1, dy1 = x1.diff(t), y1.diff(t) # velocity of the first pendulum
dx2, dy2 = x2.diff(t), y2.diff(t) # velocity of the second pendulum

# define the kinetic energy of the system
T1 = 1/2*m1*(dx1**2 + dy1**2) # kinetic energy of the first pendulum
T2 = 1/2*m2*(dx2**2 + dy2**2) # kinetic energy of the second pendulum
T = T1 + T2 # total kinetic energy

# define the potential energy of the system
V1 = m1*g*y1 # potential energy of the first pendulum
V2 = m2*g*y2 # potential energy of the second pendulum
V = V1 + V2 # total potential energy

# lagrangian
L = T - V

# get the lagrange equations
LEQθ1 = L.diff(θ1) - (L.diff(dθ1)).diff(t) - μ1*dθ1 + u # lagrange equation for the first joint
LEQθ2 = L.diff(θ2) - (L.diff(dθ2)).diff(t) - μ2*dθ2 # lagrange equation for the second join

# lambdify the equations of motion
sol = sp.solve([LEQθ1, LEQθ2], [ddθ1, ddθ2], simplify=False)
ddθ1 = sol[ddθ1].simplify()
ddθ2 = sol[ddθ2].simplify()

f = sp.lambdify((θ1, θ2, dθ1, dθ2, u), [ddθ1, ddθ2], 'numpy')

# kinetic and potential energy
fT = sp.lambdify((θ1, θ2, dθ1, dθ2), T, 'numpy')
fV = sp.lambdify((θ1, θ2, dθ1, dθ2), V, 'numpy')

def kinetic_energy(x): return fT(*x.T)
def potential_energy(x): return fV(*x.T)

del t, θ1, θ2, u, dθ1, dθ2, x1, y1, x2, y2, T1, T2, T, V, L, LEQθ1, LEQθ2, sol, ddθ1, ddθ2 



###################################################################################################################

## Euler integration
# SATURATE_VELOCITIES = 1e3
def euler_step(x, u, dt):
    '''Integrate the differential equation using the Euler method'''
    x, dx = x[:2], x[2:] # split the state vector into position and velocity
    # dx = np.clip(dx, -SATURATE_VELOCITIES, SATURATE_VELOCITIES) # saturate the velocities
    dx = dx + np.array(f(*x, *dx, u))*dt # compute the new velocity
    x = x + dx*dt # compute the new position
    return np.concatenate([x, dx]) # return the new state vector

## Symplectic Euler integration
def symplectic_step(x, u, dt): # NOTE: THIS IS COMPLETELY WRONG
    raise NotImplementedError('This is not the correct symplectic Euler method')
    '''Integrate the differential equation using the symplectic Euler method'''
    ke = kinetic_energy(x)
    x, dx = x[:2], x[2:] # split the state vector into position and velocity
    dx = dx + np.array(f(*x, *dx, u))*dt # compute the new velocity
    x = x + dx*dt # compute the new position
    nx = np.concatenate([x, dx]) # new state vector
    nke = kinetic_energy(nx) # new kinetic energy
    v0, v1 = nx[2], nx[3] # new velocities
    # correct the velocities
    dke = nke - ke # change in kinetic energy
    v0, v1 = v0*np.sqrt(1+dke/nke), v1*np.sqrt(1+dke/nke) # correct the velocities
    return np.array([x[0], x[1], v0, v1]) # return the new state vector


def step(x, u, dt): return euler_step(x, u, dt)