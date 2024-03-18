
import numpy as np
import sympy as sp
from plotting import animate_double_pendulum, plot_double
import matplotlib.pyplot as plt

###################################################################################################################
l1 = 1.1  # First arm
l2 = 1  # Second arm
g = 9.81  # gravity
μ1 = 0.8  # friction coefficient first joint
μ2 = 0.8  # friction coefficient second joint
m1 = 1  # mass of the first pendulum
m2 = 1.1  # mass of the second pendulum

INPUT_CLIP = 50  # maximum control input

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
V = m1*g*y1 + m2*g*y2 # total potential energy

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
    x, dx = x[:2], x[2:] # split the state vector into position and velocity
    dx = dx + np.array(f(*x, *dx, u))*dt # compute the new velocity
    x = x + dx*dt # compute the new position
    return np.concatenate([x, dx]) # return the new state vector

#simulate a run
def simulate(x0, simT, dt, u):
    '''Simulate the pendulum'''
    n = int(simT/dt) # number of time steps
    t = np.linspace(0, simT, n) # time vector
    x = np.zeros((n, 4)) # [θ1, θ2, dθ1, dθ2] -> state vector
    eu = expand_input(u, simT, n) # expand the control input
    eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x[0] = x0 # initial conditions
    for i in range(1, n):
        x[i] = step(x[i-1], eu[i], dt) # integrate the differential equation
    return x, t, eu



if __name__  == '__main__':
    x0 = np.array([np.pi/2, np.pi/2, 0, 0]) # initial conditions
    simT = 10 # simulation time
    dt = 1e-2 # time step
    u = np.zeros(int(simT/dt)) # control input
    # u[:int(simT/dt)//2] = 10 # control input
    # u[int(simT/dt)//2:] = -10 # control input

    # Simulate the pendulum
    x,t,eu = simulate(x0, simT, dt, u) # simulate the pendulum

    J = np.sum(kinetic_energy(x) + potential_energy(x)) # calculate the cost
    print(f'cost: {J:.2f}')

    # calculate the energies
    T = kinetic_energy(x) # kinetic energy
    V = potential_energy(x) # potential energy

    # plot the state and energies
    plot_double(x, t, eu, T, V, figsize=(12,10))

    # animate the pendulum
    a1 = animate_double_pendulum(x, eu, dt, l1, l2, figsize=(4,4))

    plt.show()
