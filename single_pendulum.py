import numpy as np; π = np.pi
from eqns_single_pendulum import *

def step(x, u, dt, wa=True): 
    '''Integrate the differential equation using the Euler method
    x: state vector (# pendulums, 2)
    u: control input (# pendulums)
    dt: time step
    wa: wrap around the angle to [-π, π]'''
    θ, dθ = x.T # split the state vector
    ddθ = f_ddθ1(θ, dθ, u) # acceleration
    dθ = dθ + ddθ*dt # integrate the velocity
    θ = θ + dθ*dt # integrate the angle
    if wa: θ = (θ+π) % (2*π) - π # normalize the angle to [-π, π]
    return np.array([θ, dθ]).T # new state vector

if __name__ == '__main__':

    # initial conditions
    x0 = np.array([0.1, 0]) # initial conditions
    dt = 0.0001 # time step
    t = np.linspace(0, 10, int(10/dt)) # time vector
    u = 0*np.sin(t) # control input
    x = np.zeros((len(t), 2)) # state vector
    x[0] = x0 # initial conditions

    # simulate the pendulum
    for i in range(1, len(t)): x[i] = step(x[i-1], u[i], dt)

    # plot the results
    from plotting import *

    ap1 = animate_pendulum(x, u, dt, l1, 60, (6,6))

    # simulate multiple pendulums
    NP = 10 # number of pendulums
    xs = np.zeros((len(t), NP, 2)) # state vector
    us = np.zeros((len(t), NP)) # control input
    xs[0,:] = 15*np.random.rand(NP, 2) # initial conditions
    for i in range(1, len(t)): xs[i] = step(xs[i-1], us[i], dt)

    anims = animate_pendulums(xs.transpose(1,0,2), us.T, dt, l1, 60, (6,6))

    plt.show()
