
import numpy as np; π = np.pi
from eqns_double_pendulum import *
from plotting import *
###################################################################################################################

## Euler integration
# SATURATE_VELOCITIES = 1e3
def step(x, u, dt, wa=True):
    θ1, θ2, dθ1, dθ2 = x.T
    ddθ1 = f_ddθ1(θ1, θ2, dθ1, dθ2, u)
    ddθ2 = f_ddθ2(θ1, θ2, dθ1, dθ2, u)
    dθ1 = dθ1 + ddθ1*dt
    dθ2 = dθ2 + ddθ2*dt
    θ1 = θ1 + dθ1*dt
    θ2 = θ2 + dθ2*dt
    if wa: 
        θ1 = (θ1+π) % (2*π) - π
        θ2 = (θ2+π) % (2*π) - π
    out = np.array([θ1, θ2, dθ1, dθ2]).T
    return out

if __name__ == '__main__':
    # create a simulation
    N, dt = 10000, 0.001 # number of steps and time step
    t = np.linspace(0, N*dt, N) # time vector
    u = 0*np.sin(t) # control input
    x = np.zeros((N, 4)) # state vector
    x[0] = np.array([0.1, 0, 0, 0]) # initial conditions

    # simulate the pendulum
    for i in range(1, N): 
        x[i] = step(x[i-1], u[i], dt)

    # animate the results
    anim = animate_double_pendulum(x, u, dt, l1, l2, 60, (6,6))

    # simulate multiple pendulums
    NP = 10 # number of pendulums
    xs = np.zeros((N, NP, 4)) # state vector
    us = np.zeros((N, NP)) # control input
    xs[0,:] = 15*np.random.rand(NP, 4) # initial conditions
    for i in range(1, N): 
        xs[i] = step(xs[i-1], us[i], dt)

    anims = animate_double_pendulums(xs.transpose(1,0,2), us.T, dt, l1, l2, 60, (6,6))

    plt.show()