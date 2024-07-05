import numpy as np 
import matplotlib.pyplot as plt
π = np.pi

# single pendulum
from eqns_single_pendulum import f_ddθ1, l1, g, μ1, m1

# create a simulation
N, dt = 1000, 0.01 # number of steps and time step
t = np.linspace(0, N*dt, N) # time vector
u = 0*np.sin(t) # control input
x = np.zeros((N, 2)) # state vector
x[0] = np.array([0.1, 0]) # initial conditions

# simulate the pendulum
for i in range(1, N): 
    x[i] = x[i-1] + np.array([x[i-1, 1], f_ddθ1(*x[i-1], u[i])])*dt

from plotting import animate_pendulum, animate_pendulums
anim = animate_pendulum(x, u, dt, l1, 60, (6,6))

# simulate multiple pendulums 
NP = 50 # number of pendulums
xs = np.zeros((N, NP, 2)) # state vector
us = np.zeros((N, NP)) # control input
xs[0,:] = 15*np.random.rand(NP, 2) # initial conditions
for i in range(1, N): 
    xs[i] = xs[i-1] + np.array([xs[i-1,:, 1], f_ddθ1(*xs[i-1].T, us[i])]).T*dt

# create a static plot with all the 10 pendulums, angles and velocities
fig, ax = plt.subplots(1, 2, figsize=(6,6))
for i in range(NP):
    ax[0].plot(t, xs[:,i,0], label=f'Pendulum {i+1}')
    ax[1].plot(t, xs[:,i,1], label=f'Pendulum {i+1}')
ax[0].set_title('Angle')
ax[1].set_title('Velocity')
ax[0].legend()
ax[1].legend()

anims = animate_pendulums(xs.transpose(1,0,2), us.T, dt, l1, 60, (6,6))

# double pendulum
from eqns_double_pendulum import f_ddθ1, f_ddθ2, l1, l2, m1, m2, g, μ1, μ2

# create a simulation
x = np.zeros((N, 4)) # state vector
x[0] = np.array([0.1, 0, 0, 0]) # initial conditions

def step(x, u, dt):
    θ1, θ2, dθ1, dθ2 = x.T
    ddθ1 = f_ddθ1(θ1, θ2, dθ1, dθ2, u)
    ddθ2 = f_ddθ2(θ1, θ2, dθ1, dθ2, u)
    dθ1 = dθ1 + ddθ1*dt
    dθ2 = dθ2 + ddθ2*dt
    θ1 = θ1 + dθ1*dt
    θ2 = θ2 + dθ2*dt
    out = np.array([θ1, θ2, dθ1, dθ2]).T
    return out
    
# simulate the pendulum
for i in range(1, N): 
    x[i] = step(x[i-1], u[i], dt)

from plotting import animate_double_pendulum, animate_double_pendulums
anim2 = animate_double_pendulum(x, u, dt, l1, l2, 60, (6,6))

# simulate multiple pendulums
xs = np.zeros((N, NP, 4)) # state vector
us = np.zeros((N, NP)) # control input
xs[0,:] = 15*np.random.rand(NP, 4) # initial conditions
for i in range(1, N): 
    xs[i] = step(xs[i-1], us[i], dt)

# create a static plot with all the 10 pendulums, angles and velocities
fig, ax = plt.subplots(1, 2, figsize=(6,6))
for i in range(NP):
    ax[0].plot(t, xs[:,i,0], label=f'Pendulum {i+1}')
    ax[1].plot(t, xs[:,i,2], label=f'Pendulum {i+1}')
ax[0].set_title('Angle')
ax[1].set_title('Velocity')

anims2 = animate_double_pendulums(xs.transpose(1,0,2), us.T, dt, l1, l2, 60, (6,6))

plt.show()



