
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp

#enable latex rendering
sp.init_printing()

###################################################################################################################
l1 = 1.1  # First arm
l2 = 1  # Second arm
g = 9.81  # gravity
μ1 = 0.8  # friction coefficient first joint
μ2 = 0.8  # friction coefficient second joint
m1 = 1  # mass of the first pendulum
m2 = 1.1  # mass of the second pendulum

dt = 0.01  # time step
SIMT = 30
fps = 60

# use lagrangian mechanics to derive the equations of motion
# define the symbolic variables
t = sp.symbols('t')
θ1, θ2 = sp.symbols('θ1 θ2', cls=sp.Function)
#define as functions of time
θ1, θ2 = θ1(t), θ2(t) # angles of the joints
ω1, ω2 = θ1.diff(t), θ2.diff(t) # angular velocities of the joints
α1, α2 = ω1.diff(t), ω2.diff(t) # angular accelerations of the joints

#define position of all the masses
x1, y1 = l1*sp.sin(θ1), -l1*sp.cos(θ1) # position of the first pendulum
x2, y2 = x1 + l2*sp.sin(θ2), y1 - l2*sp.cos(θ2) # position of the second pendulum

# define the kinetic energy of the system
T1 = 1/2*m1*(x1.diff(t)**2 + y1.diff(t)**2) # kinetic energy of the first pendulum
T2 = 1/2*m2*(x2.diff(t)**2 + y2.diff(t)**2) # kinetic energy of the second pendulum
T = T1 + T2 # total kinetic energy

# define the potential energy of the system
V = m1*g*y1 + m2*g*y2 # total potential energy

# lagrangian
L = T - V

# get the lagrange equations
# LEQθ1 = ((L.diff(ω1)).diff(t) - L.diff(θ1)).simplify()
# LEQθ2 = ((L.diff(ω2)).diff(t) - L.diff(θ2)).simplify()
LEQθ1 = sp.diff(L, θ1) - sp.diff(sp.diff(L, ω1), t)
LEQθ2 = sp.diff(L, θ2) - sp.diff(sp.diff(L, ω2), t)

# solve the lagrange equations for the accelerations
sol = sp.solve([LEQθ1, LEQθ2], [α1, α2])
sol_α1 = sol[α1].simplify()
sol_α2 = sol[α2].simplify()

# lambdify the equations of motion
model1 = sp.lambdify((θ1, θ2, ω1, ω2), sol_α1)
model2 = sp.lambdify((θ1, θ2, ω1, ω2), sol_α2)

# Integrate the differential equations  
tv = np.arange(0, SIMT, dt) # time vector
θ1v, θ2v, ω1v, ω2v, α1v, α2v = [np.zeros(len(tv)) for _ in range(6)] # initialize the variables
θ1v[0], θ2v[0] = np.pi/2, np.pi/2 # initial conditions
ω1v[0], ω2v[0] = 0, 0 # initial conditions

for i in range(1, len(tv)):
    α1v[i] = model1(θ1v[i-1], θ2v[i-1], ω1v[i-1], ω2v[i-1])
    α2v[i] = model2(θ1v[i-1], θ2v[i-1], ω1v[i-1], ω2v[i-1])
    ω1v[i] = ω1v[i-1] + α1v[i]*dt
    ω2v[i] = ω2v[i-1] + α2v[i]*dt
    θ1v[i] = θ1v[i-1] + ω1v[i]*dt
    θ2v[i] = θ2v[i-1] + ω2v[i]*dt


# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(tv, θ1v, label='θ1')
ax[0].plot(tv, θ2v, label='θ2')
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Angle [rad]')
ax[0].set_title('Angles')
ax[0].legend()
ax[0].grid()

ax[1].plot(tv, ω1v, label='ω1')
ax[1].plot(tv, ω2v, label='ω2')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Angular velocity [rad/s]')
ax[1].set_title('Angular velocities')
ax[1].legend()
ax[1].grid()

# Animation
n = int(1/fps/dt)
tv, θ1v, θ2v, ω1v, ω2v, α1v, α2v = tv[::n], θ1v[::n], θ2v[::n], ω1v[::n], ω2v[::n], α1v[::n], α2v[::n]

fig, ax = plt.subplots()
lim = (l1+l2)*1.2
ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Pendulum')
line1, = ax.plot([], [], 'o-', lw=2)
line2, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text

def animate(i):
    x1, y1 = l1*np.sin(θ1v[i]), -l1*np.cos(θ1v[i])
    x2, y2 = x1 + l2*np.sin(θ2v[i]), y1 - l2*np.cos(θ2v[i])
    line1.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    time_text.set_text(time_template % (i*1/fps))
    return line1, line2, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(tv)), init_func=init,
                                interval=1/fps*1000, blit=True)
plt.show()






