
import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from tqdm import tqdm
sp.init_printing()

###################################################################################################################
l1 = 1  # First arm
l2 = 1.1  # Second arm
g = 9.81  # gravity
μ1 = 0.4  # friction coefficient first joint
μ2 = 0.2  # friction coefficient second joint
m1 = 1.3  # mass of the first pendulum
m2 = .1  # mass of the second pendulum
mc = 1  # mass of the cart

dt = 0.0001  # time step
SIMT = 30 # simulation time
fps = 60 # frames per second

# use lagrangian mechanics to derive the equations of motion
# define the symbolic variables
t = sp.symbols('t')
θ1, θ2, xc = sp.symbols('θ1 θ2 x_c', cls=sp.Function)
#define as functions of time
θ1, θ2 = θ1(t), θ2(t) # angles of the joints
ω1, ω2 = θ1.diff(t), θ2.diff(t) # angular velocities of the joints
α1, α2 = ω1.diff(t), ω2.diff(t) # angular accelerations of the joints
xc = xc(t) # position of the cart
v = xc.diff(t) # velocity of the cart
a = v.diff(t) # acceleration of the cart

#define position of all the masses
x1, y1 = xc + l1*sp.sin(θ1), -l1*sp.cos(θ1) # position of the first pendulum
x2, y2 = xc + l1*sp.sin(θ1) + l2*sp.sin(θ2), -l1*sp.cos(θ1) - l2*sp.cos(θ2) # position of the second pendulum

# define the kinetic energy of the system
T1 = 1/2*m1*(x1.diff(t)**2 + y1.diff(t)**2) # kinetic energy of the first pendulum
T2 = 1/2*m2*(x2.diff(t)**2 + y2.diff(t)**2) # kinetic energy of the second pendulum
Tc = 1/2*mc*xc.diff(t)**2 # kinetic energy of the cart
T = T1 + T2 + Tc # total kinetic energy

# define the potential energy of the system
V = m1*g*y1 + m2*g*y2 # total potential energy

# define the lagrangian
L = T - V

# get the lagrange equations
LEQθ1 = (L.diff(θ1) - (L.diff(ω1)).diff(t)).simplify()
LEQθ2 = (L.diff(θ2) - (L.diff(ω2)).diff(t)).simplify()
LEQxc = (L.diff(xc) - (L.diff(v)).diff(t)).simplify()
print('Lagrange equations derived')

# solve the lagrange equations for the accelerations
sol = sp.solve([LEQθ1, LEQθ2, LEQxc], [α1, α2, a])
sol_α1 = sol[α1].simplify()
sol_α2 = sol[α2].simplify()
sol_a = sol[a].simplify()
print('Lagrange equations solved for the accelerations')

# add the friction terms
sol_α1 = sol_α1 - μ1*ω1
sol_α2 = sol_α2 - μ2*ω2

# lambdify the equations of motion
model_α1 = sp.lambdify((θ1, θ2, ω1, ω2, xc, v), sol_α1, 'numpy')
model_α2 = sp.lambdify((θ1, θ2, ω1, ω2, xc, v), sol_α2, 'numpy')
model_a = sp.lambdify((θ1, θ2, ω1, ω2, xc, v), sol_a, 'numpy')


# Integrate the differential equation
nt = np.arange(0, SIMT, dt)
nθ1, nω1, nα1, nθ2, nω2, nα2, nxc, nv, na = [np.zeros(len(nt)) for _ in range(9)]

nθ1[0] = π+0.1 # initial angle of the first pendulum
nθ2[0] = π-0.1 # initial angle of the second pendulum

for i in tqdm(range(1, len(nt))):
    nα1[i] = model_α1(nθ1[i-1], nθ2[i-1], nω1[i-1], nω2[i-1], nxc[i-1], nv[i-1])
    nα2[i] = model_α2(nθ1[i-1], nθ2[i-1], nω1[i-1], nω2[i-1], nxc[i-1], nv[i-1])
    na[i] = model_a(nθ1[i-1], nθ2[i-1], nω1[i-1], nω2[i-1], nxc[i-1], nv[i-1])
    nω1[i] = nω1[i-1] + nα1[i]*dt
    nω2[i] = nω2[i-1] + nα2[i]*dt
    nv[i] = nv[i-1] + na[i]*dt
    nθ1[i] = nθ1[i-1] + nω1[i]*dt
    nθ2[i] = nθ2[i-1] + nω2[i]*dt
    nxc[i] = nxc[i-1] + nv[i]*dt
print('Differential equation integrated')

#plot
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(nt, nθ1, label='θ1')
ax[0].plot(nt, nθ2, label='θ2')
ax[0].set_xlabel('time [s]'), ax[0].set_ylabel('angle [rad]')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(nt, nxc, label='x')
ax[1].set_xlabel('time [s]'), ax[1].set_ylabel('position [m]')
ax[1].legend()
ax[1].grid(True)

plt.show()

# Animate the double pendulum
n = int(1/fps/dt) # display one frame every n time steps
nt, nθ1, nω1, nα1, nθ2, nω2, nα2, nxc, nv, na = nt[::n], nθ1[::n], nω1[::n], nα1[::n], nθ2[::n], nω2[::n], nα2[::n], nxc[::n], nv[::n], na[::n]

fig1, ax = plt.subplots(figsize=(10, 10))
lim = (l1+l2)*1.2
ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
ax.set_title('Double pendulum on a cart')

cart, = ax.plot([], [], 'o-', lw=2, color='white')
line2, = ax.plot([], [], 'o-', lw=2, color='red')
line1, = ax.plot([], [], 'o-', lw=2, color='blue')

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    cart.set_data([], [])
    time_text.set_text('')
    return line1, line2, cart, time_text

def animate(i):
    x1 = nxc[i] + l1*np.sin(nθ1[i])
    y1 = -l1*np.cos(nθ1[i])
    x2 = x1 + l2*np.sin(nθ2[i])
    y2 = y1 - l2*np.cos(nθ2[i])
    cart.set_data([nxc[i]-0.1, nxc[i]+0.1], [0, 0])
    line2.set_data([x1, x2], [y1, y2])
    line1.set_data([nxc[i], x1], [0, y1])
    time_text.set_text(time_template % (i/fps))
    return line1, line2, cart, time_text

ani = animation.FuncAnimation(fig1, animate, np.arange(1, len(nt)), init_func=init,
                                interval=1/fps*600, blit=True)
plt.show()