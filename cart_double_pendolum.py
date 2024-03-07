
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from sympy import Rational as frac

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
mc = 1.2  # mass of the cart

dt = 0.01  # time step
SIMT = 30

# use lagrangian mechanics to derive the equations of motion
# define the symbolic variables
t = sp.symbols('t')
θ1, θ2, xc = sp.symbols(r'\theta_1 \theta_2 x_c', cls=sp.Function)
#define as functions of time
θ1, θ2 = θ1(t), θ2(t) # angles of the joints
ω1, ω2 = θ1.diff(t), θ2.diff(t) # angular velocities of the joints
α1, α2 = ω1.diff(t), ω2.diff(t) # angular accelerations of the joints
xc = xc(t) # position of the cart
v = xc.diff(t) # velocity of the cart
a = v.diff(t) # acceleration of the cart

#define position of all the masses
x1, y1 = xc + l1*sp.sin(θ1), -l1*sp.cos(θ1) # position of the first pendulum
x2, y2 = x1 + l2*sp.sin(θ2), y1 - l2*sp.cos(θ2) # position of the second pendulum

# define the kinetic energy of the system
T1 = frac(1,2)*m1*(x1.diff(t)**2 + y1.diff(t)**2) # kinetic energy of the first pendulum
T2 = frac(1,2)*m2*(x2.diff(t)**2 + y2.diff(t)**2) # kinetic energy of the second pendulum
Tc = frac(1,2)*mc*v**2 # kinetic energy of the cart
T = T1 + T2 + Tc # total kinetic energy

# define the potential energy of the system
V = m1*g*y1 + m2*g*y2 # total potential energy

# define the lagrangian
L = T - V

# get the lagrange equations
LEQθ1 = ((L.diff(ω1)).diff(t) - L.diff(θ1)).simplify()
LEQθ2 = ((L.diff(ω2)).diff(t) - L.diff(θ2)).simplify()
LEQxc = ((L.diff(v)).diff(t) - L.diff(xc)).simplify()
print('Lagrange equations derived')

# solve the lagrange equations
α1 = sp.solve(LEQθ1, α1)[0] - μ1*ω1
α2 = sp.solve(LEQθ2, α2)[0] - μ2*ω2
a = sp.solve(LEQxc, a)[0]
print('Lagrange equations solved')

# lambdify the equations
model = sp.lambdify((t, θ1, θ2, ω1, ω2, xc, v), (α1, α2, a), 'numpy')
print('Model lambdified')

# Integrate the differential equation
nt = np.arange(0, SIMT, dt)
nθ1, nθ2, nω1, nω2, nxc, nv = np.zeros(len(nt)), np.zeros(len(nt)), np.zeros(len(nt)), np.zeros(len(nt)), np.zeros(len(nt)), np.zeros(len(nt))
nθ1[0], nθ2[0], nω1[0], nω2[0], nxc[0], nv[0] = np.pi, np.pi, 0, 0, 0, 0

for i in range(1, len(nt)):
    nα1, nα2, na = model(0, nθ1[i-1], nθ2[i-1], nω1[i-1], nω2[i-1], nxc[i-1], nv[i-1])
    nω1[i] = nω1[i-1] + nα1*dt
    nω2[i] = nω2[i-1] + nα2*dt
    nθ1[i] = nθ1[i-1] + nω1[i]*dt
    nθ2[i] = nθ2[i-1] + nω2[i]*dt
    nv[i] = nv[i-1] + na*dt
    nxc[i] = nxc[i-1] + nv[i]*dt
print('Differential equation integrated')

# Plot
fig, ax = plt.subplots()
lim = (l1 + l2)*1.5
ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
print('Plot created')

# subsample everything to match the fps
fps = 30
n = int(1/(fps*dt))
nt, nθ1, nθ2, nω1, nω2, nxc, nv = nt[::n], nθ1[::n], nθ2[::n], nω1[::n], nω2[::n], nxc[::n], nv[::n]
print('Data subsampled')

# Animation
line1, = ax.plot([], [], 'o-', lw=3, color='blue')
line2, = ax.plot([], [], 'o-', lw=3, color='red')
trace, = ax.plot([], [], lw=1, color='purple')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    x1, x2 = [0, l1*np.sin(nθ1[0])],[l1*np.sin(nθ1[0]), l1*np.sin(nθ1[0]) + l2*np.sin(nθ2[0])]
    y1, y2 = [0, -l1*np.cos(nθ1[0])], [-l1*np.cos(nθ1[0]), -l1*np.cos(nθ1[0]) - l2*np.cos(nθ2[0])]
    line1.set_data(x1, y1), line2.set_data(x2, y2)
    time_text.set_text('')
    return line1, line2, trace, time_text

def animate(i):
    x1, x2 = [0, l1*np.sin(nθ1[i])],[l1*np.sin(nθ1[i]), l1*np.sin(nθ1[i]) + l2*np.sin(nθ2[i])]
    y1, y2 = [0, -l1*np.cos(nθ1[i])], [-l1*np.cos(nθ1[i]), -l1*np.cos(nθ1[i]) - l2*np.cos(nθ2[i])]
    line1.set_data(x1, y1), line2.set_data(x2, y2)
    tb, te = max(i-100, 0), i
    trace.set_data([l1*np.sin(nθ1[tb:te]) + l2*np.sin(nθ2[tb:te]), -l1*np.cos(nθ1[tb:te]) - l2*np.cos(nθ2[tb:te])])
    time_text.set_text(time_template % (i*1/fps))
    return line1, line2, trace, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(nt)), init_func=init,
                                interval=1/fps*100, blit=True)
plt.show()
