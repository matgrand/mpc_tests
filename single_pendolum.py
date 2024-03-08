import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
sp.init_printing()

# Constants
g = 9.81 # [m/s^2] gravity
l = 1 # [m] length of the pendulum
m = 1 # [kg] mass of the pendulum
μ = 0.5 # [kg/s] damping coefficient

# Initial conditions
θ0 = np.pi/2
ω0 = 0
# Time
dt = 0.0001
simT = 30 # [s] simulation time
fps = 60 # [Hz] frames per second

# Variables
t = sp.symbols('t')
θ = sp.symbols('θ', cls=sp.Function)(t)
ω = θ.diff(t) # angular velocity 
α = ω.diff(t) # angular acceleration

# let's use the Lagrangian method, no friction
x, y = l*sp.sin(θ), -l*sp.cos(θ) # position of the pendulum
T = (1/2)*m*(x.diff(t)**2 + y.diff(t)**2) # Kinetic energy
V = m*g*y # Potential energy
L = T - V # Lagrangian

# Euler-Lagrange equation
leq = ((L.diff(ω)).diff(t) - L.diff(θ)).simplify()

# Solve the differential equation and add the damping term
α = sp.solve(leq, α)[0] - μ*ω

model = sp.lambdify((t, θ, ω), α, 'numpy')

# Integrate the differential equation
nt = np.arange(0, simT, dt)
nθ, nω, nα = np.zeros(len(nt)), np.zeros(len(nt)), np.zeros(len(nt))
nθ[0], nω[0] = θ0, ω0

for i in range(1, len(nt)):
    nα[i] = model(0, nθ[i-1], nω[i-1])
    nω[i] = nω[i-1] + nα[i]*dt
    nθ[i] = nθ[i-1] + nω[i]*dt

# # Plot
# fig, ax = plt.subplots()
# ax.plot(nt, nθ)
# ax.plot(nt, nω)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Angle [rad]')
# ax.set_title('Pendulum')
    
# subsample everything to match the fps
n = int(1/(fps*dt))
nt, nθ, nω, nα = nt[::n], nθ[::n], nω[::n], nα[::n]

# Animation
fig, ax = plt.subplots()
lim = l*1.5
ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.grid()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Pendulum')
line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    x, y = l*np.sin(nθ[i]), -l*np.cos(nθ[i])
    line.set_data([0, x], [0, y])
    time_text.set_text(time_template % (i/fps))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(nt)), init_func=init,
                                interval=1/fps*600, blit=True)
plt.show()





