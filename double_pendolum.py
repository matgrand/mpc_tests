import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import sin, cos, tan, pi

OFFSET = +pi/2

###################################################################################################################
L1 = 1.1  # First arm
L2 = 1  # Second arm
g = 9.81  # gravity
mu1 = 0.0  # friction coefficient first joint
mu2 = 0.0  # friction coefficient second joint
m1 = 1.0  # mass of the first pendulum
m2 = 1.0  # mass of the second pendulum

dt = 0.01
T = 30
###################################################################################################################
## CONTROLLER
# BOTH, ONLY1, ONLY2, FREE = True, False, False, False # both
# BOTH, ONLY1, ONLY2, FREE = False, True, False, False # only 1
# BOTH, ONLY1, ONLY2, FREE = False, False, True, False # only 2
BOTH, ONLY1, ONLY2, FREE = False, False, False, True # free

assert sum([BOTH, ONLY1, ONLY2, FREE]) == 1, "Only one controller can be active"
print(f'{"BOTH" if BOTH else "ONLY1" if ONLY1 else "ONLY2" if ONLY2 else "FREE"} controller active')
# SETPOINT
θstar1 = pi
θstar2 = pi
# start position deviation
std1 = 5 # [deg] deviation from the setpoint
std2 = 5 # [deg] deviation from the setpoint
###################################################################################################################

def model_step(θ1, θ2, ω1, ω2, τ1, τ2, dt):
    #remove offset
    θ1, θ2 = θ1 - OFFSET, θ2 - OFFSET
    # Compute the derivatives of θ1, θ2, ω1, and ω2
    dθ1_dt = ω1
    dθ2_dt = ω2
    dω1_dt = (-g*(2*m1+m2)*sin(θ1) - m2*g*sin(θ1-2*θ2) - 2*sin(θ1-θ2)*m2*(ω2**2*L2+ω1**2*L1*cos(θ1-θ2))) / (L1*(2*m1+m2-m2*cos(2*θ1-2*θ2))) - mu1*ω1 + τ1
    dω2_dt = (2*sin(θ1-θ2)*(ω1**2*L1*(m1+m2) + g*(m1+m2)*cos(θ1) + ω2**2*L2*m2*cos(θ1-θ2))) / (L2*(2*m1+m2-m2*cos(2*θ1-2*θ2))) - mu2*ω2 + τ2
    # Update θ1, θ2, ω1, and ω2 using Euler's method
    θ1 = θ1 + dt * dθ1_dt
    θ2 = θ2 + dt * dθ2_dt
    ω1 = ω1 + dt * dω1_dt
    ω2 = ω2 + dt * dω2_dt
    #return with offset
    return θ1 + OFFSET, θ2 + OFFSET, ω1, ω2
    return θ1, θ2, ω1, ω2

class PIDController:
    def __init__(self, Kp1=50, Ki1=10, Kd1=10
                 , Kp2=50, Ki2=10, Kd2=10):
        self.Kp1, self.Ki1, self.Kd1 = Kp1, Ki1, Kd1
        self.Kp2, self.Ki2, self.Kd2 = Kp2, Ki2, Kd2
        self.error_sum1, self.prev_error1 = 0, 0
        self.error_sum2, self.prev_error2 = 0, 0

    def compute_control(self, e1, e2, dt):
        # Proportional term
        P1 = self.Kp1 * e1
        P2 = self.Kp2 * e2
        # Integral term
        self.error_sum1 += e1 * dt
        self.error_sum2 += e2 * dt
        I1 = self.Ki1 * self.error_sum1
        I2 = self.Ki2 * self.error_sum2
        # Derivative term
        D1 = self.Kd1 * (e1 - self.prev_error1) / dt
        D2 = self.Kd2 * (e2 - self.prev_error2) / dt
        self.prev_error1 = e1
        self.prev_error2 = e2
        # Compute the control signal
        c1 = P1 + I1 + D1 # control signal for the first joint
        c2 = P2 + I2 + D2 # control signal for the second joint
        return c1, c2
    
class ONLY1Controller:
    def __init__(self, Kp1=20, Ki1=30, Kd1=20
                 , Kp2=-50, Ki2=0, Kd2=0):
        self.Kp1, self.Ki1, self.Kd1 = Kp1, Ki1, Kd1
        self.Kp2, self.Ki2, self.Kd2 = Kp2, Ki2, Kd2
        self.error_sum1, self.prev_error1 = 0, 0
        self.error_sum2, self.prev_error2 = 0, 0

    def compute_control(self, e1, e2, dt):
        # Proportional term
        P1 = self.Kp1 * e1
        P2 = self.Kp2 * e2
        # Integral term
        self.error_sum1 += e1 * dt
        self.error_sum2 += e2 * dt
        I1 = self.Ki1 * self.error_sum1
        I2 = self.Ki2 * self.error_sum2
        # Derivative term
        D1 = self.Kd1 * (e1 - self.prev_error1) / dt
        D2 = self.Kd2 * (e2 - self.prev_error2) / dt
        self.prev_error1 = e1
        self.prev_error2 = e2
        # Compute the control signal
        c1 = P1 + I1 + D1 + P2 + I2 + D2 # control signal for the first joint
        return c1, 0

def simulate_double_pendulum(θ1_0, θ2_0, ω1_0, ω2_0, dt, T):
    #initialize the controller
    if BOTH: controller = PIDController()
    elif ONLY1: controller = ONLY1Controller() # TODO fix this
    elif ONLY2: controller = ONLY1Controller() # TODO
    elif FREE: controller = PIDController(0,0,0,0,0,0)
    # Initialize arrays to store time, angles, and angular velocities
    t = np.arange(0, T, dt)
    θ1, θ2, ω1, ω2 = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t) #states
    τ1, τ2 = np.zeros_like(t), np.zeros_like(t)  # control signals
    # Set initial conditions
    θ1[0], θ2[0], ω1[0], ω2[0] = θ1_0, θ2_0, ω1_0, ω2_0
    # Simulate the dynamics of the double pendulum
    for i in range(1, len(t)):
        e1 = θstar1 - θ1[i-1] + OFFSET
        e2 = (θstar2 - θ2[i-1]) - e1 + OFFSET
        τ1[i], τ2[i] = controller.compute_control(e1, e2, dt)
        # model step
        θ1[i], θ2[i], ω1[i], ω2[i] = model_step(θ1[i-1], θ2[i-1], ω1[i-1], ω2[i-1], τ1[i], τ2[i], dt)
    return t, θ1, θ2, ω1, ω2, τ1, τ2


def create_animation(t, θ1, θ2, L1, L2):
    # Animate the double pendulum with trace
    fig = plt.figure(figsize=(15, 15))
    fl = (L1+L2)*1.1
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-fl, fl), ylim=(-fl, fl))
    ax.set_aspect('equal')
    ax.grid()

    line1, = ax.plot([], [], 'o-', lw=2, color='blue')
    line2, = ax.plot([], [], 'o-', lw=2, color='red')
    trace, = ax.plot([], [], '-', lw=0.5, color='purple')  # Added trace
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        trace.set_data([], [])  # Initialize trace
        time_text.set_text('')
        return line1, line2, trace, time_text

    def animate(i):
        x1, x2 = [0, L1*sin(θ1[i])],[L1*sin(θ1[i]), L1*sin(θ1[i]) + L2*sin(θ2[i])]
        y1, y2 = [0, -L1*cos(θ1[i])], [-L1*cos(θ1[i]), -L1*cos(θ1[i]) - L2*cos(θ2[i])]
        line1.set_data(x1, y1), line2.set_data(x2, y2)
        trace.set_data(np.append(trace.get_xdata(), x2[1]), np.append(trace.get_ydata(), y2[1]))
        time_text.set_text(time_template % (i*dt))
        return line1, line2, trace, time_text
    
    ani = animation.FuncAnimation(fig, animate, range(1, len(t)),
                                    interval=dt*1000, blit=True, init_func=init)
    plt.show()


if __name__ == '__main__':

    # Set the initial conditions
    θ1_0 = θstar1 + np.random.normal(0, np.deg2rad(std1)) + OFFSET
    θ2_0 = θstar2 + np.random.normal(0, np.deg2rad(std2)) + OFFSET 
    ω1_0 = 0.0 
    ω2_0 = 0.0

    # Simulate the dynamics of the double pendulum
    t, θ1, θ2, ω1, ω2, τ1, τ2 = simulate_double_pendulum(θ1_0, θ2_0, ω1_0, ω2_0, dt, T)

    # Plot the angles, angular velocities, and input torques in a single figure with 3 plots, one below the other
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    θ1 = (θ1 + 2 * pi) % (2 * pi) -OFFSET # shift angles to be in range [0, 2π]
    θ2 = (θ2 + 2 * pi) % (2 * pi) -OFFSET # shift angles to be in range [0, 2π]
    # Convert to degrees
    θ1deg, θ2deg = np.rad2deg(θ1), np.rad2deg(θ2)
    ax1.plot(t, θ1deg, label='θ1')
    ax1.plot(t, θ2deg, label='θ2')
    ax1.set_ylabel('angle (deg)')
    ax1.grid()
    ax2.plot(t, ω1, label='ω1')
    ax2.plot(t, ω2, label='ω2')
    ax2.set_ylabel('angular velocity (rad/s)')
    ax2.grid()
    ax3.plot(t, τ1, label='τ1')
    ax3.plot(t, τ2, label='τ2')
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('input torque')
    ax3.grid()
    ax1.legend(), ax2.legend(), ax3.legend()

    # Animate the double pendulum with trace
    create_animation(t, θ1, θ2, L1, L2)
