import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 1.0  # Length of the pendulum
g = 9.81  # gravity
mu = 0.8  # friction coefficient

def simulate_pendulum(theta0, omega0, dt, T):
    # Initialize arrays to store time, angle, and angular velocity
    t = np.arange(0, T, dt)
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)

    # Set initial conditions
    theta[0] = theta0
    omega[0] = omega0

    # Simulate the dynamics of the pendulum
    for i in range(1, len(t)):
        # Compute the derivatives of theta and omega
        dtheta_dt = omega[i-1]
        domega_dt = -g/L * np.sin(theta[i-1]) - mu/L * omega[i-1]

        # Update theta and omega using Euler's method
        theta[i] = theta[i-1] + dt * dtheta_dt
        omega[i] = omega[i-1] + dt * domega_dt

    return t, theta, omega



###################################################################################################################
L1 = 1.0  # First arm
L2 = 1.0  # Second arm
g = 9.81  # gravity
mu1 = 0.8  # friction coefficient first joint
mu2 = 0.8  # friction coefficient second joint
m1 = 1.0  # mass of the first pendulum
m2 = 1.0  # mass of the second pendulum

def simulate_double_pendulum(theta1_0, theta2_0, omega1_0, omega2_0, dt, T):
    # Initialize arrays to store time, angles, and angular velocities
    t = np.arange(0, T, dt)
    theta1 = np.zeros_like(t)
    theta2 = np.zeros_like(t)
    omega1 = np.zeros_like(t)
    omega2 = np.zeros_like(t)

    # Set initial conditions
    theta1[0] = theta1_0
    theta2[0] = theta2_0
    omega1[0] = omega1_0
    omega2[0] = omega2_0

    # Simulate the dynamics of the double pendulum
    for i in range(1, len(t)):
        # Compute the derivatives of theta1, theta2, omega1, and omega2
        dtheta1_dt = omega1[i-1]
        dtheta2_dt = omega2[i-1]
        domega1_dt = (-g*(2*m1+m2)*np.sin(theta1[i-1]) - m2*g*np.sin(theta1[i-1]-2*theta2[i-1]) - 2*np.sin(theta1[i-1]-theta2[i-1])*m2*(omega2[i-1]**2*L2+omega1[i-1]**2*L1*np.cos(theta1[i-1]-theta2[i-1]))) / (L1*(2*m1+m2-m2*np.cos(2*theta1[i-1]-2*theta2[i-1])))
        domega2_dt = (2*np.sin(theta1[i-1]-theta2[i-1])*(omega1[i-1]**2*L1*(m1+m2) + g*(m1+m2)*np.cos(theta1[i-1]) + omega2[i-1]**2*L2*m2*np.cos(theta1[i-1]-theta2[i-1]))) / (L2*(2*m1+m2-m2*np.cos(2*theta1[i-1]-2*theta2[i-1])))

        # Update theta1, theta2, omega1, and omega2 using Euler's method
        theta1[i] = theta1[i-1] + dt * dtheta1_dt
        theta2[i] = theta2[i-1] + dt * dtheta2_dt
        omega1[i] = omega1[i-1] + dt * domega1_dt
        omega2[i] = omega2[i-1] + dt * domega2_dt

    return t, theta1, theta2, omega1, omega2







if __name__ == '__main__':
    
    # single pendulum
    # Set the initial conditions
    theta0 = 0.1
    omega0 = 0.0
    dt = 0.001
    T = 20

    # Simulate the dynamics of the pendulum
    t, theta, omega = simulate_pendulum(theta0, omega0, dt, T)

    #animate the pendulum
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 1), ylim=(-1, 1))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        thisx = [0, np.sin(theta[i])]
        thisy = [0, -np.cos(theta[i])]
        
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(t)),   
                                    interval=dt*1000, blit=True, init_func=init)
    plt.show()




    # # Plot the angle and angular velocity of the pendulum
    # plt.figure()
    # plt.plot(t, theta, label='angle')
    # plt.plot(t, omega, label='angular velocity')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (rad) / Angular velocity (rad/s)')
    # plt.legend()
    # plt.title('Single Pendulum dynamics')

    # double pendulum
    # Set the initial conditions
    theta1_0 = np.pi+0.001
    theta2_0 = np.pi-0.001
    omega1_0 = 0.0 
    omega2_0 = 0.0

    # Simulate the dynamics of the double pendulum
    t, theta1, theta2, omega1, omega2 = simulate_double_pendulum(theta1_0, theta2_0, omega1_0, omega2_0, dt, T)


    #animate the double pendulum
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        thisx = [0, np.sin(theta1[i]), np.sin(theta1[i]) + np.sin(theta2[i])]
        thisy = [0, -np.cos(theta1[i]), -np.cos(theta1[i]) - np.cos(theta2[i])]
        
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text
    
    ani = animation.FuncAnimation(fig, animate, range(1, len(t)),
                                    interval=dt*1000, blit=True, init_func=init)
    plt.show()



    # # Plot the angles and angular velocities of the double pendulum
    # plt.figure()
    # plt.plot(t, theta1, label='angle 1')
    # plt.plot(t, theta2, label='angle 2')
    # plt.plot(t, omega1, label='angular velocity 1')
    # plt.plot(t, omega2, label='angular velocity 2')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (rad) / Angular velocity (rad/s)')
    # plt.legend()
    # plt.title('Double Pendulum dynamics')
    # plt.show()