import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm
import os
from plotting import animate_cart_double, C
# parameters
l1 = 1  # First arm
l2 = 1  # Second arm
g = 9.81  # gravity
μ0 = 0.4  # friction coefficient of the cart
μ1 = 0.4  # friction coefficient first joint
μ2 = 0.4  # friction coefficient second joint
m0 = 1  # mass of the cart
m1 = 1  # mass of the first pendulum
m2 = 1  # mass of the second pendulum

dt = 0.001  # time step
SIMT = 20 # simulation time

class PhysicalSystem():
    def __init__(self, l1, l2, g, μ0, μ1, μ2, m0, m1, m2):
        self.l1, self.l2, self.g = l1, l2, g
        self.μ0, self.μ1, self.μ2 = μ0, μ1, μ2
        self.m0, self.m1, self.m2 = m0, m1, m2
        eq_path = f'models/cart_double_{l1}_{l2}_{g}_{μ0}_{μ1}_{μ2}_{m0}_{m1}_{m2}.txt'
        # use lagrangian mechanics to derive the equations of motion
        t = sp.symbols('t') # time
        θ0, θ1, θ2 = sp.symbols('θ0 θ1 θ2', cls=sp.Function) #states
        θ0, θ1, θ2 = θ0(t), θ1(t), θ2(t) # position of the cart and angles of the joints
        dθ0, dθ1, dθ2 = θ0.diff(t), θ1.diff(t), θ2.diff(t) # velocity of the cart and angular velocities of the joints
        ddθ0, ddθ1, ddθ2 = dθ0.diff(t), dθ1.diff(t), dθ2.diff(t) # acceleration of the cart and angular accelerations of the joints
        u = sp.symbols('u',cls=sp.Function)(t)# control input, force applied to the cart
        w1, w2, w3 = sp.symbols('w_1 w_2 w_3', cls=sp.Function)
        w1, w2, w3 = w1(t), w2(t), w3(t) # disturbance forces

        if not os.path.exists(eq_path): # if the equations of motion are not saved
            x1, y1 = θ0 + l1*sp.sin(θ1), l1*sp.cos(θ1) # position of the first pendulum
            x2, y2 = x1 + l2*sp.sin(θ2), y1 + l2*sp.cos(θ2) # position of the second pendulum
            dx1, dy1 = x1.diff(t), y1.diff(t) # velocity of the first pendulum
            dx2, dy2 = x2.diff(t), y2.diff(t) # velocity of the second pendulum

            min_V = -m1*g*l1 - m2*g*(l1+l2) # minimum potential energy, to set the zero level

            T = 1/2 * (m0*dθ0**2 + m1*(dx1**2+dy1**2) + m2*(dx2**2+dy2**2)) #kinetic energy
            V = m1*g*y1 + m2*g*y2 - min_V #potential energy
            L = T - V #lagrangian

            # get the lagrange equations with the control input and disturbance forces
            LEQθ0 = L.diff(θ0) - (L.diff(dθ0)).diff(t) -μ0*dθ0 +u +w1 # lagrange equation for the cart
            LEQθ1 = L.diff(θ1) - (L.diff(dθ1)).diff(t) -μ1*dθ1 +w2 # lagrange equation for the first joint
            LEQθ2 = L.diff(θ2) - (L.diff(dθ2)).diff(t) -μ2*dθ2 +w3 # lagrange equation for the second joint
            print('Lagrange equations derived')

            print(f'leq0: {LEQθ0}')

            # solve the lagrange equations for the accelerations
            sol = sp.solve([LEQθ0, LEQθ1, LEQθ2], [ddθ0, ddθ1, ddθ2], simplify=False)
            ddθ0 = sol[ddθ0].simplify() #- μ0*dθ0 + u + w1
            ddθ1 = sol[ddθ1].simplify() #- μ1*dθ1 + w2
            ddθ2 = sol[ddθ2].simplify() #- μ2*dθ2 + w3
            print('Lagrange equations solved')

            #save the equations of motion as txt file
            with open(eq_path, 'w') as file: 
                file.write(str(ddθ0)+'\n'+str(ddθ1)+'\n'+str(ddθ2)+'\n'+str(T)+'\n'+str(V))
            del ddθ0, ddθ1, ddθ2, LEQθ0, LEQθ1, LEQθ2, T, V

        #load the equations of motion from the txt file
        with open(eq_path, 'r') as file: ddθ0, ddθ1, ddθ2, T, V = [sp.sympify(line) for line in file]

        # lambdify the equations of motion
        self.f = sp.lambdify((θ0, θ1, θ2, dθ0, dθ1, dθ2, u, w1, w2, w3), [ddθ0, ddθ1, ddθ2], 'numpy')

        #lamdifify the kinetic and potential energy
        self.fT = sp.lambdify((θ0, θ1, θ2, dθ0, dθ1, dθ2), T, 'numpy')
        self.fV = sp.lambdify((θ0, θ1, θ2, dθ0, dθ1, dθ2), V, 'numpy')

    def step(self, x, u, w, dt):
        '''compute a time step of the equations of motion using euler's method'''
        x, dx = x[:3], x[3:] #split the state vector into position and velocity
        dx = dx + np.array(self.f(*x, *dx, u, *w))*dt #compute the new velocity
        x = x + dx*dt #compute the new position
        return np.concatenate([x, dx]) #return the new state vector

    def kinetic_energy(self, x):
        ''' compute the kinetic energy of the system
            T = 1/2 * (m0*dθ0**2 + m1*(dx1**2+dy1**2) + m2*(dx2**2+dy2**2)) #kinetic energy
        '''
        return self.fT(*x.T)
    
    def potential_energy(self, x):
        '''compute the potential energy of the system
            V = m1*g*y1 + m2*g*y2
        '''
        return self.fV(*x.T)

class PID():
    '''simple PID controller'''
    def __init__(self, kp, ki, kd, dt):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral = 0
        self.previous_error = 0

    def control(self, error):
        '''compute the control input'''
        self.integral += error*self.dt
        derivative = (error - self.previous_error)/self.dt
        self.previous_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative
    


if __name__ == '__main__':
    # create the model
    sys = PhysicalSystem(l1, l2, g, μ0, μ1, μ2, m0, m1, m2)

    #controller
    pid = PID(5, .5, 1, dt)

    # time vector
    t = np.arange(0, SIMT, dt)
    # initialize the state vector
    x = np.zeros((len(t), 6))
    u = np.zeros(len(t))
    # initial conditions
    x[0, 1], x[0, 2] = 1,-1
    rand_freq = np.random.uniform(0.1, 1.5)

    # simulate the system
    for i in tqdm(range(1, len(t))):
        # w = np.random.normal(0, .5, 3) # generate random disturbance forces
        # ui = pid.control(-x[i-1, 0])
        ui = 13*np.sin(2*π*rand_freq*t[i]) - 2*x[i-1,0] # control input
        w = [0, 0, 0]
        x[i] = sys.step(x[i-1], ui, w, dt) 
        u[i] = ui

    T = sys.kinetic_energy(x)
    V = sys.potential_energy(x)

    # plot 
    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    titles = ['x', 'θ1', 'θ2', 'dx', 'dθ1', 'dθ2']
    for a in range(2):
        for b in range(3):
            ax[a, b].plot(t, x[:, 3*a+b], color=C)
            ax[a, b].set_title(titles[3*a+b])
            ax[a, b].grid(True)
    ax[2, 0].plot(t, T, label='T, kinetic energy', color=C)
    ax[2, 0].plot(t, V, label='V, potential energy', color='blue')
    ax[2, 0].plot(t, T+V, '--', label='T+V', color='white')
    # ax[2, 0].plot(t, T-V, '--', label='T-V', color='green')
    ax[2, 0].set_title('Energies')
    ax[2, 0].legend()
    ax[2, 0].grid(True)
    ax[2, 0].set_yscale('log')
    ax[2, 1].plot(t, u, label='u, control input', color=C)
    ax[2, 1].set_title('Control input')
    ax[2, 1].grid(True)
    plt.tight_layout()

    a1 = animate_cart_double(x, u, dt, l1, l2, figsize=(10, 10))
    plt.show()