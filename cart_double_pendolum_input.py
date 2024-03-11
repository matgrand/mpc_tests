import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
from tqdm import tqdm

# parameters
l1 = 1  # First arm
l2 = 1.1  # Second arm
g = 9.81  # gravity
μ0 = 0.0  # friction coefficient of the cart
μ1 = 0.0  # friction coefficient first joint
μ2 = 0.0  # friction coefficient second joint
m0 = 1  # mass of the cart
m1 = 1  # mass of the first pendulum
m2 = 1  # mass of the second pendulum

dt = 0.0001  # time step
SIMT = 30 # simulation time
fps = 60 # frames per second



def create_model():
    '''returns a function that compute a time step of the equations of motion'''
    # use lagrangian mechanics to derive the equations of motion
    t = sp.symbols('t') # time
    θ0, θ1, θ2 = sp.symbols('θ0 θ1 θ2', cls=sp.Function) #states
    θ0, θ1, θ2 = θ0(t), θ1(t), θ2(t) # position of the cart and angles of the joints
    dθ0, dθ1, dθ2 = θ0.diff(t), θ1.diff(t), θ2.diff(t) # velocity of the cart and angular velocities of the joints
    ddθ0, ddθ1, ddθ2 = dθ0.diff(t), dθ1.diff(t), dθ2.diff(t) # acceleration of the cart and angular accelerations of the joints
    u = sp.symbols('u',cls=sp.Function)(t)# control input, force applied to the cart
    w1, w2, w3 = sp.symbols('w_1 w_2 w_3', cls=sp.Function)
    w1, w2, w3 = w1(t), w2(t), w3(t) # disturbance forces

    x1, y1 = θ0 + l1*sp.sin(θ1), l1*sp.cos(θ1) # position of the first pendulum
    x2, y2 = x1 + l2*sp.sin(θ2), y1 + l2*sp.cos(θ2) # position of the second pendulum
    dx1, dy1 = x1.diff(t), y1.diff(t) # velocity of the first pendulum
    dx2, dy2 = x2.diff(t), y2.diff(t) # velocity of the second pendulum

    T = 1/2 * (m0*dθ0**2 + m1*(dx1**2+dy1**2) + m2*(dx2**2+dy2**2)) #kinetic energy
    V = m1*g*y1 + m2*g*y2 #potential energy
    L = T - V #lagrangian

    # get the lagrange equations with the control input and disturbance forces
    LEQθ0 = (L.diff(θ0) - (L.diff(dθ0)).diff(t) -μ0*dθ0 +u +w1).simplify() # lagrange equation for the cart
    LEQθ1 = (L.diff(θ1) - (L.diff(dθ1)).diff(t) -μ1*dθ1 +w2).simplify() # lagrange equation for the first joint
    LEQθ2 = (L.diff(θ2) - (L.diff(dθ2)).diff(t) -μ2*dθ2 +w3).simplify() # lagrange equation for the second joint

    print('Lagrange equations derived')

    # solve the lagrange equations for the accelerations
    sol = sp.solve([LEQθ0, LEQθ1, LEQθ2], [ddθ0, ddθ1, ddθ2])
    ddθ0 = sol[ddθ0].simplify() #- μ0*dθ0 + u + w1
    ddθ1 = sol[ddθ1].simplify() #- μ1*dθ1 + w2
    ddθ2 = sol[ddθ2].simplify() #- μ2*dθ2 + w3

    print('Lagrange equations solved')

    # lambdify the equations of motion
    f = sp.lambdify((θ0, θ1, θ2, dθ0, dθ1, dθ2, u, w1, w2, w3), [ddθ0, ddθ1, ddθ2], 'numpy')

    def model(x, u, w, dt):
        '''compute a time step of the equations of motion using euler's method'''
        x, dx = x[:3], x[3:] #split the state vector into position and velocity
        dx = dx + np.array(f(*x, *dx, u, *w))*dt #compute the new velocity
        x = x + dx*dt #compute the new position
        return np.concatenate([x, dx]) #return the new state vector
        
    return model

if __name__ == '__main__':
    # create the model
    model = create_model()

    # control input
    u = 0
    # disturbance forces
    w = [0, 0, 0]

    # time vector
    t = np.arange(0, SIMT, dt)
    # initialize the state vector
    x = np.zeros((len(t), 6))
    # initial conditions
    x[0, 1], x[0, 2] = 0.1, -0.1

    # simulate the system
    for i in tqdm(range(1, len(t))):
        x[i] = model(x[i-1], u, w, dt)
        
    # animate the system
    x = x[::int(1/fps/dt)] # display one frame every n time steps]
    fig, ax = plt.subplots(figsize=(10, 10))
    lim = 1.1*(l1+l2)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')

    line1 = ax.plot([], [], 'o-', lw=2)[0]
    line2 = ax.plot([], [], 'o-', lw=2)[0]
    cart = ax.plot([], [], 'o-', lw=2)[0]

    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        cart.set_data([], [])
        time_text.set_text('')
        return line1, line2, cart, time_text
    
    def animate(i):
        x1, y1 = x[i,0] + l1*np.sin(x[i,1]), l1*np.cos(x[i,1])
        x2, y2 = x1 + l2*np.sin(x[i,2]), y1 + l2*np.cos(x[i,2])
        line1.set_data([x[i,0], x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        cart.set_data([x[i,0]-0.1, x[i,0]+0.1], [0,0])
        time_text.set_text(time_template % (i/fps))
        return line1, line2, cart, time_text
    
    ani = animation.FuncAnimation(fig, animate, range(0, len(x)), init_func=init, blit=True, interval=500/fps)

    plt.show()