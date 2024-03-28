import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu 
 
SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True

if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

CLIP = False # clip the control input
INPUT_CLIP = 50 # clip the control input

# function to simulate a run
def simulate(x0, t, eu):
    '''Simulate the pendulum'''
    n = len(t) # number of time steps
    x = np.zeros((n, 2)) # [θ, dθ] -> state vector
    x[0] = x0 # initial conditions
    for i in range(1, n): x[i] = step(x[i-1], eu[i], t[1]-t[0])   
    return x

# cost function
costs = [[],[],[]] # costs to plot later
labels = ['T', 'V', 'u']
# def cost(x, eu, append=False):
#     '''Cost function'''
#     n = len(x) # number of time steps
#     te = kinetic_energy(x) # kinetic energy
#     ve = potential_energy(x) # potential energy
#     wl = np.linspace(0, 1, n) # weight for the time
#     te = 60 * te * wl # kinetic energy
#     ve = -100 * ve # potential energy
#     eu = 0 * eu**2 * wl # control input
#     # debug, append the energies
#     if append: costs[0].append(te), costs[1].append(-ve), costs[2].append(eu)
#     final_cost = np.sum(te) + np.sum(ve) + np.sum(eu)
#     return final_cost / n 
def cost(x, eu, append=False):
    n = len(x) # number of time steps
    p = (np.mod(x[:,0] + np.pi, 2*np.pi) - np.pi)/np.pi # p is between -1 and 1
    wp = np.sqrt(np.abs(p)) # use position as a weight
    ve = -100 * potential_energy(x) # potential energy
    te = 300 * kinetic_energy(x) * wp # kinetic energy
    uc = 5 * eu**2 #* np.linspace(0, 1, n) # control input
    if append: costs[0].append(te), costs[1].append(ve), costs[2].append(uc)
    final_cost =  np.sum(te) + np.sum(ve) + np.sum(uc)
    return final_cost / n

def grad(p, u, x0, t):
    '''Calculate the gradient, using finite differences'''
    d = np.zeros(len(u)) # initialize the gradient 
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    c = cost(simulate(x0, t, eu), eu) # calculate the cost
    for j in range(len(u)):
        up = np.copy(u) # copy the control input
        up[j] += p # perturb the control input
        eup = expu(up, t) # expand the control input
        if CLIP: eup = np.clip(eup, -INPUT_CLIP, INPUT_CLIP) # clip the control input
        d[j] = cost(simulate(x0, t, eup), eup) - c # calculate the gradient
    return d

def mpc_iter(x0, t, lr, opt_iters, min_lr, input_size):
    ''' Model Predictive Control
        x0: initial state
        t: time steps 
    '''
    u = np.zeros(input_size) # control input
    n = len(t) # number of time steps
    # first iteration
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum
    J = cost(x, eu, append=True) # calculate the cost
    lri = lr # learning rate

    # debug: save the states and control inputs
    xs = np.zeros((opt_iters, n, 2)) # state
    us, Ts, Vs = (np.zeros((opt_iters, n)) for _ in range(3)) # control input, kinetic and potential energy
    xs[0], us[0], Ts[0], Vs[0] = x, eu, kinetic_energy(x), potential_energy(x) # save the state and control input

    for i in range(1,opt_iters):
        Jgrad = grad(lri, u, x0, t) # calculate the gradient
        new_u = u - Jgrad*lri # update the control input
        eu = expu(new_u, t) # expand the control input
        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0, t, eu)  # simulate the pendulum
        new_J = cost(x, eu, append=True) # calculate the cost

        if new_J < J: # decreasing cost
            u, J = new_u, new_J # update the control input and cost 
            lri *= 1.2 # increase the learning rate
        else: # increasing cost
            lri *= 0.95 # decrease the learning rate
            if lri < min_lr: 
                xs[i:],us[i:],Ts[i:],Vs[i:]=xs[i-1],us[i-1],Ts[i-1],Vs[i-1] # save the state and control input
                break # stop if the learning rate is too small

        xs[i], us[i], Ts[i], Vs[i] = x, eu, kinetic_energy(x), potential_energy(x)  # save the state and control input
        if i%1 == 0: print(f'  {i}/{opt_iters} cost: {J:.2f}, lri: {lri:.1e}    ', end='\r')
        
    print(f'                 cost: {J:.2f}, lri: {lri:.1e}    ', end='\r')
    return u, xs, us, Ts, Vs

xss, uss, Tss, Vss = [], [], [], [] # states and energies to plot later
def test_1iter_mpc():
    #initial state: [angle, angular velocity]
    if SP: x0 = np.array([0.2 #+ np.pi 
                        , 3]) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    T = 1 # simulation time
    to = np.linspace(0, T, int(T*100)) # time steps optimization

    INPUT_SIZE = int(16)  # number of control inputs

    OPT_ITERS = 500 #1000
    MIN_LR = 1e-4 # minimum learning rate

    lr = 1e-1 # learning rate for the gradient descent

    ## RUN THE MPC
    u, xs, us, Ts, Vs = mpc_iter(x0, to, lr, OPT_ITERS, MIN_LR, INPUT_SIZE) # run the MPC

    # SIMULATION 
    t = np.linspace(0, T, int(T*100)) # time steps
    eu = expu(u, t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum

    ##  PLOTTING
    # plot the state and energies
    if SP:
        a2 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=False)
        xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
        to_plot = np.array([xs1, xs2, us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_pendulum(x, eu, t[1]-t[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        raise
    plt.show()

def test_mpc():
    ''' Test the MPC'''
        #initial state: [angle, angular velocity]
    if SP: x0 = np.array([0.2 + np.pi 
                        , 0]) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    T = 5 # simulation time
    H = 1 # horizon of the MPC
    assert T % H == 0, 'T must be divisible by H' # for more readable code

    OPT_FREQ = 100 # frequency of the time steps optimization
    SIM_FREQ = 100 # frequency of the time steps simulation

    if SP: INPUT_SIZE = int(16)  # number of control inputs
    if DP: INPUT_SIZE = int(16)  # number of control inputs

    OPT_ITERS = 200 #1000
    MIN_LR = 1e-6 # minimum learning rate

    if SP: lr = 1e-1 # learning rate for the gradient descent

    mpc_iters = int(T/H) # number of MPC iterations

    x0i = x0 # initial state
    u, uss, xss, Tss, Vss, costss = [],[],[],[],[],[] # states and energies to plot later
    for i in range(mpc_iters):
        global costs
        costs = [[],[],[]] # reset the costs
        ## RUN THE MPC
        toi = np.linspace(0, H, int(H*OPT_FREQ)) # time steps optimization
        tsi = np.linspace(0, H, int(H*SIM_FREQ)) # time steps simulation

        ui, xs, us, Ts, Vs = mpc_iter(x0i, toi, lr, OPT_ITERS, MIN_LR, INPUT_SIZE) # run the MPC
        eu = expu(ui, tsi) # expand the control input to simulation times
        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0, tsi, eu) # simulate the pendulum
        x0i = x[-1] # last state of the simulation is the initial state of the next optimization

        # save the results
        u.append(ui), uss.append(us), xss.append(xs), Tss.append(Ts), Vss.append(Vs), costss.append(np.array(costs))
        #print recap
        print(f'iteration: {i+1}/{mpc_iters} cost: {cost(x, eu):.2f}    ')
    
    #reassemble the results
    eu = np.concatenate([expu(ui, tsi) for ui in u]) # control input
    xs, us, Ts, Vs, costs = [np.concatenate(a, axis=1) for a in [xss, uss, Tss, Vss, costss]]

    # final simulation
    ts = np.linspace(0, T, int(T*SIM_FREQ)) # time steps
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, ts, eu) # simulate the pendulum

    ##  PLOTTING
    # plot the state and energies
    to = np.linspace(0, T, int(T*OPT_FREQ)) # time steps optimization
    if SP:
        a2 = animate_costs(costs, labels=labels, figsize=(6,4), logscale=True)
        xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
        to_plot = np.array([xs1, xs2, us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_pendulum(x, eu, ts[1]-ts[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        raise

    plt.show()


def single_free_evolution():
    ''' Show the free evolution of the single pendulum'''
    #initial state: [angle, angular velocity]
    x0 = np.array([0.2, 0]) # [rad, rad/s] # SINGLE PENDULUM
    # Time
    t = np.linspace(0, 20, 20*100000) # time steps
    eu = np.zeros(len(t)) # control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum
    ap1 = animate_pendulum(x, eu, t[1]-t[0], l, 60, (6,6))
    plt.show()

def plot_cost_function():
    #create a 3d plot
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #create a meshgrid
    N = 200
    xs = np.linspace(-1, 1, N)
    ys = np.linspace(-10, 10, N)

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)

    for i in tqdm(range(N)):
        for j in range(N):
            p,v = xs[i], ys[j] # position and velocity
            xi = np.array([p,v])
            u = 0
            p = (np.mod(xi[0] + np.pi, 2*np.pi) - np.pi)/np.pi # p is between -1 and 1
            wp = np.sqrt(np.abs(0.1*p))/0.1
            ve = -100 * potential_energy(xi)
            te = 10 * kinetic_energy(xi) * wp
            uc = 0 * u**2
            Z[i,j] = te + ve + uc


    #plot the cost function
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.set_zlabel('cost')
    plt.show()


if __name__ == '__main__':

    # single_free_evolution()
    test_1iter_mpc()
    # test_mpc()
    # plot_cost_function()