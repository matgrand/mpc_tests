
import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from plotting import *
from inputs import addittive_resample as expu
# from inputs import frequency_resample as expu 
from numpy.random import uniform, normal
from time import time
import multiprocess as mp #note: not multiprocessing
import os
# np.set_printoptions(precision=3, suppress=True) #set numpy print options
# np.random.seed(0)
 
SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP
OPT_FREQ = 2*60 # frequency of the time steps optimization
SIM_FREQ = 10*OPT_FREQ # frequency of the time steps simulation
assert SIM_FREQ % OPT_FREQ == 0 # for more readable code
CLIP = True # clip the control input
INPUT_CLIP = 10 # clip the control input (if < 9.81, it needs the swing up)
MIN_IMPROVEMENT = 1e-8 # minimum improvement for the gradient descent
SGD = 1 # stochastic gradient descent percentage of the gradient

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True
if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

# function to simulate a run
def simulate(x0, t, eu):
    '''Simulate the pendulum'''
    n, l1, dt = len(t), len(x0), t[1]-t[0] # number of time steps, control inputs, time step
    x = np.zeros((n, l1)) # [θ, dθ, ...] -> state vector
    x[0] = x0 # initial conditions
    for i in range(1, n): x[i] = step(x[i-1], eu[i], dt)   
    return x

# cost function
costs = [[],[],[]] # costs to plot later
labels = ['T', 'V', 'u']
if SP:
    def cost(x, eu, u0, append=False):
        n = len(x) # number of time steps
        # p = (np.mod(x[:,0]+π, 2*π)-π) / π # p is between -1 and 1
        p = x[:,0] / π # p is between -1 and 1
        wp = np.sqrt(np.abs(p)) # use position as a weight for T
        # wp = np.abs(p) # use position as a weight for T
        if n > 1: tw = np.linspace(0, 1, n)#**2 # time weight
        elif n == 1: tw = np.array([1])
        else: raise ValueError('n must be > 0')
        ve = -1 * potential_energy(x) * tw # potential energy
        te = 1 * kinetic_energy(x) * wp * tw # kinetic energy
        uc = 0*0.01 * eu**2 * wp # control input
        cc = 0.1 * (eu[0] - u0)**2 * n # continuity cost
        if append: costs[0].append(te), costs[1].append(ve), costs[2].append(uc)
        final_cost =  np.sum(te) + np.sum(ve) + np.sum(uc) + cc
        return final_cost / n
    def cost(x, eu, u0, append=False):
        n = len(x) # number of time steps
        w = np.linspace(0, 1, n) # time weight
        ve = -1 * potential_energy(x) * w # potential energy
        return np.sum(ve) / n
elif DP:
    def cost(x, eu, u0, append=False):
        n = len(x) # number of time steps
        p0 = (np.mod(x[:,0]+π, 2*π)-π) / π # p is between -1 and 1
        p1 = (np.mod(x[:,1]+π, 2*π)-π) / π # p is between -1 and 1
        wp0 = np.sqrt(np.abs(p0)) # use position as a weight for T
        wp1 = np.sqrt(np.abs(p1)) # use position as a weight for T
        wp = wp0 * wp1 # use position as a weight for T
        ve = -1 * potential_energy(x) # potential energy
        te = 1 * kinetic_energy(x) * wp # kinetic energy
        uc = 0*0.01 * eu**2 * wp # control input
        cc = 0*0.1 * (eu[0] - u0)**2 * n # continuity cost
        if append: costs[0].append(te), costs[1].append(ve), costs[2].append(uc)
        final_cost =  np.sum(te) + np.sum(ve) + np.sum(uc) + cc
        return final_cost / n
    def cost(x, eu, u0, append=False):
        n = len(x) # number of time steps
        α0, α1 = x[:,0], x[:,1]# angles
        # v0, v1 = x[:,2]/π, x[:,3]/π # angular velocities
        # c = 1 - np.exp(-3 * p0**2 -1 * p1**2) # cost
        c = 1 * α0**2 + 1 * α1**2 # height of the second pendulum  
        # c = c * np.linspace(0, 1, n)**2 # time weight
        if append: costs[0].append(α0), costs[1].append(α1), costs[2].append(c)
        return np.sum(c) / n
elif CDP:
    raise NotImplementedError('Cart double pendulum not implemented')
else:
    raise NotImplementedError('Model not implemented')

g_c, pool = None, None # global cost and pool for multiprocessing
def grad(p, temp, u, x0, u0, t): # multiprocessing version
    '''Calculate the gradient, using finite differences'''
    def grad_j(j):
        up = np.copy(u) # copy the control input
        up[j] += p * (1 + temp) * normal(0,1) # perturb the control input and add noise
        eup = expu(up, t)
        if CLIP: eup = np.clip(eup, -INPUT_CLIP, INPUT_CLIP)
        dc = cost(simulate(x0, t, eup), eup, u0) - g_c
        if dc == 0: return 1 #
        else: return dc
    global g_c, pool
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    g_c = cost(simulate(x0, t, eu), eu, u0) # calculate the cost and set it as the global variable
    pool = mp.Pool() # create the pool
    idxs = list(np.random.choice(len(u), int(SGD*len(u)), replace=False)) # stochastic gradient descent
    d = pool.map(grad_j, idxs) # calculate the gradient
    pool.close(), pool.join()
    ret = np.zeros(len(u))
    ret[idxs] = d
    return ret

def mpc_iter(x0, u0, t, lr, opt_iters, min_lr, input_size, app_cost=False):
    ''' Model Predictive Control
        x0: initial state
        t: time steps 
    '''
    #initialize input
    u = np.zeros(input_size) # control input
    # u = 0.2*INPUT_CLIP*uniform(-1,1,input_size) # control input
    # u = 0.2* x0[0]**2 * uniform(-1,1,input_size) # control input
    n = len(t) # number of time steps
    lri = lr # learning rate
    # first iteration
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum
    J = cost(x, eu, u0, append=app_cost) # calculate the cost
    # debug: save the states and control inputs
    xs = np.zeros((opt_iters, n, len(x0))) # state
    us, Ts, Vs = (np.zeros((opt_iters, n)) for _ in range(3)) # input, kinetic and potential energy
    xs[0], us[0], Ts[0], Vs[0] = x, eu, kinetic_energy(x), potential_energy(x) # save state + input

    TEMPS = np.linspace(100, 0, opt_iters) # temperature (for the injected noise)

    start_time = time()
    for i in range(1,opt_iters):
        Jgrad = grad(lri, TEMPS[i], u, x0, u0, t) # calculate the gradient
        new_u = u - Jgrad*lri # update the control input
        eu = expu(new_u, t) # expand the control input
        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0, t, eu)  # simulate the pendulum
        new_J = cost(x, eu, u0, append=app_cost) # calculate the cost
        if new_J < (J-MIN_IMPROVEMENT):
            u, J = new_u, new_J # update the control input and cost 
            lri *= 1 # increase the learning rate 1.3
        else: # increasing cost
            lri *= 1 # decrease the learning rate 0.9
            if lri < min_lr: 
                xs[i:],us[i:],Ts[i:],Vs[i:]=xs[i-1],us[i-1],Ts[i-1],Vs[i-1]# save state + input
                break # stop if the learning rate is too small

        xs[i], us[i], Ts[i], Vs[i] = x, eu, kinetic_energy(x), potential_energy(x)  # save state + input
        if i%1 == 0: print(f'  {i}/{opt_iters} cost: {J:.4f}, lri: {lri:.1e}, temp: {TEMPS[i]:.4f}, eta: {(time()-start_time)*(opt_iters-i)/i:.2f} s    ', end='\r')
        
    print(f'                 cost: {J:.4f}, lri: {lri:.1e}    ', end='\r')
    return u, xs, us, Ts, Vs

xss, uss, Tss, Vss = [], [], [], [] # states and energies to plot later
def test_1iter_mpc():
    print('Running the MPC 1 iteration...')
    #initial state: [angle, angular velocity]
    if SP: x0 = np.array([π+0.1,2]) # [π+0.1,2] # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.01 + π
                          ,-0.01 + π
                          ,0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    if SP: T = 5 # simulation time
    if DP: T = 3 # simulation time
    to = np.linspace(0, T, int(T*OPT_FREQ)) # time steps optimization

    if SP: INPUT_SIZE = int(8*T)  # number of control inputs
    if DP: INPUT_SIZE = int(16*T)  # number of control inputs

    OPT_ITERS = int(200 * (2-SGD)) #1000
    MIN_LR = 1e-10 # minimum learning rate

    lr = .2 # learning rate for the gradient descent

    ## RUN THE MPC
    u, xs, us, Ts, Vs = mpc_iter(x0, 0.0, to, lr, OPT_ITERS, MIN_LR, INPUT_SIZE, app_cost=True) # run the MPC

    # SIMULATION 
    t = np.linspace(0, T, int(T*SIM_FREQ)) # time steps
    eu = expu(u, t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum

    x_free = simulate(x0, t, 0*eu) # simulate the pendulum without control input

    ##  PLOTTING
    # plot the state and energies
    a12, a1f1, a1p1 = None, None, None
    if SP:
        # a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
        xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
        to_plot = np.array([xs1, xs2, us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1f1 = animate_pendulum(x_free, 0*eu, t[1]-t[0], l1, fps=60, figsize=(10,10), title='Pendulum free')
        a1p1 = animate_pendulum(x, eu, t[1]-t[0], l1, fps=60, figsize=(10,10), title='Pendulum')
    if DP:
        # a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=False)
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1f1 = animate_double_pendulum(x_free, 0*eu, t[1]-t[0], l1, l2, fps=60, figsize=(10,10), title='Double Pendulum free')
        a1p1 = animate_double_pendulum(x, eu, t[1]-t[0], l1, l2, fps=60, figsize=(10,10), title='Double Pendulum')
    print()
    return a12, a13, a1p1, a1f1

def test_mpc():
    ''' Test the MPC'''
        #initial state: [angle, angular velocity]
    if SP: x0 = np.array([π-0.3, 0]) #+ uniform(-.2,.2, 2) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    if SP:
        T = 5 # simulation time 5
        OH = 1.5 # optimization horizon of the MPC 1.5
        AH = .5 # action horizon of the MPC .5
    if DP:
        T = 5
        OH = 3
        AH = .5
    assert T % AH == 0 # for more readable code

    INPUT_SIZE = int(16*OH)  # number of control inputs

    OPT_ITERS = int(100 * (2-SGD)) #150
    MIN_LR = 1e-6 # minimum learning rate

    lr = 1 # learning rate for the gradient descent

    mpc_iters = int(T/AH) # number of MPC iterations

    x0i, u0i = x0, 0 # initial state and input
    u, uss, xss, Tss, Vss = [],[],[],[],[] # states and energies to plot later
    all_x, all_ts, all_eu = [],[],[] # states, time steps and control inputs to plot later
    for i in range(mpc_iters):
        ## RUN THE MPC
        toi = np.linspace(0, OH, int(OH*OPT_FREQ)) # time steps optimization
        tai = np.linspace(0, AH, int(AH*SIM_FREQ)) # time steps action
        tsi = np.linspace(0, OH, int(OH*SIM_FREQ)) # time steps simulation

        ui, xs, us, Ts, Vs = mpc_iter(x0i, u0i, toi, lr, OPT_ITERS, MIN_LR, INPUT_SIZE) # run the MPC
        eu = expu(ui, tsi) # expand the control input to simulation times        
        eu = eu[:len(tai)] # crop the control input to the action horizon

        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0i, tai, eu) # simulate the pendulum
        x0i, u0i = x[-1], eu[-1] # update the initial state and control input

        # save the results
        cr = int(len(tai) * OPT_FREQ/SIM_FREQ) # crop index 
        xs, us, Ts, Vs = xs[:,:cr], us[:,:cr], Ts[:,:cr], Vs[:,:cr] # crop the results
        all_x.append(x), all_ts.append(tai), all_eu.append(eu) 
        u.append(ui), uss.append(us), xss.append(xs), Tss.append(Ts), Vss.append(Vs)
        print(f'iteration: {i+1}/{mpc_iters} cost: {cost(x, eu, eu[0]):.4f} {" "*25}')
    
    #reassemble the results
    x, ts, eu = [np.concatenate(a) for a in [all_x, all_ts, all_eu]]
    xs, us, Ts, Vs = [np.concatenate(a, axis=1) for a in [xss, uss, Tss, Vss]]

    ##  PLOTTING
    to = np.linspace(0, T, int(T*OPT_FREQ)) # time steps optimization
    if SP:
        to_plot = np.array([xs[:,:,0], xs[:,:,1], us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_pendulum(x, eu, ts[1]-ts[0], l1, fps=60, figsize=(10,10), title='Pendulum')
    if DP:
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_double_pendulum(x, eu, ts[1]-ts[0], l1, l2, fps=60, figsize=(10,10), title='Double Pendulum')
    print()
    return a3, ap1

def free_evolutions():
    ''' Show the free evolution of the single pendulum'''
    #initial state: [angle, angular velocity]
    if SP: x0 = np.array([0.2, .1]) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    # Time
    T = 16 # simulation time
    f = SIM_FREQ # frequency of the time steps

    t = np.linspace(0, T, int(T*f)) # time steps
    dt = t[1]-t[0] # time step
    eu = np.zeros_like(t) # control input (zero)
    print(f'T: {T}, f: {f}, dt: {dt:.4f}')

    # multiple reverse simulations from instability point
    NS = 50 # number of simulations
    x0s = np.array([0, 0]) + normal(0,1,(NS,2)) # initial states
    xs = np.zeros((NS, len(t), len(x0))) # states
    for i in tqdm(range(NS)): xs[i] = simulate(x0s[i], t, eu) # reverse simulate the pendulums
    Ts = np.array([kinetic_energy(x) for x in xs]) # kinetic energy
    Vs = np.array([potential_energy(x) for x in xs]) # potential energy
    TVs = Ts + Vs # total energy

    #plot the kinetic and potential energies
    tvt, ax = plt.subplots(3,1, figsize=(10,8))
    ax[0].plot(t, Ts.T, alpha=0.5)
    ax[0].set_title('Kinetic Energy')
    ax[1].plot(t, Vs.T, alpha=0.5)
    ax[1].set_title('Potential Energy')
    ax[2].plot(t, TVs.T, alpha=0.5)
    ax[2].set_title('Total Energy')

    XS, US = xs, np.zeros((NS, len(t))) # states and control inputs
    if SP: 
        fea = animate_pendulums(XS,US,dt,l1,60,(6,6))
        fest = plot_state_trajectories(XS, center=True)
    elif DP: 
        fea = animate_double_pendulums(XS,US,dt,l1,l2,60,(6,6))
    else: fea, fest = None, None
    return fea, fest, tvt

def plot_cost_function():
    assert SP, 'Single pendulum only for now...'
    #create a 3d plot
    from matplotlib import cm
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #create a meshgrid
    N = 200
    xs = np.linspace(-1, 1, N)
    ys = np.linspace(-1, 1, N)

    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)

    for i in tqdm(range(N)):
        for j in range(N):
            xij = np.array([xs[i], ys[j]]).reshape(1,2)
            eu = np.zeros((1,))
            Z[j,i] = cost(xij, eu, 0)

    #plot the cost function
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.set_zlabel('cost')
    return fig

if __name__ == '__main__':
    os.system('clear')
    main_start = time()

    # pc = plot_cost_function()
    sf = free_evolutions()
    t1 = test_1iter_mpc()
    tm = test_mpc()

    print(f'\nTotal time: {time()-main_start:.2f} s')
    plt.show()
    exit()