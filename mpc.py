
import numpy as np; π = np.pi
import matplotlib.pyplot as plt
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu 
from numpy.random import uniform
from time import time
import multiprocess as mp #note: not multiprocessing
 
 
SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True

if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

CLIP = True # clip the control input
INPUT_CLIP = 7 # clip the control input (if < 9.81, it needs the swing up)
MIN_IMPROVEMENT = 1e-8 # minimum improvement for the gradient descent

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
if SP:
    def cost(x, eu, u0, append=False):
        n = len(x) # number of time steps
        p = (np.mod(x[:,0]+π, 2*π)-π) / π # p is between -1 and 1
        wp = np.sqrt(np.abs(p)) # use position as a weight for T
        # wp = np.abs(p) # use position as a weight for T
        ve = -1 * potential_energy(x) # potential energy
        te = 1 * kinetic_energy(x) * wp # kinetic energy
        uc = 0*0.01 * eu**2 * wp # control input
        cc = 0.1 * (eu[0] - u0)**2 * n # continuity cost
        if append: costs[0].append(te), costs[1].append(ve), costs[2].append(uc)
        final_cost =  np.sum(te) + np.sum(ve) + np.sum(uc) + cc
        return final_cost / n
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
elif CDP:
    raise NotImplementedError('Cart double pendulum not implemented')
else:
    raise NotImplementedError('Model not implemented')

g_u, g_x0, g_u0, g_t, g_p, g_c, pool = None, None, None, None, None, None, None
def grad(p, u, x0, u0, t): # multiprocessing version
    '''Calculate the gradient, using finite differences'''
    def grad_j(j):
        up = np.copy(g_u)
        up[j] += g_p
        eup = expu(up, g_t)
        if CLIP: eup = np.clip(eup, -INPUT_CLIP, INPUT_CLIP)
        return cost(simulate(g_x0, g_t, eup), eup, g_u0) - g_c
    global g_u, g_x0, g_u0, g_t, g_p, g_c, pool
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    c = cost(simulate(x0, t, eu), eu, u0) # calculate the cost
    g_u, g_x0, g_u0, g_t, g_p, g_c = u, x0, u0, t, p, c # set the global variables
    pool = mp.Pool(mp.cpu_count()) # create the pool
    d = pool.map(grad_j, list(range(len(u)))) # calculate the gradient
    pool.close(), pool.join()
    return np.array(list(d))

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
    xs = np.zeros((opt_iters, n, 2)) # state
    us, Ts, Vs = (np.zeros((opt_iters, n)) for _ in range(3)) # input, kinetic and potential energy
    xs[0], us[0], Ts[0], Vs[0] = x, eu, kinetic_energy(x), potential_energy(x) # save state + input

    for i in range(1,opt_iters):
        Jgrad = grad(lri, u, x0, u0, t) # calculate the gradient
        new_u = u - Jgrad*lri # update the control input
        eu = expu(new_u, t) # expand the control input
        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0, t, eu)  # simulate the pendulum
        new_J = cost(x, eu, u0, append=app_cost) # calculate the cost

        if new_J < (J - MIN_IMPROVEMENT): # decreasing cost
            u, J = new_u, new_J # update the control input and cost 
            lri *= 1.3 # increase the learning rate
        else: # increasing cost
            lri *= 0.9 # decrease the learning rate
            if lri < min_lr: 
                xs[i:],us[i:],Ts[i:],Vs[i:]=xs[i-1],us[i-1],Ts[i-1],Vs[i-1]# save state + input
                break # stop if the learning rate is too small

        xs[i], us[i], Ts[i], Vs[i] = x, eu, kinetic_energy(x), potential_energy(x)  # save state + input
        if i%1 == 0: print(f'  {i}/{opt_iters} cost: {J:.4f}, lri: {lri:.1e}    ', end='\r')
        
    print(f'                 cost: {J:.4f}, lri: {lri:.1e}    ', end='\r')
    return u, xs, us, Ts, Vs

xss, uss, Tss, Vss = [], [], [], [] # states and energies to plot later
def test_1iter_mpc():
    #initial state: [angle, angular velocity]
    if SP: x0 = np.array([0.2 + π 
                        , 2]) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    T = 3 # simulation time
    to = np.linspace(0, T, int(T*100)) # time steps optimization

    INPUT_SIZE = int(16*T)  # number of control inputs

    OPT_ITERS = 500 #1000
    MIN_LR = 1e-6 # minimum learning rate

    lr = 1e-1 # learning rate for the gradient descent

    ## RUN THE MPC
    u, xs, us, Ts, Vs = mpc_iter(x0, 0, to, lr, OPT_ITERS, MIN_LR, INPUT_SIZE, app_cost=True) # run the MPC

    # SIMULATION 
    t = np.linspace(0, T, int(T*100)) # time steps
    eu = expu(u, t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum

    ##  PLOTTING
    # plot the state and energies
    if SP:
        a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
        xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
        to_plot = np.array([xs1, xs2, us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1p1 = animate_pendulum(x, eu, t[1]-t[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1p1 = animate_double_pendulum(x, eu, t[1]-t[0], l1, l2, fps=60, figsize=(6,6), title='Double Pendulum')


def test_mpc():
    ''' Test the MPC'''
        #initial state: [angle, angular velocity]
    if SP: x0 = np.array([π, 0]) + uniform(-.2,.2, 2) # [rad, rad/s] # SINGLE PENDULUM
        
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    if CDP: raise NotImplementedError('Cart double pendulum not implemented')

    # Time
    T = 5 # simulation time
    OH = 1.5 # optimization horizon of the MPC
    AH = .5 # action horizon of the MPC
    assert T % AH == 0 # for more readable code

    OPT_FREQ = 100 # frequency of the time steps optimization
    SIM_FREQ = 1000 # frequency of the time steps simulation
    assert SIM_FREQ % OPT_FREQ == 0 # for more readable code

    INPUT_SIZE = int(16*OH)  # number of control inputs

    OPT_ITERS = 300 #1000
    MIN_LR = 1e-6 # minimum learning rate

    lr = 1e-1 # learning rate for the gradient descent

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
        print(f'iteration: {i+1}/{mpc_iters} cost: {cost(x, eu, eu[0]):.4f} {" "*20}')
    
    #reassemble the results
    x, ts, eu = [np.concatenate(a) for a in [all_x, all_ts, all_eu]]
    xs, us, Ts, Vs = [np.concatenate(a, axis=1) for a in [xss, uss, Tss, Vss]]


    ##  PLOTTING
    to = np.linspace(0, T, int(T*OPT_FREQ)) # time steps optimization
    if SP:
        to_plot = np.array([xs[:,:,0], xs[:,:,1], us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_pendulum(x, eu, ts[1]-ts[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_double_pendulum(x, eu, ts[1]-ts[0], l1, l2, fps=60, figsize=(6,6), title='Double Pendulum')

def single_free_evolution():
    ''' Show the free evolution of the single pendulum'''
    #initial state: [angle, angular velocity]
    x0 = np.array([0.2, 0]) # [rad, rad/s] # SINGLE PENDULUM
    # Time
    t = np.linspace(0, 20, 20*10000) # time steps
    eu = np.zeros(len(t)) # control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum
    ap1 = animate_pendulum(x, eu, t[1]-t[0], l, 60, (6,6))
    plt.show()

def plot_cost_function():
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
            p,v = xs[i], ys[j] # position and velocity
            xi = np.array([p,v]) # state vector
            p = (np.mod(p + π, 2*π) - π)/π # p is between -1 and 1
            wp = np.sqrt(np.abs(p))
            # wp = np.abs(p)
            wv = np.sqrt(np.abs(v))
            # ve = -10 * (potential_energy(xi))
            # te = 1 * kinetic_energy(xi)# * wp
            ve = wp
            te = wv
            Z[j,i] = te + ve

    #plot the cost function
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.set_zlabel('cost')

if __name__ == '__main__':
    main_start = time()

    plot_cost_function()
    plt.show()
    # single_free_evolution()
    test_1iter_mpc()
    plt.show()
    test_mpc()



    print(f'\nTotal time: {time()-main_start:.2f} s')
    plt.show()