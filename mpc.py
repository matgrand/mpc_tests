
import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu 
from numpy.random import uniform, normal
from time import time
import multiprocess as mp #note: not multiprocessing
import os
np.set_printoptions(precision=3, suppress=True) #set numpy print options
# os.environ['KMP_DUPLICATE_LIB_OK']='True' # for the multiprocessing to work on MacOS
# np.random.seed(0)
 
SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP
OPT_FREQ = 600 # frequency of the time steps optimization
SIM_FREQ = 10*OPT_FREQ # frequency of the time steps simulation
assert SIM_FREQ % OPT_FREQ == 0 # for more readable code
CLIP = True # clip the control input
INPUT_CLIP = 6 # clip the control input (if < 9.81, it needs the swing up)
MIN_IMPROVEMENT = 1e-8 # minimum improvement for the gradient descent
SGD = .5 # stochastic gradient descent percentage of the gradient

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True
if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

# function to simulate a run
def simulate(x0, t, eu):
    '''Simulate the pendulum'''
    n, l, dt = len(t), len(x0), t[1]-t[0] # number of time steps, control inputs, time step
    x = np.zeros((n, l)) # [θ, dθ, ...] -> state vector
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
        tw = np.linspace(0, 1, n)#**2 # time weight
        ve = -1 * potential_energy(x) * tw # potential energy
        te = 1 * kinetic_energy(x) * wp * tw # kinetic energy
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

g_u, g_x0, g_u0, g_t, g_p, g_c, pool, g_ps = None, None, None, None, None, None, None, None
def grad(p, u, x0, u0, t): # multiprocessing version
    '''Calculate the gradient, using finite differences'''
    def grad_j(j):
        up = np.copy(g_u)
        # up[j] += g_p
        up[j] += g_ps[j]
        eup = expu(up, g_t)
        if CLIP: eup = np.clip(eup, -INPUT_CLIP, INPUT_CLIP)
        dc = cost(simulate(g_x0, g_t, eup), eup, g_u0) - g_c
        if dc == 0: return 10
        else: return dc
    global g_u, g_x0, g_u0, g_t, g_p, g_c, pool, g_ps
    eu = expu(u,t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    c = cost(simulate(x0, t, eu), eu, u0) # calculate the cost
    
    # ps = None
    ps = normal(p, p/6, len(u)) # random perturbations

    g_u, g_x0, g_u0, g_t, g_p, g_c, g_ps = u, x0, u0, t, p, c, ps # set the global variables
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

    start_time = time()
    for i in range(1,opt_iters):
        Jgrad = grad(lri, u, x0, u0, t) # calculate the gradient
        new_u = u - Jgrad*lri # update the control input
        eu = expu(new_u, t) # expand the control input
        if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP)
        x = simulate(x0, t, eu)  # simulate the pendulum
        new_J = cost(x, eu, u0, append=app_cost) # calculate the cost
        if new_J < (J-MIN_IMPROVEMENT):
            u, J = new_u, new_J # update the control input and cost 
            lri *= 1.3 # increase the learning rate
        else: # increasing cost
            lri *= 0.9 # decrease the learning rate
            if lri < min_lr: 
                xs[i:],us[i:],Ts[i:],Vs[i:]=xs[i-1],us[i-1],Ts[i-1],Vs[i-1]# save state + input
                break # stop if the learning rate is too small

        xs[i], us[i], Ts[i], Vs[i] = x, eu, kinetic_energy(x), potential_energy(x)  # save state + input
        if i%1 == 0: print(f'  {i}/{opt_iters} cost: {J:.4f}, lri: {lri:.1e}, eta: {(time()-start_time)*(opt_iters-i)/i:.2f} s    ', end='\r')
        
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

    OPT_ITERS = int(500 * (2-SGD)) #1000
    MIN_LR = 1e-10 # minimum learning rate

    lr = 1e2 # learning rate for the gradient descent

    ## RUN THE MPC
    u, xs, us, Ts, Vs = mpc_iter(x0, 0, to, lr, OPT_ITERS, MIN_LR, INPUT_SIZE, app_cost=True) # run the MPC

    # SIMULATION 
    t = np.linspace(0, T, int(T*SIM_FREQ)) # time steps
    eu = expu(u, t) # expand the control input
    if CLIP: eu = np.clip(eu, -INPUT_CLIP, INPUT_CLIP) # clip the control input
    x = simulate(x0, t, eu) # simulate the pendulum

    x_free = simulate(x0, t, 0*eu) # simulate the pendulum without control input

    ##  PLOTTING
    # plot the state and energies
    if SP:
        a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=True)
        xs1, xs2 = xs[:, :, 0], xs[:, :, 1] # angles and angular velocities splitted
        to_plot = np.array([xs1, xs2, us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1f1 = animate_pendulum(x_free, 0*eu, t[1]-t[0], l, fps=60, figsize=(6,6), title='Pendulum free')
        a1p1 = animate_pendulum(x, eu, t[1]-t[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        a12 = animate_costs(np.array(costs), labels=labels, figsize=(6,4), logscale=False)
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a13 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        a1f1 = animate_double_pendulum(x_free, 0*eu, t[1]-t[0], l1, l2, fps=60, figsize=(6,6), title='Double Pendulum free')
        a1p1 = animate_double_pendulum(x, eu, t[1]-t[0], l1, l2, fps=60, figsize=(6,6), title='Double Pendulum')
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
        ap1 = animate_pendulum(x, eu, ts[1]-ts[0], l, fps=60, figsize=(6,6), title='Pendulum')
    if DP:
        to_plot = np.array([xs[:,:,0], xs[:,:,1], xs[:,:,2], xs[:,:,3], us, Ts, Vs])
        a3 = general_multiplot_anim(to_plot, to, ['x1','x2','x3','x4','u','T','V'], fps=5, anim_time=30, figsize=(10,8))
        ap1 = animate_double_pendulum(x, eu, ts[1]-ts[0], l1, l2, fps=60, figsize=(6,6), title='Double Pendulum')
    print()
    return a3, ap1

def single_free_evolution():
    ''' Show the free evolution of the single pendulum'''
    #initial state: [angle, angular velocity]
    if SP: x0 = np.array([0.2, .1]) # [rad, rad/s] # SINGLE PENDULUM
    if DP: x0 = np.array([0.1, 0.1, 0, 0]) # [rad, rad/s, rad, rad/s] # DOUBLE PENDULUM
    # Time
    T = 4 # simulation time
    f = SIM_FREQ # frequency of the time steps

    t = np.linspace(0, T, int(T*f)) # time steps
    dt = t[1]-t[0] # time step
    print(f'T: {T}, f: {f}, dt: {dt:.4f}')

    sf11, sf12, sf13, sf14 = None, None, None, None
    eu = 0 + np.zeros(len(t)) # control input
    
    x = simulate(x0, t, eu) # simulate the pendulum

    tr = t[::-1] # reverse the time steps
    xr0 = x[-1] # final state
    xr = simulate(xr0, tr, eu) # reverse simulate the pendulum
    xrr = xr[::-1] # reverse the reverse simulation

    # XS, US = np.array([x,xrr]), np.array([eu,eu]) # states and control inputs
    # if SP: sf14 = animate_pendulums(XS,US,dt,l,60,(6,6))
    # if DP: sf14 = animate_double_pendulums(XS,US,dt,l1,l2,60,(6,6))

    # multiple reverse simulations from instability point
    NS = 50 # number of simulations
    # x0s = np.zeros_like(xr0) + normal(0, 0.001, (NS, len(xr0))) # initial states
    x0s = np.zeros_like(xr0) + normal(0, 0.5, (NS, len(xr0))) # initial states
    xrs = np.zeros((NS, len(tr), len(xr0))) # reverse states
    for i in tqdm(range(NS)): xrs[i] = simulate(x0s[i], tr, eu) # reverse simulate the pendulums
    Trs = np.array([kinetic_energy(x) for x in xrs]) # kinetic energy
    Vrs = np.array([potential_energy(x) for x in xrs]) # potential energy
    TVrs = Trs + Vrs # total energy

    x0sr = xrs[:,-1] # final states
    xrrs = np.zeros_like(xrs) # reverse reverse states
    for i in tqdm(range(NS)): xrrs[i] = simulate(x0sr[i], t, eu) # reverse reverse simulate the pendulums
    Trrs = np.array([kinetic_energy(x) for x in xrrs]) # kinetic energy
    Vrrs = np.array([potential_energy(x) for x in xrrs]) # potential energy
    TVrrs = Trrs + Vrrs # total energy

    #plot the kinetic and potential energies
    fig, ax = plt.subplots(3,2, figsize=(10,8))
    ax[0,0].plot(tr, Trs.T, alpha=0.5)
    ax[0,1].plot(t, Trrs.T, alpha=0.5)
    ax[0,0].set_title('Kinetic Energy')
    ax[1,0].plot(tr, Vrs.T, alpha=0.5)
    ax[1,1].plot(t, Vrrs.T, alpha=0.5)
    ax[1,0].set_title('Potential Energy')
    ax[2,0].plot(tr, TVrs.T, alpha=0.5)
    ax[2,1].plot(t, TVrrs.T, alpha=0.5)
    ax[2,0].set_title('Total Energy')

    xrsr = xrs[:,::-1] # reverse the reverse simulations
    XS, US = xrsr, np.zeros((NS, len(tr))) # states and control inputs
    print(f'XS: {XS.shape}, US: {US.shape}')
    if SP: sf13 = animate_pendulums(XS,US,dt,l,60,(6,6))
    if DP: sf13 = animate_double_pendulums(XS,US,dt,l1,l2,60,(6,6))

    XS = xrrs # states
    if SP: sf12 = animate_pendulums(XS,US,dt,l,60,(6,6))
    if DP: sf12 = animate_double_pendulums(XS,US,dt,l1,l2,60,(6,6))
    
    if SP:
        # plot trajecotries in state space
        # translate all the angles from [-π, π] to [0, 2π]
        xrs[:,:,0] = np.where(xrs[:,:,0]<0, xrs[:,:,0]+2*π, xrs[:,:,0])
        xrrs[:,:,0] = np.where(xrrs[:,:,0]<0, xrrs[:,:,0]+2*π, xrrs[:,:,0])
        sf15, ax = plt.subplots(1,1, figsize=(6,6))
        ax.plot(xrsr[:,:,0].T, xrsr[:,:,1].T, alpha=0.5)
        ax.plot(xrrs[:,:,0].T, xrrs[:,:,1].T, alpha=0.5)
        ax.set_xlabel('angle')
        ax.set_ylabel('angular velocity')
        ax.grid(True)
        ax.set_xticks(np.arange(0, 2*π+1, π/2))
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        sf16 = plot_state_trajectories(xrsr)
    else: sf15, sf16 = None, None


    return sf11, sf12, sf13, sf14, sf15, sf16



    # return sf11

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
            c = cost(np.array([xi]), np.array([0]), 0)
            Z[j,i] = c 

    #plot the cost function
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.set_zlabel('cost')
    return fig

def create_Q_table():
    def dist_angle(a1, a2):
        '''Calculate the distance between two angles'''
        return np.abs(np.arctan2(np.sin(a1-a2), np.cos(a1-a2))) # / π
    
    def dist_velocity(v1, v2):
        '''Calculate the distance between two velocities'''
        return np.abs(v1-v2)
    
    def is_outside(x):
        '''Check if the state is outside the grid
        split x into angles and velocities, angles are first half of x'''
        return np.any(x[N//2:] < VMIN-DGV/2) or np.any(x[N//2:] > VMAX+DGV/2)

    def get_xgrid(idxs):
        '''Get the state from the grid'''
        assert SP
        a, v = As[idxs[0]], Vs[idxs[1]]
        return np.array([a, v])

    def get_closest(x, idxs=None):
        assert SP
        da = dist_angle(x[0], As)
        dv = dist_velocity(x[1], Vs)
        ia, iv = np.argmin(da), np.argmin(dv)
        return (ia, iv), get_xgrid((ia, iv))
    
    def reach_next(x, xg, u, t=1.0):
        '''Reach the next state given the control input
        return (is_outside, is_stable, new_state, steps/SIM_FREQ)'''
        xu = x.copy() # current state
        for ss in range(int(t*SIM_FREQ)): # simulate the pendulum
            xu = step(xu, u, dt) # simulation step
            if is_outside(xu): return True, False, xu, ss/SIM_FREQ # out of bounds, break
            da = dist_angle(xu[:N//2], xg[:N//2])
            dv = dist_velocity(xu[N//2:], xg[N//2:])
            in_a = np.any(da < DGA/2) # angle close to current grid point
            in_v = np.any(dv < DGV/2) # velocity close to current grid point
            too_far_a = np.any(da > DGA) # too far in angle, we skipped a grid point
            too_far_v = np.any(dv > DGV) # too far in velocity, we skipped a grid point
            assert not too_far_a or not too_far_v, f'too far in angle and velocity: da:{da}, dv:{dv}, dga:{DGA}, dgv:{DGV}'
            if not in_a or not in_v: return False, False, xu, ss/SIM_FREQ # we arrived at a new grid point
        return False, True, xu, ss/SIM_FREQ # we are stable, no new states reached

    def reachable_states(x, us, t=1.0):  
        '''Get the reachable states from the current state
        return (reachable states from current state, time steps cost, indexes of the control inputs)'''
        reachable, costs, idxs = [],[],[] # reachable states and costs
        for iu, u in enumerate(us):
            is_outside, is_stable, nx, cu = reach_next(x, x, u, t)
            if is_outside or is_stable: continue
            reachable.append(nx), costs.append(cu), idxs.append(iu)
        return reachable, costs, idxs

    def get_best_input(x, us):
        reachable, _, uis = reachable_states(x, us) # reachable states
        xgis = [get_closest(x)[0] for x in reachable] # reachable grid states indexes
        costs = [Q[xgi] for xgi in xgis] # costs of the reachable states
        if len(costs) == 0: return None
        best_i = np.argmin(costs)
        return us[uis[best_i]]


    def plot_Q_stuff(Q, As, Vs, paths, bus, explored):
        if Q is not None:
            Q[np.isinf(Q)] = 0 + np.max(Q[~np.isinf(Q)]) # replace the inf values
            # plot the Q function
            Q = - Q # invert the Q function
            # from (12,19) to (19,12)
            Q = Q.T
            fig1 = plt.figure(figsize=(12,12))
            ax1 = fig1.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(As, Vs)
            assert Q.shape == X.shape == Y.shape, f'Q: {Q.shape}, X: {X.shape}, Y: {Y.shape}'
            ax1.plot_surface(X, Y, Q, cmap=cm.coolwarm)
            ax1.set_xlabel('angle')
            ax1.set_ylabel('angular velocity')
            ax1.set_zlabel('cost')
        else: fig1 = None

        if bus is not None:
            # plot the best control inputs
            bus = bus.T
            fig2 = plt.figure(figsize=(12,12))
            ax2 = fig2.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(As, Vs)
            ax2.plot_surface(X, Y, bus, cmap=cm.coolwarm)
            ax2.set_xlabel('angle')
            ax2.set_ylabel('angular velocity')
            ax2.set_zlabel('control input')
        else: fig2 = None

        # plot the paths
        if paths is not None:
            pats = [np.array(p).T for p in paths]
            print(f'paths[0]: {pats[0].shape}')
            fig0 = plot_state_trajectories(paths, figsize=(12,12))
        else: fig0 = None

        # plot the sequence of visited states
        if explored is not None:
            fig3, ax = plt.subplots(1,1, figsize=(12,12))
            xs = np.array(explored).T
            xs = xs[:,:10000]
            ax.plot(xs[0], xs[1], linewidth=1)
            ax.plot(xs[0][0], xs[1][0], 'go', markersize=2)
            ax.plot(xs[0][-1], xs[1][-1], 'ro', markersize=2)
            ax.set_xlabel('angle')
            ax.set_ylabel('angular velocity')
            ax.grid()
        else: fig3 = None
        return fig1, fig2, fig0, fig3

    def find_optimal_inputs(Q, Qe, As, Vs, us):
        #find the best inputs for each state
        bus = np.zeros_like(Q)
        for i, a in tqdm(enumerate(As)):
            for j, v in enumerate(Vs):
                if not Qe[i,j]: continue # not explored
                bus[i,j] = get_best_input(np.array([a,v]), us)
        return bus

    def generate_optimal_paths(bus, Qe):
        assert SP
        if np.sum(Qe) < 0.1*GP: return None
        # plot some optimal paths starting from some random states
        paths, pi = [], 0 # paths to ret, pi=path index
        while pi < 100: # generate 100 paths
            x = np.array([uniform(AMIN, AMAX), uniform(VMIN, VMAX)]) # random state
            xgi, xg = get_closest(x) # closest grid point
            if not Qe[xgi]: continue # not explored
            path = [x] # path
            for i in range(3*SIM_FREQ): # simulate the pendulum
                u = bus[xgi] # best control input
                x = step(x, u, -dt) # simulation step NOTE: positive time
                if is_outside(x): break 
                xgi, _ = get_closest(x) # closest grid point
                path.append(x)
            paths.append(path)
            pi += 1
            print(f'paths: {pi}/100    ', end='\r')
        print(f'generated {len(paths)} paths')
        return paths
    
    '''
    stuff to do:
    - create a tree of reachable states given the control inputs + plot it
    - DONE? solve the problem of the simulation steps not reaching the grid point
    - create ways of pruning the tree like: 
        - ignore already optimized states, somehow use depth to do it
        - use continuity constraint to ignore very different inputs
    - mix both the breadth search and the depth first method
    - consider only contiguos steps, skipping steps is teoretically wrong
    - consider applying simple reinforcement learning pipeline (no reverse time) first
    '''

    def explore_depth_first(Q, Qe, x0):
        
        explored, depths = [],[] #explored states and depths reached
        def explore_tree(x, depth, xc, idxs=None): # x: current state, depth: depth, xc: cost, idxs: grid indexes
            '''Explore the tree of states and control inputs'''
            if depth == MAX_DEPTH_DF: return # maximum depth reached
            if is_outside(x): return # out of bounds, return
            if len(explored) > MAX_VISITS: return # maximum number of visited states reached
            # get closest grid point
            xg_idx, xg = get_closest(x, idxs) # grid index
            # if Qe[xg_idx]: return # already explored
            Qe[xg_idx] = True # mark as explored
            #debug
            explored.append(xg), depths.append(depth) # save the explored states
            if len(explored) % 100 == 0: print(f'expl: {100*np.sum(Qe)/GP:.1f}%, vis: {len(explored)}, max depth: {max(depths)}, cost: {xc:.0f}    ', end='\r')
            # update the Q function
            if xc < Q[xg_idx]: Q[xg_idx] = xc # update the Q function
            else: return # no improvement, return
            # explore the tree
            c = Q[xg_idx] # current Q value
            reach, tcs, ius = reachable_states(x, us) # reachable states
            for nx, tc, iu in zip(reach, tcs, ius): # cycle through the reachable states
                csi = c + tc #+ np.abs(us[iu])/MAXU # cost of the state after tc steps
                explore_tree(nx, depth+1, csi) # explore the tree
            return 
        
        explore_tree(x0, 0, 0)
        return Q, Qe, explored
    
    def explore_breadth_firts(Q, Qe, x0):
        explored = [] # explored states
        curr_states, curr_costs = [x0], [0] # current states and costs
        for d in (range(MAX_DEPTH_BF)): 
            print(f'depth: {d}/{MAX_DEPTH_BF}, states: {len(curr_states)}, costs: {len(curr_costs)}, expl:{100*np.sum(Qe)/GP:.1f} %    ')
            next_states, next_costs = [], [] # next states and costs
            for x, c in zip(curr_states, curr_costs):
                explored.append(x) # save the explored states
                if len(explored) > MAX_VISITS: return Q, Qe, explored # maximum number of visited states reached
                xgi, xg = get_closest(x) # closest grid point
                Qe[xgi] = True # mark as explored
                if c < Q[xgi]: Q[xgi] = c # update the Q function
                else: continue # no improvement, continue
                reach, tcs, ius = reachable_states(x, us)
                for nx, tc, iu in zip(reach, tcs, ius):
                    csi = c + tc #+ np.abs(us[iu])/MAXU # cost of the state after tc steps
                    next_states.append(nx), next_costs.append(csi)
            curr_states, curr_costs = next_states, next_costs
        return Q, Qe, explored
    
    ########################################################################################################################
    ########################################################################################################################
    ### PARAMETERS #########################################################################################################
    AGRID = 12 # number of grid points angles
    VGRID = 11 # number of grid points velocities
    UGRID = 11 # number of grid points for the control inputs
    MAXV = 10 # [rad/s] maximum angular velocity
    MAXU = 0.1 # maximum control input

    AMIN, AMAX = -π, π-(2*π)/AGRID # minimum and maximum angles
    VMIN, VMAX = -MAXV, MAXV # minimum and maximum velocities
    if SP: N = 2 # number of states
    if DP: N = 4 # number of states
    GP = AGRID**(N//2)*VGRID**(N//2) # number of grid points
    MAX_DEPTH_DF = 400 # maximum depth of the tree for depth first
    MAX_DEPTH_BF = 7 # maximum depth of the tree for breadth first
    dt = - 1 / OPT_FREQ # time step ( NOTE: negative for exploring from the instability point )
    MAX_VISITS = 1e6 # number of states visited by the algorithm
    if dt > 0: print('Warning: dt is positive')

    As = np.linspace(AMIN, AMAX, AGRID) # angles
    Vs = np.linspace(VMIN, VMAX, VGRID) # velocities
    DGA = dist_angle(As[0], As[1]) # distance between grid points for the angles
    DGV = dist_velocity(Vs[0], Vs[1]) # distance between grid points for the velocities
    us = np.linspace(-MAXU, MAXU, UGRID) # control inputs   
    if SP: Q = np.ones((AGRID, VGRID)) * np.inf # Q function
    if DP: Q = np.ones((AGRID, AGRID, VGRID, VGRID)) * np.inf # Q function
    Qe = np.zeros_like(Q) # Q function explored
    print(f'Q shape: {Q.shape}, us shape: {us.shape}, GP: {GP}, MAX_DEPTH_DF: {MAX_DEPTH_DF}, MAX_DEPTH_BF: {MAX_DEPTH_BF}')
    
    ########################################################################################################################
    ########################################################################################################################
        
    x0 = np.array([0,0]) # initial state
    # depth first
    print('Depth first')
    Qb, Qeb = Q.copy(), Qe.copy()
    Qb, Qeb, explb = explore_depth_first(Qb, Qeb, x0)
    print(f'expl: {100*np.sum(Qeb)/GP:.1f}%, vis: {len(explb)}')
    # breadth first
    print('Breadth first')
    Qd, Qed = Q.copy(), Qe.copy()
    Qd, Qed, expld = explore_breadth_firts(Qd, Qed, x0)
    print(f'expl: {100*np.sum(Qed)/GP:.1f}%, vis: {len(expld)}')

    # find the optimal control inputs
    print('Optimal inputs')
    busb = find_optimal_inputs(Qb, Qeb, As, Vs, us)
    busd = find_optimal_inputs(Qd, Qed, As, Vs, us)

    # generate optimal paths
    print('Optimal paths')
    pathsb = generate_optimal_paths(busb, Qeb)
    pathsd = generate_optimal_paths(busd, Qed)

    # plot the results
    print('Plotting')
    figsb = plot_Q_stuff(Qb, As, Vs, pathsb, busb, explb)
    figsd = plot_Q_stuff(Qd, As, Vs, pathsd, busd, expld)

    return figsb, figsd

if __name__ == '__main__':
    os.system('clear')
    main_start = time()

    # pc = plot_cost_function()
    # sf = single_free_evolution()
    # t1 = test_1iter_mpc()
    # tm = test_mpc()
    q = create_Q_table()

    print(f'\nTotal time: {time()-main_start:.2f} s')
    plt.show()
    exit()