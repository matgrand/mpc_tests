import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu 
# from mpc import simulate
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
OPT_FREQ = 5*60 # frequency of the time steps optimization
SIM_FREQ = 10*OPT_FREQ # frequency of the time steps simulation
assert SIM_FREQ % OPT_FREQ == 0 # for more readable code

# different imports for the different models
if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True
if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *

########################################################################################################################
### PARAMETERS #########################################################################################################
########################################################################################################################
AGRID = 161 #161 # number of grid points angles 24
VGRID = int(1.2*AGRID)+1 # number of grid points velocities 25
UGRID = 13 # number of grid points for the control inputs
UCONTR = 66 # density of the input for control
MAXV = 8 # [rad/s] maximum angular velocity
MAXU = 5 # maximum control input

ALWAYS_RECALCULATE = True # always recalculate the Q function

if SP: N = 2 # number of states
if DP: N = 4 # number of states
GP = AGRID**(N//2)*VGRID**(N//2) # number of grid points
MAX_DEPTH_DF = 400 # maximum depth of the tree for depth first
MAX_DEPTH_BF = 100 # maximum depth of the tree for breadth first
NAIVE_DEPTH = 240 # maximum depth of the tree for naive exploration
DT = - 1 / OPT_FREQ # time step ( NOTE: negative for exploring from the instability point )
MAX_VISITS = 3e6 # number of states visited by the algorithm
COHERENT_INPUS = False # use coeherent inputs
if DT > 0: print('Warning: DT is positive')

# function to simulate a run
def simulate(x0, t, eu):
    '''Simulate the pendulum'''
    n, l, dt = len(t), len(x0), t[1]-t[0] # number of time steps, control inputs, time step
    x = np.zeros((n, l)) # [θ, dθ, ...] -> state vector
    x[0] = x0 # initial conditions
    for i in range(1, n): x[i] = step(x[i-1], eu[i], dt)   
    return x

def dist_angle(a1, a2):
    '''Calculate the distance between two angles'''
    if WRAP_AROUND: return np.abs(np.arctan2(np.sin(a1-a2), np.cos(a1-a2))) # / π
    else: np.abs(a1-a2) # / π

def dist_velocity(v1, v2):
    '''Calculate the distance between two velocities'''
    return np.abs(v1-v2)

def is_outside(x):
    '''Check if the state is outside the grid
    split x into angles and velocities, angles are first half of x'''
    return np.any(x[N//2:] < -MAXV-DGV/2) or np.any(x[N//2:] > MAXV+DGV/2)

def get_xgrid(idxs):
    '''Get the state from the grid'''
    assert SP
    a, v = As[idxs[0]], Vs[idxs[1]]
    return np.array([a, v])

def get_closest(x, n=1): # ret (idxs, xgrid)
    '''Get the closest n grid points to the state x'''
    assert SP
    da = dist_angle(x[0], As)
    dv = dist_velocity(x[1], Vs)
    idxs = []
    for _ in range(n):
        ia, iv = np.argmin(da), np.argmin(dv)
        da[ia], dv[iv] = np.inf, np.inf
        idxs.append((ia, iv))
    if n ==1: return idxs[0], get_xgrid(idxs[0])
    else: return idxs, [get_xgrid(idx) for idx in idxs]

def reach_next(x, xg, u, t=0.003, dt=DT):
    '''Reach the next state given the control input
    @return (is_outside, is_stable, new_state, steps/SIM_FREQ)'''
    xu = x.copy() # current state
    for ss in range(int(t*SIM_FREQ)): # simulate the pendulum
        xu = step(xu, u, DT) # simulation step
        if is_outside(xu): return True, False, xu, ss/SIM_FREQ # out of bounds, break
        da = dist_angle(xu[:N//2], xg[:N//2])
        dv = dist_velocity(xu[N//2:], xg[N//2:])
        in_a = np.any(da < DGA/2) # angle close to current grid point
        in_v = np.any(dv < DGV/2) # velocity close to current grid point
        too_far_a = np.any(da > DGA) # too far in angle, we skipped a grid point
        too_far_v = np.any(dv > DGV) # too far in velocity, we skipped a grid point
        assert not too_far_a or not too_far_v, f'too far in angle and velocity: da:{da}, dv:{dv}, dga:{DGA}, dgv:{DGV}'
        if not in_a or not in_v: return False, False, xu, ss/SIM_FREQ # we arrived at a new grid point
    print('Warning: simulation did not reach the grid point')
    return False, True, xu, ss/SIM_FREQ # we are stable, no new states reached

def reachable_states(x, us, iu=None, t=0.003, dt=DT):  
    '''Get the reachable states from the current state
    @return (reachable states from current state, time steps cost, indexes of the control inputs)'''
    reachable, costs, idxs = [],[],[] # reachable states and costs
    if iu is not None: ius = get_coeherent_input_idxs(iu, us) # coeherent inputs
    else: ius = range(len(us)) # all inputs
    for i in ius:
        u = us[i] # control input
        is_outside, is_stable, nx, cu = reach_next(x, x, u, t, dt)
        if is_outside or is_stable: continue
        reachable.append(nx), costs.append(cu), idxs.append(i)
    return reachable, costs, idxs

def get_best_input(Q, x, us, dt=DT):
    reachable, uc, uis = reachable_states(x, us, dt) # reachable states
    # print(f'reachable: {reachable}')
    xgis = [get_closest(x)[0] for x in reachable] # reachable grid states indexes
    #check if all the xgis are the same
    if len(xgis) == 0: return None
    costs = [Q[xgi] for xgi in xgis] # costs of the reachable states
    # add th cost of the control input
    costs = np.array(costs) + np.abs(us[uis])/MAXU
    best_i = np.argmin(costs)
    return us[uis[best_i]]

def get_coeherent_input_idxs(ui, us, dist=1):
    '''Get the coeherent inputs from the given input
    coeherent meaning if we are pushing we keep pushing, if we are pulling we keep pulling'''
    if COHERENT_INPUS: return [ui + i for i in range(-dist, dist+1) if 0 <= ui + i < len(us)]
    else: return range(len(us))

def plot_Q_stuff(Q, As, Vs, paths_inputs, bus, explored):
    if Q is not None:
        Q[np.isinf(Q)] = 0 + np.max(Q[~np.isinf(Q)]) # replace the inf values
        # plot the Q function
        Q = - Q.T # invert the Q function
        #plot a color matrix
        fig1, ax1 = plt.subplots(1,1, figsize=(10,10))
        cax = ax1.matshow(Q, cmap=cm.coolwarm)
        fig1.colorbar(cax)
        #add the grid
        ax1.grid(True)
        ax1.set_xticks(np.arange(0, len(As), len(As)/8))
        ax1.set_xticklabels(['-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4'])
        print(np.arange(0, len(Vs), len(Vs)/8))
        ax1.set_yticks(np.arange(0, len(Vs), len(Vs)/8))
        ax1.set_yticklabels([f'{MAXV}', f'{0.75*MAXV}', f'{0.5*MAXV}', f'{0.25*MAXV}', '0', f'-{0.25*MAXV}', f'-{0.5*MAXV}', f'-{0.75*MAXV}'])
        ax1.set_xlabel('angle')
        ax1.set_ylabel('angular velocity')
        ax1.set_title('Q function')
    else: fig1 = None

    if Q is not None:
        # plot 3d 
        fig2 = plt.figure(figsize=(10,10))
        ax2 = fig2.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(As, Vs)
        ax2.plot_surface(X, Y, Q, cmap=cm.coolwarm)
        ax2.set_xlabel('angle')
        ax2.set_ylabel('angular velocity')
        ax2.set_zlabel('cost')
        ax2.set_title('Q function')
    else: fig2 = None

    if bus is not None:
        # plot the best control inputs
        bus = bus.T
        fig2 = plt.figure(figsize=(10,10))
        ax2 = fig2.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(As, Vs)
        ax2.plot_surface(X, Y, bus, cmap=cm.coolwarm)
        ax2.set_xlabel('angle')
        ax2.set_ylabel('angular velocity')
        ax2.set_zlabel('control input')
        ax2.set_title('best control input')
    else: fig2 = None

    # plot the paths
    if paths_inputs is not None:
        paths, inputs = paths_inputs
        fig0 = plot_state_trajectories(paths, (Q, As, Vs), figsize=(10,10), title='Optimal paths')
        paths = np.array([np.array(p) for p in paths])
        inputs = np.array([np.array(inp) for inp in inputs])
        print(f'paths: {paths.shape}, inputs: {inputs.shape}')
        anim0 = animate_pendulums(paths, inputs, -DT, l, fps=60, figsize=(10,10))
    else: fig0, anim0 = None, None

    # plot the sequence of visited states
    if explored is not None and False:
        fig3, ax = plt.subplots(1,1, figsize=(10,10))
        xs = np.array(explored).T
        xs = xs[:,:10000]
        ax.plot(xs[0], xs[1], linewidth=1)
        ax.plot(xs[0][0], xs[1][0], 'go', markersize=2)
        ax.plot(xs[0][-1], xs[1][-1], 'ro', markersize=2)
        ax.set_xlabel('angle')
        ax.set_ylabel('angular velocity')
        ax.set_title('explored states')
        ax.grid()
    else: fig3 = None
    return fig1, fig2, fig0, fig3, anim0

def find_optimal_inputs(Q, Qe, As, Vs, us):
    #find the best inputs for each state
    bus = np.zeros_like(Q)
    for i, a in enumerate(tqdm(As, desc='optimal inputs')):
        for j, v in enumerate(Vs):
            if not Qe[i,j]: continue # not explored
            bus[i,j] = get_best_input(Q, np.array([a,v]), us, -DT)
    return bus

def get_Q_val(Q, x):
    '''Get the value of the Q function for a given state
    Use the 4 closest grid points to interpolate the Q function'''
    xg_idx, xg = get_closest(x, 4)
    Qvals = [Q[xgi] for xgi in xg_idx]
    # return np.mean(Qvals)
    # return np.min(Qvals)
    # return np.max(Qvals)
    xg = np.array(xg)
    das = dist_angle(x[0], xg[:,0])
    dvs = dist_velocity(x[1], xg[:,1])
    was, wvs = das/np.sum(das), dvs/np.sum(dvs)
    ws = 0.1*was + 0.9*wvs
    return np.sum([w*Qv for w, Qv in zip(ws, Qvals)])

def generate_optimal_paths(Q, Qe, cus, control_freq=30.0, n_paths=100, length_seconds=10):
    assert SP and np.all(Qe), 'generate_optimal_paths only works with SP and all states explored'
   
    fig, ax = plt.subplots(figsize=(10,10))
    Qplot = (Q.copy()).astype(np.int32)
    print(f'Qmax: {np.max(Qplot)}, Qmin: {np.min(Qplot)}')
    #plot the Q function before the trajectories
    Qcolors = cm.coolwarm(np.linspace(1, 0, np.max(Qplot)+1))
    for ia, a in enumerate(As):
        for iv, v in enumerate(Vs):
            # ax.plot(a, v, 's', color=Qcolors[Qplot[ia, iv]], markersize=4)
            ax.plot(a, v, 'o', color=Qcolors[Qplot[ia, iv]], markersize=4)

    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    ax.set_title('Optimal control paths')
    ax.grid(True)
    ax.set_xticks(np.arange(-π, π+1, π/2))
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    # plot some optimal paths starting from some random states
    paths, inputs = [],[] # paths to ret, pi=path index
    # assert n_paths % 2 == 0, 'n_paths must be even'
    # x0s = [np.array([uniform(-π, π), uniform(-MAXV/4, MAXV/4)]) for _ in range(n_paths//2)] # random states
    # x0s = x0s + [-x0 for x0 in x0s] # add the symmetric states
    # assert len(x0s) == n_paths, f'len(x0s): {len(x0s)}, n_paths: {n_paths}'
    x0s = [np.array([-π/2, 0])]#, np.array([-π/2, 0])] # initial states
    for x0 in tqdm(x0s, desc='paths'):
        print(f'\n\nx0: {x0}')
        path, input = [x0],[0] # path
        t = np.linspace(0, 1/control_freq, int(OPT_FREQ/control_freq)) # time vector
        x = x0 # current state
        for _ in range(int(length_seconds*control_freq)): # simulate the pendulum
            nxs = [] # next states
            for u in cus:
                nx = simulate(x, t, np.ones_like(t)*u)

                θs, dθs = nx.T #plotting
                interruptions = np.where(np.abs(np.diff(θs)) > π/2)[0]
                θs = np.split(θs, interruptions+1)
                dθs = np.split(dθs, interruptions+1)
                for θ, dθ in zip(θs, dθs):
                    # ax.plot(θ, dθ, ':', lw=1, color=cm.viridis(u/MAXU))
                    ax.plot(θ, dθ, ':', lw=1, color=cm.coolwarm(u/MAXU))

                nxs.append(nx[-1]) # last state
            nxs = np.array(nxs)
            # best_u_idx = np.argmin([Q[get_closest(nx)[0]] for nx in nxs])
            best_u_idx = np.argmin([get_Q_val(Q, nx) for nx in nxs])
            print(f'best_u_idx: {best_u_idx}, best_u: {cus[best_u_idx]}')
            best_u = cus[best_u_idx]*np.ones_like(t)
            bnx = simulate(x, t, best_u) # simulate the best control input
            path.extend(bnx), input.extend(best_u)
            x = bnx[-1] # update the current state
        #reassemble the path and input
        path, input = np.array(path), np.array(input)

        #plot the final path in black
        #split the path in parts
        interruptions = np.where(np.abs(np.diff(path[:,0])) > π/2)[0]
        ppaths = np.split(path, interruptions+1)
        for p in ppaths:
            ax.plot(p[:,0], p[:,1], 'k', lw=1)

        paths.append(path), inputs.append(input)
    print(f'generated {len(paths)} paths')
    return paths, inputs, fig

'''
stuff to do:
- a lot of the states created fall into very few grid states
- create a tree of reachable states given the control inputs + plot it
- DONE? solve the problem of the simulation steps not reaching the grid point
- create ways of pruning the tree like: 
    - ignore already optimized states, somehow use depth to do it
    - use continuity constraint to ignore very different inputs
- mix both the breadth search and the depth first method
- consider only contiguos steps, skipping steps is teoretically wrong?
- consider applying simple reinforcement learning pipeline (no reverse time) first
- create a smart grid using simulation to find the best grid points
'''

def explore_depth_first(Q, Qe, x0):
    explored, depths = [],[] #explored states and depths reached
    def explore_tree(x, depth, xc, iu, idxs=None): # x: current state, depth: depth, xc: cost, idxs: grid indexes
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
        reach, tcs, ius = reachable_states(x, US, iu) # reachable states
        for nx, tc, iu in zip(reach, tcs, ius): # cycle through the reachable states
            csi = c + tc #+ np.abs(us[iu])/MAXU # cost of the state after tc steps
            explore_tree(nx, depth+1, csi, iu) # explore the tree
        return 
    explore_tree(x0, 0, 0, iu=len(US)//2)
    return Q, Qe, explored

def explore_breadth_firts(Q, Qe, x0):
    explored = [] # explored states
    curr_states, curr_costs, curr_ius = [x0], [0], [len(US)//2] # current states and costs
    for d in (range(MAX_DEPTH_BF)): 
        print(f'depth: {d}/{MAX_DEPTH_BF}, states: {len(curr_states)}, costs: {len(curr_costs)}, expl:{100*np.sum(Qe)/GP:.1f} %    ')
        next_states, next_costs, next_ius = [], [], [] # next state, costs and input indexes
        for x, c, iu in zip(curr_states, curr_costs, curr_ius):
            explored.append(x) # save the explored states
            if len(explored) > MAX_VISITS: return Q, Qe, explored # maximum number of visited states reached
            xgi, xg = get_closest(x) # closest grid point
            Qe[xgi] = True # mark as explored
            if c < Q[xgi]: Q[xgi] = c # update the Q function
            else: continue # no improvement, continue
            reach, tcs, ius = reachable_states(x, US, iu)
            for nx, tc, iu in zip(reach, tcs, ius):
                csi = c + tc #+ np.abs(us[iu])/MAXU # cost of the state after tc steps
                next_states.append(nx), next_costs.append(csi), next_ius.append(iu)
        curr_states, curr_costs, curr_ius = next_states, next_costs, next_ius
    return Q, Qe, explored

def naive_explore(Q, Qe, x0):
    assert not COHERENT_INPUS, 'naive_explore does not work with COHERENT_INPUS'
    explored = [] # explored states
    curr_xgis, curr_xgs = [get_closest(x0)[0]], [get_closest(x0)[1]]
    for d in (range(NAIVE_DEPTH)): 
        if len(curr_xgis) == 0: break # no more states to explore
        print(f'depth: {d}/{NAIVE_DEPTH}, states: {len(curr_xgis)}    ', end='\r')
        next_xgis, next_xgs = [], []
        for xgi, xg in zip(curr_xgis, curr_xgs):
            if Qe[xgi]: continue # already visited, skip
            explored.append(xg) # save the explored states
            Qe[xgi] = True
            Q[xgi] = d+1 # temporary Q function
            reachable, cus, uis = reachable_states(xg, US) # reachable states from the current state, input costs, input indexes
            reachable_grid = [get_closest(r) for r in reachable] # reachable grid points
            for (nxgi, nxg), cu, i in zip(reachable_grid, cus, uis):
                if nxgi in next_xgis: continue # already in the next states
                next_xgis.append(nxgi), next_xgs.append(nxg)
        curr_xgis, curr_xgs = next_xgis, next_xgs
    return Q, Qe, explored

def fix_Q_edges(Q):
    ''' set the 'edges' of the Q function to the maximum value'''
    mask = np.zeros_like(Q, dtype=bool)
    for a in range(AGRID):
        for v in range(VGRID):
            a0, a1 = (a-1)%AGRID, (a+1)%AGRID
            v0, v1 = max(v-1, 0), min(v+1, VGRID-1)
            # to_check = [Q[a0,v], Q[a1,v], Q[a,v0], Q[a,v1]]
            to_check = [Q[a0,v0], Q[a0,v], Q[a0,v1], Q[a,v0], Q[a,v], Q[a,v1], Q[a1,v0], Q[a1,v], Q[a1,v1]]
            if np.max(to_check) - np.min(to_check) > 20: mask[a,v] = True
    Q[mask] = NAIVE_DEPTH + 1
    Q[mask] = 0
    return Q

def test_explore_space(): 
    print(f'US: {US}')
    # lets plot a graph of visitable nodes
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    #plot the grid with small black dots
    [ax.plot(a,v, 'ko', markersize=1) for a in As for v in Vs]
    x0 = np.array([0,0]) # initial state
    DEPTH = 230
    # define DEPTH random colors
    colors = plt.cm.viridis(np.linspace(1, 0, DEPTH))
    curr_states, curr_ius = [get_closest(x0)], [len(US)//2]
    Qes = np.zeros_like(Q) # temporary Q function
    Qese = np.zeros_like(Q) # visited states
    for d in (range(DEPTH)): 
        if len(curr_states) == 0: break # no more states to explore
        print(f'depth: {d}/{DEPTH}, states: {len(curr_states)}    ', end='\r')
        next_states, next_ius = [], []
        for (xgi, xg), iu in zip(curr_states, curr_ius):
            if Qese[xgi]: continue # already visited, skip
            Qese[xgi] = True
            Qes[xgi] = d+1 # temporary Q function
            #plot a point of the current state
            ax.plot(xg[0], xg[1], 'o', color=colors[d])
            reach, _, _ = reachable_states(xg, US, iu)
            reachable_grid = [get_closest(x) for x in reach]
            for nxgi, nxg in reachable_grid:
                next_states.append((nxgi, nxg)), next_ius.append(iu)
        curr_states, curr_ius = next_states, next_ius
    print()
    ax.grid(True)
    ax.set_xticks(np.arange(-π, π+1, π/2))
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    #set Qes to the maximum depth if Q is not Qese
    Qes = np.where(Qese, Qes, d+2)
    # calculate the optimal control inputs
    bus = find_optimal_inputs(Qes, Qese, As, Vs, US)
    # generate the optimal paths
    paths, inputs = generate_optimal_paths(bus, Qese)
    # plot the results
    fig, fig2, _, _ = plot_Q_stuff(Qes, As, Vs, paths, bus, None)

    #make all pathe the same length by adding the last state
    if paths is not None:
        max_len = max([len(p) for p in paths])
        for p in paths: 
            while len(p) < max_len: p.append(p[-1])
        for inp in inputs:
            while len(inp) < max_len: inp.append(inp[-1])
        paths = np.array([np.array(p) for p in paths])
        inputs = np.array([np.array(inp) for inp in inputs])
        #keep only the paths that have a last angle < π/
        paths_good = paths[np.where(np.abs(paths[:,-1,0]) < π/6)]
        if len(paths_good) > 0:
            paths = paths_good
            inputs = inputs[np.where(np.abs(paths[:,-1,0]) < π/6)]
        anim = animate_pendulums(paths, inputs, -DT, l, fps=60, figsize=(10,10))
        idx = np.random.randint(len(paths))
        path, input = paths[idx], inputs[idx]
        anim2 = animate_pendulum(path, input, -DT, l, figsize=(10,10))
    return fig, fig2, anim, anim2

def test_gridless():
    if not COHERENT_INPUS:
        print('\nFor test_gridless COHERENT_INPUS must be True, otherwise exponential growth too fast')
        return None
    from scipy.spatial import Delaunay, ConvexHull
    # explore states without a grid
    x0 = np.array([0,0]) # initial state
    small = 1e-6
    square = np.array([[-small,-small], [small,-small], [small,small], [-small,small]])
    print(f'US: {US}')
    # lets plot a graph of visitable nodes
    DEPTH = 8
    MAX_STATES = 1000
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    T = 0.1 #[s] time to simulate 
    t = np.linspace(-T, 0, int(T*SIM_FREQ))
    print(f't: {t}')
    # define DEPTH random colors
    colors = plt.cm.viridis(np.linspace(1, 0, DEPTH+1))
    curr_states, curr_uis = [x0], [len(US)//2]# current states, current control inputs indexes
    hull = ConvexHull([x+x0 for x in square]) # convex hull
    for d in (range(DEPTH)):
        print(f'depth: {d}/{DEPTH}, states: {len(curr_states)}    ')
        next_states, next_uis = [], []
        verts = hull.points[hull.vertices].reshape(-1, N) # vertices of the convex hull
        ax.add_patch(plt.Polygon(verts[:,:2], edgecolor=colors[d], fill=False))
        for x, ui in zip(curr_states, curr_uis):
            ax.plot(x[0], x[1], 'o', color=colors[d], markersize=2) #*(DEPTH-d))
            uis = get_coeherent_input_idxs(ui, US, dist=1) # coeherent inputs
            for i in uis: 
                u = US[i]
                nx = simulate(x, t, np.ones_like(t)*u)[-1]
                if Delaunay(verts).find_simplex(nx) > 0: continue  #check if nx is inside the convex hull
                next_states.append(nx), next_uis.append(i)
                # ax.plot([x[0], nx[0]], [x[1], nx[1]], '--', color=colors[d], linewidth=1)
        if len(next_states) > MAX_STATES: 
            idxs = np.random.choice(len(next_states), MAX_STATES, replace=False)
            next_states = [next_states[i] for i in idxs]
        new_verts = np.vstack([verts, next_states])
        hull = ConvexHull(new_verts) # convex hull
        curr_states, curr_uis = next_states, next_uis
    ax.set_xlabel('angle')
    ax.set_ylabel('angular velocity')
    # ax.set_xlim(-π, π)
    # ax.set_ylim(-MAXV, MAXV)
    # ax.set_xticks(np.arange(-π, π+1, π/2))
    # ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.grid(True)
    return fig

def test_explore_grid():
    print(f'US: {US}')
    for u in tqdm(US):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        [ax.plot(a,v, 'ko', markersize=1) for a in As for v in Vs] # plot the grid
        SAMPLES = 100
        for a in As:
            for v in Vs:
                das = uniform(-DGA/2, DGA/2, SAMPLES) # random angles
                dvs = uniform(-DGV/2, DGV/2, SAMPLES)
                nnas, nnvs = [],[]
                nas, nvs = [],[]
                for da, dv in zip(das, dvs):
                    na, nv = a+da, v+dv
                    na = (na + π) % (2*π) - π # wrap around
                    # ax.plot(na, nv, 'ro', markersize=1)
                    x0 = np.array([na, nv])
                    t = 0.1
                    t = np.linspace(-t, 0, int(t*OPT_FREQ))
                    nna, nnv = simulate(x0, t, u*np.ones_like(t))[-1]
                    if np.abs(nna-na) > π: continue
                    nnas.append(nna), nnvs.append(nnv), nas.append(na), nvs.append(nv)
                nnas, nnvs = np.array(nnas), np.array(nnvs)
                nas, nvs = np.array(nas), np.array(nvs)
                # get lengts of the paths
                # lens = np.sqrt((nnas-nas)**2 + (nnvs-nvs)**2)
                lens = np.sqrt((nnvs-nvs)**2)
                # plot the paths, use colors to show the length
                colors = plt.cm.viridis(lens/np.max(lens))
                for na, nv, nna, nnv, c in zip(nas, nvs, nnas, nnvs, colors):
                    # plot a dashed from na to nna
                    ax.plot([na, nna], [nv, nnv], '-', color=c, linewidth=1)
        ax.set_xlabel('angle')  
        ax.set_ylabel('angular velocity')
        ax.set_title(f'u: {u}')
        ax.grid(True)
        ax.set_xticks(np.linspace(-π, π, 9))
        ax.set_xticklabels(['-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])

    

    return fig

if __name__ == '__main__':
    os.system('clear')
    main_start = time()

    As = np.linspace(-π, π, AGRID, endpoint=False) # angles
    Vs = np.linspace(-MAXV, MAXV, VGRID) # velocities
    DGA = dist_angle(As[0], As[1]) # distance between grid points for the angles
    DGV = dist_velocity(Vs[0], Vs[1]) # distance between grid points for the velocities
    US = np.linspace(-1*MAXU, 1*MAXU, UGRID) # inputs   
    CUS = np.linspace(-1*MAXU, 1*MAXU, UCONTR) # control inputs
    if SP: Q = np.ones((AGRID, VGRID)) * np.inf # Q function
    if DP: Q = np.ones((AGRID, AGRID, VGRID, VGRID)) * np.inf # Q function
    Qe = np.zeros_like(Q) # Q function explored
    print(f'Q shape: {Q.shape}, US shape: {US.shape}, GP: {GP}')

    Q_name = f'Q_{M}_{AGRID}_{VGRID}_{UGRID}_{UCONTR}_{MAXV}_{MAXU}_{NAIVE_DEPTH}_{OPT_FREQ}_{COHERENT_INPUS}.npy'
    
    ########################################################################################################################
    ########################################################################################################################

    # f = test_explore_space()
    # f = test_gridless()
    # f = test_explore_grid()

    x0 = np.array([0,0]) # initial state

    # print('Depth first')
    # Q, Qe, expl = explore_depth_first(Q, Qe, x0)
    
    # print('Breadth first')
    # Qb, Qeb, explb = explore_breadth_firts(Q, Qe, x0)

    print('Naive')
    # check if Q already exists
    if os.path.exists(f'tmp/{Q_name}') and not ALWAYS_RECALCULATE: # load the Q function
        Q = np.load(f'tmp/{Q_name}')
        Qe = np.ones_like(Q)
        expl = []
        print(f'loaded Q from file: {Q_name}')
    else:
        Q, Qe, expl = naive_explore(Q, Qe, x0)
        np.save(f'tmp/{Q_name}', Q) # save the Q function

    # # check if the Q function is symmetric and fix it
    # if SP: 
    #     n_wrong = 0
    #     for a in range(AGRID):
    #         for v in range(VGRID):
    #             if not np.isclose(Q[a,v], Q[-a%AGRID,-v%VGRID], atol=1e-3): n_wrong += 1
    #     print(f'not symmetric: {n_wrong}/{AGRID*VGRID}')
    #     #force symmetry
    #     for a in range(AGRID):
    #         for v in range(VGRID):
    #             Q[-a%AGRID,-v%VGRID] = Q[a,v]
    #     n_wrong = 0
    #     for a in range(AGRID):
    #         for v in range(VGRID):
    #             if not np.isclose(Q[a,v], Q[-a%AGRID,-v%VGRID], atol=1e-3): n_wrong += 1
    #     print(f'not symmetric: {n_wrong}/{AGRID*VGRID}')

    # Q = fix_Q_edges(Q)

    # calculate the optimal control inputs
    # bus = find_optimal_inputs(Q, Qe, As, Vs, US)
    bus = None
    # generate the optimal paths
    paths, inputs, figpaths = generate_optimal_paths(Q, Qe, CUS, control_freq=5.0, n_paths=2, length_seconds=10)
    # plot the results

    plots = plot_Q_stuff(Q, As, Vs, (paths, inputs), bus, expl)


    print(f'\nTotal time: {time()-main_start:.2f} s')
    plt.show()
    exit()