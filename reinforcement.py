import numpy as np; π = np.pi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from plotting import *
# from inputs import addittive_resample as expu
from inputs import frequency_resample as expu 
from mpc import simulate
from numpy.random import uniform, normal
from time import time
import multiprocess as mp #note: not multiprocessing
import os
# np.set_printoptions(precision=3, suppress=True) #set numpy print options
# os.environ['KMP_DUPLICATE_LIB_OK']='True' # for the multiprocessing to work on MacOS
# np.random.seed(0)
 
SP, DP, CDP = 0, 1, 2 # single pendulum, double pendulum, cart double pendulum

# Choose the model
M = SP
OPT_FREQ = 2*60 # frequency of the time steps optimization
SIM_FREQ = 10*OPT_FREQ # frequency of the time steps simulation
assert SIM_FREQ % OPT_FREQ == 0 # for more readable code

if M == SP: SP, DP, CDP = True, False, False
elif M == DP: SP, DP, CDP = False, True, False
elif M == CDP: SP, DP, CDP = False, False, True
if SP: from single_pendulum import *
elif DP: from double_pendulum import *
elif CDP: from cart_double_pendulum import *


def create_Q_table():
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

    def get_closest(x, idxs=None): # ret (idxs, xgrid)
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
        return False, True, xu, ss/SIM_FREQ # we are stable, no new states reached

    def reachable_states(x, us, iu=None, t=1.0):  
        '''Get the reachable states from the current state
        return (reachable states from current state, time steps cost, indexes of the control inputs)'''
        reachable, costs, idxs = [],[],[] # reachable states and costs
        if iu is not None: ius = get_coeherent_input_idxs(iu, us) # coeherent inputs
        else: ius = range(len(us)) # all inputs
        for i in ius:
            u = us[i] # control input
            is_outside, is_stable, nx, cu = reach_next(x, x, u, t)
            if is_outside or is_stable: continue
            reachable.append(nx), costs.append(cu), idxs.append(i)
        return reachable, costs, idxs

    def get_best_input(x, us):
        reachable, _, uis = reachable_states(x, us) # reachable states
        xgis = [get_closest(x)[0] for x in reachable] # reachable grid states indexes
        costs = [Q[xgi] for xgi in xgis] # costs of the reachable states
        if len(costs) == 0: return None
        best_i = np.argmin(costs)
        return us[uis[best_i]]

    def get_coeherent_input_idxs(ui, us, dist=1):
        '''Get the coeherent inputs from the given input
        coeherent meaning if we are pushing we keep pushing, if we are pulling we keep pulling'''
        if COHERENT_INPUS: return [ui + i for i in range(-dist, dist+1) if 0 <= ui + i < len(us)]
        else: return range(len(us))

    def plot_Q_stuff(Q, As, Vs, paths, bus, explored):
        if Q is not None:
            Q[np.isinf(Q)] = 0 + np.max(Q[~np.isinf(Q)]) # replace the inf values
            # plot the Q function
            Q = - Q # invert the Q function
            # from (12,19) to (19,12)
            Q = Q.T
            fig1 = plt.figure(figsize=(10,10))
            ax1 = fig1.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(As, Vs)
            assert Q.shape == X.shape == Y.shape, f'Q: {Q.shape}, X: {X.shape}, Y: {Y.shape}'
            ax1.plot_surface(X, Y, Q, cmap=cm.coolwarm)
            ax1.set_xlabel('angle')
            ax1.set_ylabel('angular velocity')
            ax1.set_zlabel('cost')
            ax1.set_title('Q function')
        else: fig1 = None

        if bus is not None and False:
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
        if paths is not None:
            pats = [np.array(p).T for p in paths]
            print(f'paths[0]: {pats[0].shape}')
            fig0 = plot_state_trajectories(paths, figsize=(10,10), title='Optimal paths')
        else: fig0 = None

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
        return fig1, fig2, fig0, fig3

    def find_optimal_inputs(Q, Qe, As, Vs, us):
        #find the best inputs for each state
        bus = np.zeros_like(Q)
        for i, a in enumerate(tqdm(As)):
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
            x = np.array([uniform(-π, π), uniform(-MAXV, MAXV)]) # random state
            xgi, xg = get_closest(x) # closest grid point
            if not Qe[xgi]: continue # not explored
            path = [x] # path
            for i in range(3*SIM_FREQ): # simulate the pendulum
                u = bus[xgi] # best control input
                x = step(x, u, -DT) # simulation step NOTE: positive time
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
    
    ########################################################################################################################
    ### PARAMETERS #########################################################################################################
    ########################################################################################################################
    AGRID = 88 # number of grid points angles 24
    VGRID = AGRID+1 # number of grid points velocities 25
    UGRID = 5 # number of grid points for the control inputs
    MAXV = 10 # [rad/s] maximum angular velocity
    MAXU = 6 # maximum control input

    if SP: N = 2 # number of states
    if DP: N = 4 # number of states
    GP = AGRID**(N//2)*VGRID**(N//2) # number of grid points
    MAX_DEPTH_DF = 400 # maximum depth of the tree for depth first
    MAX_DEPTH_BF = 100 # maximum depth of the tree for breadth first
    DT = - 1 / OPT_FREQ # time step ( NOTE: negative for exploring from the instability point )
    MAX_VISITS = 3e6 # number of states visited by the algorithm
    COHERENT_INPUS = False # use coeherent inputs
    if DT > 0: print('Warning: DT is positive')

    As = np.linspace(-π, π, AGRID, endpoint=False) # angles
    Vs = np.linspace(-MAXV, MAXV, VGRID) # velocities
    DGA = dist_angle(As[0], As[1]) # distance between grid points for the angles
    DGV = dist_velocity(Vs[0], Vs[1]) # distance between grid points for the velocities
    US = np.linspace(-MAXU, MAXU, UGRID) # control inputs   
    if SP: Q = np.ones((AGRID, VGRID)) * np.inf # Q function
    if DP: Q = np.ones((AGRID, AGRID, VGRID, VGRID)) * np.inf # Q function
    Qe = np.zeros_like(Q) # Q function explored
    print(f'Q shape: {Q.shape}, US shape: {US.shape}, GP: {GP}, MAX_DEPTH_DF: {MAX_DEPTH_DF}, MAX_DEPTH_BF: {MAX_DEPTH_BF}')
    
    ########################################################################################################################
    ########################################################################################################################
    
    def test_explore_space(): 
        print(f'US: {US}')
        # lets plot a graph of visitable nodes
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        #plot the grid with small black dots
        [ax.plot(a,v, 'ko', markersize=1) for a in As for v in Vs]
        x0 = np.array([0,0]) # initial state
        DEPTH = 210
        # define DEPTH random colors
        colors = plt.cm.viridis(np.linspace(1, 0, DEPTH))
        curr_states, curr_ius = [get_closest(x0)], [len(US)//2]
        Qese = np.zeros_like(Q) # visited states
        Qes = np.zeros_like(Q) # temporary Q function
        for d in (range(DEPTH)): 
            if len(curr_states) == 0: break # no more states to explore
            print(f'depth: {d}/{DEPTH}, states: {len(curr_states)}    ')
            next_states, next_ius = [], []
            for (xgi, xg), iu in zip(curr_states, curr_ius):
                if Qese[xgi]: continue
                Qese[xgi] = True
                Qes[xgi] = d+1 # temporary Q function
                #plot a point of the current state
                x, y = xg
                ax.plot(x, y, 'o', color=colors[d])
                reach, _, _ = reachable_states(xg, US, iu)
                reach_grid = [get_closest(x) for x in reach]
                for nxgi, nxg in reach_grid:
                    next_states.append((nxgi, nxg)), next_ius.append(len(US)//2)
            curr_states, curr_ius = next_states, next_ius
        ax.grid(True)
        ax.set_xticks(np.arange(-π, π+1, π/2))
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        #set Qes to the maximum depth if Q is not Qese
        Qes = np.where(Qese, Qes, d+2)

        # calculate the optimal control inputs
        bus = find_optimal_inputs(Qes, Qese, As, Vs, US)
        # generate the optimal paths
        paths = generate_optimal_paths(bus, Qese)
        # plot the results
        fig, fig2, _, _ = plot_Q_stuff(Qes, As, Vs, paths, bus, None)

        return fig, fig2

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
                

    f = test_explore_space()
    # f = test_gridless()
    # f = test_explore_grid()

    # x0 = np.array([0,0]) # initial state
    # # depth first
    # print('Depth first')
    # Qb, Qeb = Q.copy(), Qe.copy()
    # Qb, Qeb, explb = explore_depth_first(Qb, Qeb, x0)
    # print(f'\nexpl: {100*np.sum(Qeb)/GP:.1f}%, vis: {len(explb)}')
    # # breadth first
    # print('Breadth first')
    # Qd, Qed = Q.copy(), Qe.copy()
    # Qd, Qed, expld = explore_breadth_firts(Qd, Qed, x0)
    # print(f'\nexpl: {100*np.sum(Qed)/GP:.1f}%, vis: {len(expld)}')

    # # find the optimal control inputs
    # print('Optimal inputs')
    # busb = find_optimal_inputs(Qb, Qeb, As, Vs, US)
    # busd = find_optimal_inputs(Qd, Qed, As, Vs, US)

    # # generate optimal paths
    # print('Optimal paths')
    # pathsb = generate_optimal_paths(busb, Qeb)
    # pathsd = generate_optimal_paths(busd, Qed)

    # # plot the results
    # print('Plotting')
    # figsb = plot_Q_stuff(Qb, As, Vs, pathsb, busb, explb)
    # figsd = plot_Q_stuff(Qd, As, Vs, pathsd, busd, expld)

    return None




if __name__ == '__main__':
    os.system('clear')
    main_start = time()

    create_Q_table()

    print(f'\nTotal time: {time()-main_start:.2f} s')
    plt.show()
    exit()