import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
WAIT_S = 0.5 # wait time in seconds
INTERVAL = 500 # interval in milliseconds (1000 = real time)
C = (155/255,0,20/255) # unipd RGB


def animate_pendulum(x, u, dt, l, fps=60, figsize=(6,6)):
    # animate the system
    x = x[::int(1/fps/dt)] # display one frame every n time steps
    u = u[::int(1/fps/dt)] # display one frame every n time steps
    sw = int(WAIT_S*fps) # sample to wait for
    x = np.concatenate([np.array([x[0]]*sw), x, np.array([x[-1]]*sw)]) if WAIT_S > 0 else x
    u = np.concatenate([np.array([u[0]]*sw), u, np.array([u[-1]]*sw)]) if WAIT_S > 0 else u
    maxu = max(np.max(np.abs(u)), 1e-3)
    u = l*u/maxu # scale the control input
    #create a new figure
    fig, ax = plt.subplots(figsize=figsize)
    lim = 1.1*l
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True)
    # ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
    line = ax.plot([], [], 'o-', lw=2, color='blue')[0]
    input = ax.plot([], [], '-', lw=3, color=C)[0]
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        line.set_data([], [])
        input.set_data([], [])
        time_text.set_text('')
        return line, input, time_text
    def animate(i):
        xi, yi = l*np.sin(x[i,0]), l*np.cos(x[i,0])
        line.set_data([0, xi], [0, yi])
        input.set_data([0, u[i]], [-0.95*lim, -0.95*lim])
        time_text.set_text(time_template % (-WAIT_S+i/fps))
        return line, input, time_text
    anim = animation.FuncAnimation(fig, animate, range(0, len(x)), init_func=init, blit=True, interval=INTERVAL/fps)
    plt.tight_layout()
    return anim

def animate_cart_double(x, u, dt, l1, l2, fps=60, figsize=(6,6)):
    # animate the system
    x = x[::int(1/fps/dt)] # display one frame every n time steps
    u = u[::int(1/fps/dt)] # display one frame every n time steps
    sw = int(WAIT_S*fps) # sample to wait for
    x = np.concatenate([np.array([x[0]]*sw), x, np.array([x[-1]]*sw)]) if WAIT_S > 0 else x
    u = np.concatenate([np.array([u[0]]*sw), u, np.array([u[-1]]*sw)]) if WAIT_S > 0 else u
    maxu = max(np.max(np.abs(u)), 1e-3)
    u = (l1+l2)*u/maxu # scale the control input
    #create a new figure
    fig, ax = plt.subplots(figsize=figsize)
    lim = 1.1*(l1+l2)
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True)
    # ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
    ax.plot([-lim,lim], [0,0], '-',  lw=1, color='black')[0]
    line2 = ax.plot([], [], 'o-', lw=5, color='red')[0]
    line1 = ax.plot([], [], 'o-', lw=5, color='blue')[0]
    input = ax.plot([], [], '-', lw=3, color=C)[0]
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        input.set_data([], [])
        time_text.set_text('')
        return line1, line2, input, time_text
    def animate(i):
        x1, y1 = x[i,0] + l1*np.sin(x[i,1]), l1*np.cos(x[i,1])
        x2, y2 = x1 + l2*np.sin(x[i,2]), y1 + l2*np.cos(x[i,2])
        line1.set_data([x[i,0], x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        input.set_data([0, u[i]], [-0.95*lim, -0.95*lim])
        time_text.set_text(time_template % (-WAIT_S+i/fps))
        return line1, line2, input, time_text
    anim = animation.FuncAnimation(fig, animate, range(0, len(x)), init_func=init, blit=True, interval=INTERVAL/fps)
    plt.tight_layout()
    return anim

def animate_costs(costs, labels, fps=60, anim_time=5, figsize=(8,6), logscale=False):
    ''' costs should be a vector of size (ncosts, iterations, time)'''
    assert costs.ndim == 3, f'costs.ndim: {costs.ndim}'
    skip = max(costs.shape[1]//int(fps*anim_time), 1)
    costs = costs[:, ::skip, :]
    ncosts, iters, nt = costs.shape
    assert len(labels) == ncosts, f'len(labels): {len(labels)}, ncosts: {ncosts}'

    t = np.linspace(0, 1, nt)
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_ylim(np.min(costs), np.max(costs))
    ax.set_xlim(0, 1)
    ax.grid(True)

    colors = plt.cm.viridis(np.linspace(0, 1, ncosts))
    lines = [ax.plot([], [], '-', lw=2, color=colors[i], label=labels[i])[0] for i in range(ncosts)]
    if logscale: ax.set_yscale('log'), ax.set_ylim(1e-3, np.max(costs))
    else: ax.set_ylim(0, np.max(costs)*1/5)
    ax.legend()

    #initialize figure by plotting the first costs and the last costs
    for i in range(ncosts):
        ax.plot(t, costs[i, 0, :], '--', lw=1, color=colors[i])
        ax.plot(t, costs[i, -1, :], '--', lw=1, color=colors[i])

    iter_template = 'iteration = %d /' + str(iters*skip)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        for line in lines: line.set_data([], [])
        time_text.set_text('')
        return lines + [time_text]
    def animate(i):
        for j, line in enumerate(lines):
            for k in range(0, i+1, 5):
                line.set_data(t, costs[j, k, :])
            # line.set_data(t, costs[j, i, :])
        time_text.set_text(iter_template % (i*skip))
        return lines + [time_text]
    
    anim = animation.FuncAnimation(fig, animate, range(iters), init_func=init, blit=True, interval=INTERVAL/fps)
    plt.tight_layout()
    return anim
    
def plot_single(x, t, u, T, V, figsize=(12,10)):
    # plot the state and energies
    fig, ax = plt.subplots(4, 1, figsize=figsize) #figsize=(18,12))
    ax[0].plot(t, x[:,0], label='θ, angle', color=C)
    ax[0].set_ylabel('Angle [rad]')
    ax[1].plot(t, x[:,1], label='dθ, angular velocity', color=C)
    ax[1].set_ylabel('Angular velocity [rad/s]')
    ax[2].plot(t, T, label='T, kinetic energy', color='red')
    ax[2].plot(t, V, label='V, potential energy', color='blue')
    ax[2].set_ylabel('Energy [J]')
    # ax[2].set_yscale('log')
    ax[2].legend()
    ax[2].plot(t, T+V, '--',label='T+V, total energy', color='black')
    ax[2].legend(), ax[2].grid(True)
    ax[3].plot(t, u, label='u, control input', color=C)
    ax[3].set_ylabel('Control input')
    ax[0].grid(True), ax[1].grid(True), ax[3].grid(True)
    plt.tight_layout()
