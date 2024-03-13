import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
WAIT_S = 1 # wait time in seconds
INTERVAL = 500 # interval in milliseconds (1000 = real time)
C = (155/255,0,20/255) # unipd RGB


def animate_pendulum(x, dt, fps, l, figsize=(6,6)):
    # animate the system
    x = x[::int(1/fps/dt)] # display one frame every n time steps
    x = np.concatenate([np.array([x[0]]*int(WAIT_S*fps)), x]) 
    #create a new figure
    fig, ax = plt.subplots(figsize=figsize)
    lim = 1.1*l
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.grid(True)
    # ax.set_xlabel('x [m]'), ax.set_ylabel('y [m]')
    line, = ax.plot([], [], 'o-', lw=2, color=C)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    def animate(i):
        xi, yi = l*np.sin(x[i,0]), l*np.cos(x[i,0])
        line.set_data([0, xi], [0, yi])
        time_text.set_text(time_template % (-WAIT_S+i/fps))
        return line, time_text
    anim = animation.FuncAnimation(fig, animate, range(0, len(x)), init_func=init, blit=True, interval=INTERVAL/fps)
    plt.tight_layout()
    return anim

def animate_cart_double(x, dt, fps, l1, l2, figsize=(6,6)):
    # animate the system
    x = x[::int(1/fps/dt)] # display one frame every n time steps
    x = np.concatenate([np.array([x[0]]*int(WAIT_S*fps)), x]) 
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
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line1, line2, time_text
    def animate(i):
        x1, y1 = x[i,0] + l1*np.sin(x[i,1]), l1*np.cos(x[i,1])
        x2, y2 = x1 + l2*np.sin(x[i,2]), y1 + l2*np.cos(x[i,2])
        line1.set_data([x[i,0], x1], [0, y1])
        line2.set_data([x1, x2], [y1, y2])
        time_text.set_text(time_template % (-WAIT_S+i/fps))
        return line1, line2, time_text
    anim = animation.FuncAnimation(fig, animate, range(0, len(x)), init_func=init, blit=True, interval=INTERVAL/fps)
    plt.tight_layout()
    return anim