import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

plt.rcParams['figure.figsize'] = (15, 7)
os.makedirs('data', exist_ok=True)
np.set_printoptions(precision=4, suppress=True)

from casadi import SX, sin, cos, vertcat, solve as ca_solve, inv
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import shutil
if os.path.exists('c_generated_code'):
    shutil.rmtree('c_generated_code')

# ── Physical parameters ──────────────────────────────────────────────────────
m1   = 0.5    # [kg]  mass of link 1 (point mass at tip)
m2   = 0.5    # [kg]  mass of link 2
l1   = 0.5    # [m]   length of link 1
l2   = 0.5    # [m]   length of link 2
g    = 9.81   # [m/s²]

# ── MPC horizon ───────────────────────────────────────────────────────────────
N   = 100      # shooting nodes
Tf  = 2.0    # horizon [s]
dt  = Tf / N  # step size

# ── Simulation ────────────────────────────────────────────────────────────────
T_sim   = 6.0          # total sim time [s]
N_sim   = int(T_sim / dt)
u_max   = 3.0           # [Nm] torque limit

# ── Reference: both links pointing straight up ────────────────────────────────
# State convention:  x = [alpha, beta, dalpha, dbeta]
#   alpha : angle of link-1 measured from upward vertical  (0 = up, ±π = down)
#   beta  : angle of link-2 relative to link-1             (0 = both links aligned)
# Upright equilibrium: alpha=0, beta=0

x_ref = np.zeros(4)
u_ref = np.zeros(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DYNAMICS MODEL  (Lagrangian, explicit ODE)
# ─────────────────────────────────────────────────────────────────────────────

def export_double_pendulum_model() -> AcadosModel:
    """
    Double pendulum with torque actuator at the base joint.

    State:    x = [alpha, beta, dalpha, dbeta]
    Control:  u = [tau]   (torque at base joint only)

    Equations of motion from the Euler-Lagrange equations with
    absolute angle alpha for link-1 and relative angle beta for link-2:

        M(beta) * [alpha_ddot, beta_ddot]^T
            = tau_vec - C(beta, dalpha, dbeta) - G(alpha, beta)

    Mass matrix M:
        M11 = (m1+m2)*l1² + m2*l2² + 2*m2*l1*l2*cos(beta)
        M12 = M21 = m2*l2² + m2*l1*l2*cos(beta)
        M22 = m2*l2²

    Coriolis / centrifugal vector:
        C1 = -m2*l1*l2*sin(beta) * (2*dalpha*dbeta + dbeta²)
        C2 =  m2*l1*l2*sin(beta) * dalpha²

    Gravity vector:
        G1 = -(m1+m2)*g*l1*sin(alpha) - m2*g*l2*sin(alpha+beta)
        G2 = -m2*g*l2*sin(alpha+beta)

    Generalised forces:  tau_vec = [tau, 0]^T
    """
    model = AcadosModel()
    model.name = 'double_pendulum'

    # ── Symbolic variables ────────────────────────────────────────────────────
    alpha  = SX.sym('alpha')
    beta   = SX.sym('beta')
    dalpha = SX.sym('dalpha')
    dbeta  = SX.sym('dbeta')
    x = vertcat(alpha, beta, dalpha, dbeta)

    alpha_dot  = SX.sym('alpha_dot')
    beta_dot   = SX.sym('beta_dot')
    dalpha_dot = SX.sym('dalpha_dot')
    dbeta_dot  = SX.sym('dbeta_dot')
    xdot = vertcat(alpha_dot, beta_dot, dalpha_dot, dbeta_dot)

    tau = SX.sym('tau')
    u   = vertcat(tau)

    # ── Mass matrix M(beta) ───────────────────────────────────────────────────
    cb   = cos(beta)
    sb   = sin(beta)
    M11  = (m1 + m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*cb
    M12  = m2*l2**2 + m2*l1*l2*cb
    M22  = m2*l2**2

    # ── Coriolis / centrifugal ────────────────────────────────────────────────
    C1   = -m2*l1*l2*sb * (2*dalpha*dbeta + dbeta**2)
    C2   =  m2*l1*l2*sb * dalpha**2

    # ── Gravity ───────────────────────────────────────────────────────────────
    G1   = -(m1 + m2)*g*l1*sin(alpha) - m2*g*l2*sin(alpha + beta)
    G2   =  -m2*g*l2*sin(alpha + beta)

    # ── Solve M * qddot = rhs ─────────────────────────────────────────────────
    rhs1 = tau - C1 - G1
    rhs2 =     - C2 - G2

    # 2×2 inverse analytically (det = M11*M22 - M12²)
    det  = M11*M22 - M12**2
    alpha_ddot = ( M22*rhs1 - M12*rhs2) / det
    beta_ddot  = (-M12*rhs1 + M11*rhs2) / det

    # ── Explicit ODE ──────────────────────────────────────────────────────────
    f_expl = vertcat(dalpha, dbeta, alpha_ddot, beta_ddot)

    model.x          = x
    model.xdot       = xdot
    model.u          = u
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OCP SETUP
# ─────────────────────────────────────────────────────────────────────────────

def create_ocp(model: AcadosModel) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = model

    nx = 4
    nu = 1

    # ── Time horizon ──────────────────────────────────────────────────────────
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = Tf

    # ── Physics-based cost expressions ───────────────────────────────────────
    alpha, beta, dalpha, dbeta = (model.x[i] for i in range(4))
    tau = model.u[0]

    cb     = cos(beta)
    M11    = (m1 + m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*cb
    M12    = m2*l2**2 + m2*l1*l2*cb
    M22    = m2*l2**2

    # Potential energy relative to upright (= 0 at goal)
    V_expr = ((m1 + m2)*g*l1*(1 - cos(alpha))
              + m2*g*l2*(1 - cos(alpha + beta)))
    # Kinetic energy (= 0 at goal)
    T_expr = 0.5*(M11*dalpha**2 + 2*M12*dalpha*dbeta + M22*dbeta**2)

    model.cost_y_expr   = vertcat(V_expr, T_expr, alpha, beta, tau)
    model.cost_y_expr_e = vertcat(V_expr, T_expr, alpha, beta)

    # ── Cost: NONLINEAR_LS with physics-based output ──────────────────────────
    # cost = 0.5 * (w_V*V² + w_T*T² + w_a*alpha² + w_b*beta² + w_u*tau²)
    ny   = 5   # [V, T, alpha, beta, tau]
    ny_e = 4   # [V, T, alpha, beta]

    # w_V = 1e0   # weight on potential energy
    # w_T = 1e-1   # weight on kinetic energy
    # w_a = 2e1   # weight on alpha (link-1 angle error)
    # w_b = 1e1   # weight on beta  (link-2 relative angle error)
    # w_u = 1e-2  # weight on control effort

    w_V = 1e-1   # weight on potential energy
    w_T = 1e-1   # weight on kinetic energy
    w_a = 2e1   # weight on alpha (link-1 angle error)
    w_b = 1e1   # weight on beta  (link-2 relative angle error)
    w_u = 1e-2 # weight on control effort


    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    ocp.cost.W   = np.diag([w_V, w_T, w_a, w_b, w_u])
    ocp.cost.W_e = np.diag([w_V, w_T, w_a, w_b])

    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # ── Constraints ───────────────────────────────────────────────────────────
    # Initial state (will be updated every MPC step)
    ocp.constraints.x0 = np.zeros(nx)

    # Control bounds
    ocp.constraints.lbu   = np.array([-u_max])
    ocp.constraints.ubu   = np.array([ u_max])
    ocp.constraints.idxbu = np.array([0])

    # ── Solver options ────────────────────────────────────────────────────────
    ocp.solver_options.qp_solver        = 'PARTIAL_CONDENSING_HPIPM'
    # ocp.solver_options.hessian_approx   = 'EXACT' #'GAUSS_NEWTON'
    ocp.solver_options.hessian_approx   = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type  = 'IRK'
    ocp.solver_options.nlp_solver_type  = 'SQP'
    ocp.solver_options.globalization    = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 2
    ocp.solver_options.qp_solver_iter_max    = 100
    ocp.solver_options.print_level          = 0

    return ocp


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MPC SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_mpc():
    print("Building acados OCP solver …")
    model = export_double_pendulum_model()
    ocp   = create_ocp(model)
    solver = AcadosOcpSolver(ocp, json_file='double_pendulum_ocp.json')
    print("Solver built.\n")

    # ── Initial condition: hanging down with small perturbation ───────────────
    # alpha=π (down), beta=0 (aligned), small velocity kick
    x0 = np.array([np.pi - 0.1, 0.05, 0.0, 0.0])

    nx = 4
    nu = 1

    # Storage
    x_log = np.zeros((N_sim + 1, nx))
    u_log = np.zeros((N_sim,     nu))
    t_log = np.linspace(0, T_sim, N_sim + 1)
    status_log = np.zeros(N_sim, dtype=int)

    x_log[0] = x0

    # ── Warm-start: initialise solver trajectory ──────────────────────────────
    for k in range(N + 1):
        solver.set(k, 'x', x0)
    for k in range(N):
        solver.set(k, 'u', np.zeros(nu))

    x_cur = x0.copy()

    print(f"Running MPC for {N_sim} steps (T_sim={T_sim}s, dt={dt:.4f}s) …")
    for i in tqdm(range(N_sim)):
        # 1. Set current state as initial constraint
        solver.set(0, 'lbx', x_cur)
        solver.set(0, 'ubx', x_cur)

        # 2. Solve OCP
        status = solver.solve()
        status_log[i] = status

        if status not in (0, 2):
            print(f"  [step {i}] Solver returned status {status} – using previous u")

        # 3. Extract first control
        u_opt = solver.get(0, 'u')
        u_log[i] = u_opt

        # 4. Simulate one step with RK4 (using true dynamics)
        x_next = rk4_step(double_pendulum_ode, x_cur, u_opt, dt)
        x_cur  = x_next
        x_log[i + 1] = x_cur

        # 5. Shift warm-start by one step
        for k in range(N - 1):
            x_k1 = solver.get(k + 1, 'x')
            solver.set(k, 'x', x_k1)
        for k in range(N - 1):
            u_k1 = solver.get(k + 1, 'u')
            solver.set(k, 'u', u_k1)
        # Last node: keep terminal
        solver.set(N - 1, 'u', np.zeros(nu))

    solver.print_statistics()
    return t_log, x_log, u_log, status_log


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER: TRUE DYNAMICS (for RK4 simulation)
# ─────────────────────────────────────────────────────────────────────────────

def double_pendulum_ode(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    alpha, beta, dalpha, dbeta = x
    tau = u[0]

    cb  = np.cos(beta)
    sb  = np.sin(beta)
    sa  = np.sin(alpha)
    sab = np.sin(alpha + beta)

    M11 = (m1 + m2)*l1**2 + m2*l2**2 + 2*m2*l1*l2*cb
    M12 = m2*l2**2 + m2*l1*l2*cb
    M22 = m2*l2**2

    C1 = -m2*l1*l2*sb * (2*dalpha*dbeta + dbeta**2)
    C2 =  m2*l1*l2*sb * dalpha**2

    G1 = -(m1 + m2)*g*l1*sa - m2*g*l2*sab
    G2 = -m2*g*l2*sab

    rhs1 = tau - C1 - G1
    rhs2 =     - C2 - G2

    det  = M11*M22 - M12**2
    alpha_ddot = ( M22*rhs1 - M12*rhs2) / det
    beta_ddot  = (-M12*rhs1 + M11*rhs2) / det

    return np.array([dalpha, dbeta, alpha_ddot, beta_ddot])


def rk4_step(f, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
    k1 = f(x,             u)
    k2 = f(x + h/2 * k1, u)
    k3 = f(x + h/2 * k2, u)
    k4 = f(x + h   * k3, u)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(t_log, x_log, u_log, status_log):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Double Pendulum NMPC – Upright Stabilisation', fontsize=14)

    # ── Angles ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_log, np.degrees(x_log[:, 0]), label='α (link-1 from vertical)', color='tab:blue')
    ax.plot(t_log, np.degrees(x_log[:, 1]), label='β (link-2 relative)',       color='tab:orange')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, label='target (0°)')
    ax.set_ylabel('Angle [deg]')
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Angular velocities ────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_log, np.degrees(x_log[:, 2]), label='dα/dt', color='tab:blue')
    ax.plot(t_log, np.degrees(x_log[:, 3]), label='dβ/dt', color='tab:orange')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Angular velocity [deg/s]')
    ax.legend(fontsize=9)
    ax.grid(True)

    # ── Control ───────────────────────────────────────────────────────────────
    ax = axes[2]
    t_u = t_log[:-1]
    ax.step(t_u, u_log[:, 0], where='post', color='tab:red', label='τ (torque)')
    ax.axhline( u_max, color='k', linestyle=':', linewidth=0.8)
    ax.axhline(-u_max, color='k', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Torque [Nm]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=9)
    ax.grid(True)

    # Mark solver failures
    fail_t = t_u[status_log > 0]
    if len(fail_t):
        for a in axes:
            for ft in fail_t:
                a.axvline(ft, color='gray', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig('data/mpc_result.png', dpi=150)
    plt.show(block=False)
    print("Plot saved to data/mpc_result.png")

    # ── Phase portrait ────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    ax2.plot(np.degrees(x_log[:, 0]), np.degrees(x_log[:, 1]), 'b-', linewidth=0.8)
    ax2.plot(np.degrees(x_log[0, 0]), np.degrees(x_log[0, 1]), 'go', markersize=8, label='start')
    ax2.plot(0, 0, 'r*', markersize=12, label='target')
    ax2.set_xlabel('α [deg]')
    ax2.set_ylabel('β [deg]')
    ax2.set_title('Phase portrait  (α vs β)')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig('data/mpc_phase.png', dpi=150)
    # plt.show()
    print("Phase plot saved to data/mpc_phase.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ANIMATION
# ─────────────────────────────────────────────────────────────────────────────

def animate_pendulum(t_log, x_log, speedup: float = 1.0, fps: int = 30):
    """
    Animate the double pendulum trajectory.
    Saves to data/mpc_animation.mp4 and displays interactively.

    alpha = angle of link-1 from upward vertical
    beta  = angle of link-2 relative to link-1
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.patches as mpatches

    # Downsample to target fps
    total_frames = int(T_sim * fps / speedup)
    indices = np.linspace(0, len(t_log) - 1, total_frames, dtype=int)

    # Pre-compute Cartesian positions
    alpha = x_log[indices, 0]
    beta  = x_log[indices, 1]

    # link-1 tip
    x1 =  l1 * np.sin(alpha)
    y1 =  l1 * np.cos(alpha)
    # link-2 tip (absolute angle = alpha + beta)
    x2 = x1 + l2 * np.sin(alpha + beta)
    y2 = y1 + l2 * np.cos(alpha + beta)

    L = l1 + l2
    margin = 0.15

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_xlim(-L - margin, L + margin)
    ax.set_ylim(-L - margin, L + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Double Pendulum NMPC')
    ax.grid(True, alpha=0.3)

    # Draw upright target faintly
    ax.plot([0, 0], [0,  l1],       color='tab:blue',   alpha=0.15, linewidth=6, solid_capstyle='round')
    ax.plot([0, 0], [l1, l1 + l2],  color='tab:orange', alpha=0.15, linewidth=6, solid_capstyle='round')
    ax.plot(0, 0, 'ko', markersize=7, zorder=5)  # pivot

    # Animated elements
    line1,  = ax.plot([], [], 'o-',  color='tab:blue',   linewidth=3,  markersize=8,  solid_capstyle='round', label='link-1')
    line2,  = ax.plot([], [], 'o-',  color='tab:orange', linewidth=3,  markersize=8,  solid_capstyle='round', label='link-2')
    tip_dot, = ax.plot([], [], 'o',  color='tab:red',    markersize=10, zorder=6)
    trace,  = ax.plot([], [], '-',   color='tab:red',    alpha=0.4,    linewidth=0.8)
    time_txt = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9, va='top')

    # Trace history (last 1 s worth of frames)
    trace_len = fps

    ax.legend(loc='lower right', fontsize=8)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        tip_dot.set_data([], [])
        trace.set_data([], [])
        time_txt.set_text('')
        return line1, line2, tip_dot, trace, time_txt

    def update(frame):
        # link-1: pivot → joint
        line1.set_data([0, x1[frame]], [0, y1[frame]])
        # link-2: joint → tip
        line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        tip_dot.set_data([x2[frame]], [y2[frame]])
        # trace of tip
        start = max(0, frame - trace_len)
        trace.set_data(x2[start:frame+1], y2[start:frame+1])
        time_txt.set_text(f't = {t_log[indices[frame]]:.2f} s')
        return line1, line2, tip_dot, trace, time_txt

    interval_ms = 1000 / fps  # ms between frames (real-time at given speedup)
    anim = FuncAnimation(fig, update, frames=total_frames,
                         init_func=init, blit=True, interval=interval_ms)

    # Save to file
    save_path = 'data/mpc_animation.mp4'
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'Double Pendulum NMPC'}, bitrate=1800)
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    except Exception as e:
        print(f"Could not save mp4 ({e}); trying gif …")
        anim.save('data/mpc_animation.gif', fps=fps)
        print("Animation saved to data/mpc_animation.gif")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_log, x_log, u_log, status_log = run_mpc()

    # Summary
    final_err = np.abs(x_log[-1, :2])
    print(f"\nFinal angular errors:  α={np.degrees(final_err[0]):.3f}°,  β={np.degrees(final_err[1]):.3f}°")
    failures = int(np.sum(status_log > 0))
    print(f"Solver failures: {failures}/{len(status_log)}")

    np.save('data/t_log.npy',      t_log)
    np.save('data/x_log.npy',      x_log)
    np.save('data/u_log.npy',      u_log)
    np.save('data/status_log.npy', status_log)

    plot_results(t_log, x_log, u_log, status_log)
    animate_pendulum(t_log, x_log)
