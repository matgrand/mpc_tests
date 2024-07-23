from numpy import sin, cos, tan, exp, log, sqrt, pi
def f_ddθ1(θ1, θ2, dθ1, dθ2, u):
    return (-0.5*u - 2.4525*sin(θ1 - 2*θ2) + 0.5*sin(θ1 - θ2)*dθ2**2 + 0.25*sin(2*θ1 - 2*θ2)*dθ1**2 - 7.3575*sin(θ1) - 0.35*cos(θ1 - θ2)*dθ2 + 0.35*dθ1)/(0.25*cos(2*θ1 - 2*θ2) - 0.75)

def f_ddθ2(θ1, θ2, dθ1, dθ2, u):
    return (0.5*u*cos(θ1 - θ2) - 1.0*sin(θ1 - θ2)*dθ1**2 - 0.25*sin(2*θ1 - 2*θ2)*dθ2**2 + 4.905*sin(2*θ1 - θ2) - 4.905*sin(θ2) - 0.35*cos(θ1 - θ2)*dθ1 + 0.7*dθ2)/(0.25*cos(2*θ1 - 2*θ2) - 0.75)

def kinetic_energy(θ1, θ2, dθ1, dθ2):
    return 1.0*cos(θ1 - θ2)*dθ1*dθ2 + 1.0*dθ1**2 + 0.5*dθ2**2

def potential_energy(θ1, θ2):
    return 19.62*cos(θ1) + 9.81*cos(θ2)

l1 = 1
l2 = 1
g = 9.81
μ1 = 0.7
μ2 = 0.7
m1 = 1
m2 = 1
