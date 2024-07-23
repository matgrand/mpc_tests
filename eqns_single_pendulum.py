from numpy import sin, cos

def f_ddθ1(θ1, dθ1, u):
    return (9.81*sin(θ1) - 0.7*dθ1 + u)/1

def kinetic_energy(x):
    return 0.5*m1*l1**2*x.T[1]**2

def potential_energy(x):
    return m1*g*l1*(1 - cos(x.T[0]))

l1 = 1
g = 9.81
μ1 = 0.7
m1 = 1
