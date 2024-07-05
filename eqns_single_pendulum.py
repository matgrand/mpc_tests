import numpy as np

def f_ddθ1(θ1, dθ1, u):
    return (9.81*np.sin(θ1) - 0.7*dθ1 + u)/1

l1 = 1
g = 9.81
μ1 = 0.7
m1 = 1
