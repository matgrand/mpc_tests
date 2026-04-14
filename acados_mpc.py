import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

plt.rcParams['figure.figsize'] = (15, 7)
os.makedirs('data', exist_ok=True)
np.set_printoptions(precision=4, suppress=True)

from casadi import SX, sin, cos, vertcat
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import shutil
if os.path.exists('c_generated_code'):
    shutil.rmtree('c_generated_code')
