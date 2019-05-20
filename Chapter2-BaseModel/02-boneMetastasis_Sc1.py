# Metastasis Model
# Scenario 1: Periodicity 1
#
# Ariel Camacho
# Doctorate Thesis
# Guanajuato, Mexico, 2019

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import warnings
import PyDSTool
import sympy as sp

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',
       r'\sisetup{detect-all}',
       r'\usepackage{helvet}',
       r'\usepackage{sansmath}',
       r'\sansmath'
]


# parameters

#-- bone remodeling
a1 = 0.3
a2 = 0.1
b1 = 0.2
b2 = 0.02
g1 = -0.3
g2 = 0.5
#k1 = 0.07
#k2 = 0.0022

#-- bone metastasis
K = 300.0

a3 = 0.045
b3 = 0.05
s1 = 0.001
s2 = -0.00005
s3 = 0.005
s4 = 0.0
k1 = 0.07
k2 = 0.0022
y0 = np.array([10.0, 5.0, 20.0, 95.0])

# equations

x1eq = (b2/a2)**(1.0/g2)
x2eq = (b1/a1)**(1.0/g1)

def model(t, y):
    x1 = y[0]
    x2 = y[1]
    x3 = y[2]
    z  = y[3]

    dx1 = a1*x1*x2**g1 - b1*x1 + s1*x1*x3
    dx2 = a2*x2*x1**g2 - b2*x2 + s2*x2*x3
    dx3 = a3*x3*(1.0 - x3/K) - b3*x3 + s3*x1**g2*x3 + s4*x2**g1*x3
    dz  = -k1*np.sqrt(np.max([0.0, x1 - x1eq])) + k2*np.sqrt(np.max([0.0, x2 - x2eq]))

    return np.array([dx1, dx2, dx3, dz])

backend = 'vode'
#backend = 'dopri5'
#backend = 'lsoda'

t0 = 0.0
t1 = 2000.0

solver = ode(model).set_integrator(backend, atol=10**-8, rtol=10**-8,nsteps=100000,method='bdf')
solver.set_initial_value(y0, t0).set_f_params()
# suppress Fortran-printed warning
solver._integrator.iwork[2] = -1

sol = []
warnings.filterwarnings("ignore", category=UserWarning)

while solver.t < t1:
    solver.integrate(t1, step=True)
    sol.append([solver.t, solver.y[0], solver.y[1], solver.y[2], solver.y[3]])
    if not solver.successful():
        print('Woops: not success')
        break

warnings.resetwarnings()
sol2 = np.array(sol)

#%%
plt.close('all')
plt.figure(figsize=(8,5))

plt.subplot(2,2,1)
plt.plot(sol2[:,0], sol2[:,1], 'b', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Osteoclast $u$",fontsize=18)
plt.xlabel(r"Time $t$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.subplot(2,2,2)
plt.plot(sol2[:,0], sol2[:,2], 'g', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Osteoblast $v$",fontsize=18)
plt.xlabel(r"Time $t$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.subplot(2,2,3)
plt.plot(sol2[:,0], sol2[:,3], 'r', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Cancer Cells $w$",fontsize=18)
plt.xlabel(r"Time",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

plt.subplot(2,2,4)
plt.plot(sol2[:,0], sol2[:,4], 'k', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Bone Mass $z$",fontsize=18)
plt.xlabel(r"Time $t$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([t0,t1])
plt.tight_layout()
