# Base Model
# Bone remodeling model with base parameters
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
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = True


# parameters

a1 = 0.3
a2 = 0.1
b1 = 0.2
b2 = 0.02
g1 = -0.3
g2 = 0.5
k1 = 0.07
k2 = 0.0022

# equations

x1eq = (b2/a2)**(1.0/g2)
x2eq = (b1/a1)**(1.0/g1)

def model(t, y):
    x1 = y[0]
    x2 = y[1]
    z  = y[2]

    dx1 = a1*x1*x2**g1 - b1*x1
    dx2 = a2*x2*x1**g2 - b2*x2
    dz  = -k1*np.sqrt(np.max([0.0, x1 - x1eq])) + k2*np.sqrt(np.max([0.0, x2 - x2eq]))

    return np.array([dx1, dx2, dz])

backend = 'vode'
#backend = 'dopri5'
#backend = 'lsoda'

t0 = 0.0
t1 = 2000.0

y0 = np.array([10.0, 5.0, 95.0])

solver = ode(model).set_integrator(backend, atol=10**-8, rtol=10**-8,nsteps=100000,method='bdf')
solver.set_initial_value(y0, t0).set_f_params()

# suppress Fortran-printed warning
solver._integrator.iwork[2] = -1

sol = []
warnings.filterwarnings("ignore", category=UserWarning)

while solver.t < t1:
    solver.integrate(t1, step=True)
    sol.append([solver.t, solver.y[0], solver.y[1], solver.y[2]])
    if not solver.successful():
        print('Woops: not success')
        break

warnings.resetwarnings()
sol2 = np.array(sol)

plt.figure(figsize=(8,2.5))
plt.subplot(1,2,1)
plt.plot(sol2[:,0], sol2[:,1], 'b', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Osteoclast",fontsize=14)
plt.xlabel(r"Time",fontsize=14)
plt.subplot(1,2,2)
plt.plot(sol2[:,0], sol2[:,2], 'g', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Osteoblast",fontsize=14)
plt.xlabel(r"Time",fontsize=14)
plt.tight_layout()

plt.figure(figsize=(4,2.5))
plt.plot(sol2[:,0], sol2[:,3], 'k', linewidth=2)
plt.xlim([t0,t1])
plt.ylabel(r"Bone Mass",fontsize=14)
plt.xlabel(r"Time",fontsize=14)
plt.tight_layout()
