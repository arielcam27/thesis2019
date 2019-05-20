# Optimal Control 1: Cellular Level
# Mixed therapy, Scenario 1
#
# Ariel Camacho
# Doctorate Thesis
# Guanajuato, Mexico, 2019

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint, ode
from matplotlib import rc
import copy

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

plt.close('all')

# parameters

# 1) Dynamical system parameters

# ODE parameters
a1 = 0.5
a2 = 0.05
b1 = 0.2
b2 = 0.02
g1 = -0.3
g2 = 0.7
K  = 1.0e4
u1 = 1.0
u2 = 1.0

uDMax = 0.6
uRMax = 0.05

# scenario 3
a3=1.e-4
b3=0.e-2
c1=0.e-6
c2=0.e-1
c3=1.e-8
c4=-1.e-4

# Temporal parameters
T = 250.                 # Final Time
N = int(T*10)            # sub-intervals numbers
t = np.linspace(0, T, N)
h = float(T)/float(N)

# Initial condition
y1 = 4.42e-06
y2 = 4.46
y3 = 1000.0


# 2) Optimal control parameters
convx = 0.9

# State system
def model(StateVar, t, Controls):
    x1 = StateVar[0]
    x2 = StateVar[1]
    x3 = StateVar[2]

    uD = Controls[0]
    uR = Controls[1]

    dx1 = a1*x1*x2**g1*(1.0-uD) - (b1+u1*uR)*x1 + c1*x1*x3
    dx2 = a2*x1**g2*x2 - (b2+u2*uR)*x2 + c2*x2*x3
    dx3 = a3*x3*(1.0 - x3/K) - (b3+uR)*x3 + c3*x1**g2*x3 + c4*x2**g1*x3

    return np.array([dx1, dx2, dx3])


# Adjoint system
def glambda(StateVar, t, Controls, l_vec):
    x1 = StateVar[0]
    x2 = StateVar[1]
    x3 = StateVar[2]

    uD = Controls[0]
    uR = Controls[1]

    l1 = l_vec[0]
    l2 = l_vec[1]
    l3 = l_vec[2]

    dl1 = -a2*g2*l2*x1**g2*x2/x1 - c3*g2*l3*x1**g2*x3/x1 - l1*(a1*x2**g1*(-uD + 1.0) - b1 + c1*x3 - u1*uR)
    dl2 = -a1*g1*l1*x1*x2**g1*(-uD + 1.0)/x2 - c4*g1*l3*x2**g1*x3/x2 - l2*(a2*x1**g2 - b2 + c2*x3 - u2*uR)
    dl3 = -c1*l1*x1 - c2*l2*x2 - l3*(a3*(1.0 - x3/K) - a3*x3/K - b3 + c3*x1**g2 + c4*x2**g1 - uR) - 2.0*x3

    return np.array([dl1, dl2, dl3])


# Control system
def control_new(StateVar, Controls, l_vec, wD, wR):
    x1 = StateVar[0]
    x2 = StateVar[1]
    x3 = StateVar[2]

    uD = Controls[0]
    uR = Controls[1]

    l1 = l_vec[0]
    l2 = l_vec[1]
    l3 = l_vec[2]

    uDNew1 = np.minimum(uDMax*np.ones_like(l1), a1*l1*x1*x2**g1/(2.0*wD))
    uDNew2 = np.maximum(np.zeros_like(uDNew1), uDNew1)

    uRNew1 = np.minimum(uRMax*np.ones_like(l1), (u1*l1*x1 + u2*l2*x2 + l3*x3)/(2.0*wR))
    uRNew2 = np.maximum(np.zeros_like(uRNew1), uRNew1)

    New_Control = np.array([uDNew2, uRNew2])

    Updated_Control = convx*Controls + (1.0 - convx)*New_Control

    return Updated_Control

# FBSM

def runge_forward(StateVar, Controls):
    x = StateVar
    c = Controls

    for i in range(N-1):
        c_medio = 0.5*(c[:,i]+c[:,i+1])

        k1 = model( x[:,i],          i, c[:,i] )
        k2 = model( x[:,i]+h*0.5*k1, i+0.5*h, c_medio)
        k3 = model( x[:,i]+h*0.5*k2, i+0.5*h, c_medio)
        k4 = model( x[:,i]+h*k3,     i+h, c[:,i+1])

        x[:,i+1] = x[:,i] + h*(k1+2.0*k2+2.0*k3+k4)/6.0
    return x

def runge_backward(StateVar, Controls, l_vec):
    x = StateVar
    c = Controls
    l = l_vec

    for i in range(N-1,0,-1):
        c_medio = 0.5*( c[:,i]+c[:,i-1] )
        x_medio = 0.5*( x[:,i]+x[:,i-1] )

        k1 = glambda( x[:,i],   i, c[:,i], l[:,i])
        k2 = glambda( x_medio,  i-0.5*h, c_medio, l[:,i]-h*0.5*k1)
        k3 = glambda( x_medio,  i-0.5*h, c_medio, l[:,i]-h*0.5*k2)
        k4 = glambda( x[:,i-1], i-h,  c[:,i-1], l[:,i]-h*k3)

        l[:,i-1] = l[:,i] - h*(k1+2.0*k2+2.0*k3+k4)/6.0
    return l


def FBSM(wD, wR):
    print("Calculating optimal solution for wD=%f, wR=%f" % (wD, wR))

    # FBSM parameters
    test          = -1
    iteration     = 0
    tolerance     = 0.0001
    maxIterations = 1000

    # Control intial guess
    uD0 = np.zeros(N)
    uR0 = np.zeros(N)

    Controls0 = np.array([uD0, uR0])

    X0 = np.array([y1,y2,y3])

    x10    = np.zeros(N)
    x10[0] = y1

    x20    = np.zeros(N)
    x20[0] = y2

    x30    = np.zeros(N)
    x30[0] = y3

    StateVar0 = np.array([x10, x20, x30])

    # Terminal condition for the adjoint variables
    l_vec0 = np.zeros_like(StateVar0)

    while(test<0):
        iteration += 1

        # strong convergence (control, states, adjoint)
        # oldControls = copy.copy(Controls0)
        # oldStateVar = copy.copy(StateVar0)
        # oldl_vec    = copy.copy(l_vec0)

        # weak convergence (control)
        oldControls = Controls0
        oldStateVar = StateVar0
        oldl_vec    = l_vec0

        # Forward State System
        StateVar0  = runge_forward(StateVar0, Controls0)

        # Backward Adjoint System
        l_vec0 = runge_backward(StateVar0, Controls0, l_vec0)

        # Control Update
        Controls0 = control_new(StateVar0, Controls0, l_vec0, wD, wR)

        # Convergence Criteria
        errorControl = np.linalg.norm(Controls0-oldControls)
        errorState   = np.linalg.norm(StateVar0-oldStateVar)
        errorL       = np.linalg.norm(l_vec0-oldl_vec)
        errorMax     = (errorControl+errorState+errorL)/3.

        # progress numerics
        if np.mod(iteration,10)==0:
            print("Error at iteration " + str(iteration) + ":", errorMax)

        if errorMax < tolerance:
            test = 1
            print('')
            print('Number of iterations until convergence:', iteration)
        elif iteration == maxIterations:
            test = 1
            print('')
            print('Failure in convergence.')

    return StateVar0, Controls0


#-- experiment 1
wD = 1.e6
wR = 1.e10

StateVar0, Controls0 = FBSM(wD, wR)

x1Opt1 = StateVar0[0,:]
x2Opt1 = StateVar0[1,:]
x3Opt1 = StateVar0[2,:]

uDOpt1 = Controls0[0,:]
uROpt1 = Controls0[1,:]

#-- experiment 2
wD = 1.e7
wR = 1.e11

StateVar0, Controls0 = FBSM(wD, wR)

x1Opt2 = StateVar0[0,:]
x2Opt2 = StateVar0[1,:]
x3Opt2 = StateVar0[2,:]

uDOpt2 = Controls0[0,:]
uROpt2 = Controls0[1,:]


#-- no control
x10    = np.zeros(N); x10[0] = y1
x20    = np.zeros(N); x20[0] = y2
x30    = np.zeros(N); x30[0] = y3
StateVar0 = np.array([x10, x20, x30])
uD0 = np.zeros(N)
uR0 = np.zeros(N)
Controls0 = np.array([uD0,  uR0])
StateVar0  = runge_forward(StateVar0, Controls0)
x1NoControl = StateVar0[0,:]
x2NoControl = StateVar0[1,:]
x3NoControl = StateVar0[2,:]


#-- plots

plt.close('all')
plt.figure(figsize=(8,5))

plt.subplot(2,2,1)
plt.plot(t, x1NoControl, color='black', linestyle='dashed', linewidth=1, label=r"\textsf{no control}")
plt.plot(t, x1Opt1, color='blue', linestyle='solid', linewidth=2, alpha=0.4, label=r"$w_D=10^6, w_R=10^{10}$")
plt.plot(t, x1Opt2, color='blue', linestyle='solid', linewidth=2, alpha=1.0, label=r"$w_D=10^7, w_R=10^{11}$")
leg = plt.legend(loc='best', fancybox=True, framealpha=0.25,fontsize=10)
plt.xlim([0.,T])
plt.ylabel(r"Osteoclast $C$",fontsize=16)
plt.xlabel(r"Time $t$",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,2)
plt.plot(t, x2NoControl, color='black', linestyle='dashed', linewidth=1)
plt.plot(t, x2Opt2, color='green',  linestyle='solid',  linewidth=2, alpha=1.0)
plt.plot(t, x2Opt1, color='green',  linestyle='solid',  linewidth=2, alpha=0.4)
plt.xlim([0.,T])
plt.ylabel(r"Osteoblast $B$",fontsize=16)
plt.xlabel(r"Time $t$",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,3)
plt.plot(t, x3NoControl, color='black', linestyle='dashed', linewidth=1)
plt.plot(t, x3Opt2, color='red',  linestyle='solid',  linewidth=2, alpha=1.0)
plt.plot(t, x3Opt1, color='red',  linestyle='solid',  linewidth=2, alpha=0.4)
plt.xlim([0.,T])
plt.ylabel(r"Cancer Cells $T$",fontsize=16)
plt.xlabel(r"Time $t$",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,4)
plt.plot(t, uDOpt1/uDMax, color='magenta',  linestyle='solid', marker='+', markevery=200, linewidth=2.5, markersize=10, alpha=0.4, label=r"$u_D, w_D=10^6, w_R=10^{10}$")
plt.plot(t, uROpt1/uRMax, color='orange',    linestyle='solid',  marker='x', markevery=200, linewidth=2.5, markersize=8, alpha=0.4, label=r"$u_R, w_D=10^6, w_R=10^{10}$")
plt.plot(t, uDOpt2/uDMax, color='magenta',  linestyle='solid', marker='+', markevery=200, linewidth=2.5, markersize=10, alpha=1.0, label=r"$u_D, w_D=10^7, w_R=10^{11}$")
plt.plot(t, uROpt2/uRMax, color='orange',    linestyle='solid',  marker='x', markevery=200, linewidth=2.5, markersize=8, alpha=1.0, label=r"$u_R, w_D=10^7, w_R=10^{11}$")
leg = plt.legend(loc='top', fancybox=True, framealpha=0.25,fontsize=10)
plt.xlim([0.,T])
plt.ylim([-0.05,1.05])
plt.ylabel(r"Optimal Controls",fontsize=16)
plt.xlabel(r"Time $t$",fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

plt.show()
