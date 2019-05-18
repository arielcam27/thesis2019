# Optimal Control 1: Cellular Level
# Bifurcation with respect to denosumab, Scenario 1
#
# Ariel Camacho
# Doctorate Thesis
# Guanajuato, Mexico, 2019


import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import PyDSTool
import sympy as sp

plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]



# parameters

a1 = 0.5
a2 = 0.05
b1 = 0.2
b2 = 0.02
b3 = 0.
s2 = 0.
g1 = -0.3
g2 = 0.7
K  = 1.e4
u1 = 1.
u2 = 1.
#---------
uD = 0.0
#---------
#-- sc3
a3 = 1.e-4
s1 = 0.
s3 = 1.e-8
s4 = -1.e-4
bifuStep = 5.e0
bifuSteps = 2000



# equations

x1eq = (b2/a2)**(1.0/g2)
x2eq = (b1/a1)**(1.0/g1)

print "\n"
print "STEADY-STATES (CANCER-FREE)"
print x1eq, x2eq, 0.0

r = a3/K
d = a1*(1.0 - uD)*a2*r + a1*(1.0 - uD)*s2*s3 + a2*s1*s4
x1can = ( (a1*(1.0 - uD)*(r*b2+b3*s2-a3*s2)-s4*(b1*s2-b2*s1))/d )**(1.0/g2)
x2can = ( (a2*(r*b1+b3*s1-a3*s1)-s3*(b1*s2-b2*s1))/d )**(1.0/g1)
x3can = ( a1*(1.0 - uD)*a2*a3 - a1*(1.0 - uD)*a2*b3 + a1*(1.0 - uD)*s3*b2 + a2*s4*b1 )/d

print "\n"
print "STEADY-STATES (CANCER-INVASION)"
print x1can, x2can, x3can

#%%
# name of system
DSargs = PyDSTool.args(name='ode')

# parameters
DSargs.pars = {
'a1': a1,
'a2': a2,
'b1': b1,
'b2': b2,
'b3': b3,
's2': s2,
'g1': g1,
'g2': g2,
'K':  K,
'u1': u1,
'u2': u2,
#---------
'uD': uD,
#---------
#-- sc1
'a3': a3,
's1': s1,
's3': s3,
's4': s4
}

# baseline initial conditions
DSargs.ics      = {'x1': 0.5,
                   'x2': 20.0,
                   'x3': 6000.0}

# model equations
DSargs.varspecs = {
'x1': 'a1*(1.0 - uD)*x1*x2**g1 - b1*x1 + s1*x1*x3',
'x2': 'a2*x2*x1**g2 - b2*x2 + s2*x2*x3',
'x3': 'a3*x3*(1.0 - x3/K) - b3*x3 + s3*x1**g2*x3 + s4*x2**g1*x3'
}

# time
DSargs.tdomain = [0.,5.]
DSargs.pdomain = {'uD': [0.0, 0.95]}

# solve
ode2  = PyDSTool.Generator.Vode_ODEsystem(DSargs)
#ode2  = PyDSTool.Generator.Dopri_ODEsystem(DSargs)
#ode2  = PyDSTool.Generator.Radau_ODEsystem(DSargs)
traj = ode2.compute('odeSol')
pd   = traj.sample(dt=0.1)

t0 = pd['t']
x10 = pd['x1']
x20 = pd['x2']
x30 = pd['x3']

plt.figure()
plt.subplot(2,2,1)
plt.plot(t0,x10)
plt.subplot(2,2,2)
plt.plot(t0,x20)
plt.subplot(2,2,3)
plt.plot(t0,x30)
# plt.show()

DSargs.ics      = {
'x1': x1can, 'x2': x2can, 'x3': x3can
}

ode3  = PyDSTool.Generator.Vode_ODEsystem(DSargs)

PyCont = PyDSTool.ContClass(ode3)

bifPar = 'uD'

PCargs = PyDSTool.args(name='EQ1', type='EP-C')
PCargs.freepars = [bifPar]
PCargs.StepSize = bifuStep
PCargs.MaxNumPoints = bifuSteps
PCargs.MaxStepSize = bifuStep
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True

print("Calculating EQ1 curve ...")
PyCont.newCurve(PCargs)
# PyCont['EQ1'].forward()
PyCont['EQ1'].backward()

plt.close('all')
plt.figure(figsize=(8,5))

PyCont['EQ1'].display((bifPar,'x3'),axes=(1,1,1),stability=True,linewidth=2)

plt.title("")
plt.xlabel(r"Bifurcation Parameter $u_D$",fontsize=18)
plt.ylabel(r"Cancer Cells $T_I$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

PyCont.plot.toggleLabels('off')
PyCont.plot.togglePoints('off')

#-- sc2
# plt.ylim([-60,60])
# plt.xlim([-0.020,0.01])
fig = mpl.pyplot.gcf()
fig.set_size_inches(5, 3)
plt.tight_layout()

plt.show()
