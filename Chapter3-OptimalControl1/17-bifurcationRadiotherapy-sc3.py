# Optimal Control 1: Cellular Level
# Bifurcation with respect to radiotherapy, Scenario 3
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
       r'\usepackage{siunitx}',
       r'\sisetup{detect-all}',
       r'\usepackage{helvet}',
       r'\usepackage{sansmath}',
       r'\sansmath'
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
uR = 0.0
#---------
#-- sc3
a3 = 1.e-4
s1 = 0.
s3 = 1.e-8
s4 = -1.e-4
bifuStep = 5.e0
bifuSteps = 2000

# equations


#---Steady-states computations



#a1, a2 = sp.symbols(('a1', 'a2'))
#b1, b2 = sp.symbols(('b1', 'b2'))
#g1, g2 = sp.symbols(('g1', 'g2'))
#a3, b3 = sp.symbols(('a3', 'b3'))
#s1, s2, s3, s4 = sp.symbols(('s1', 's2', 's3', 's4'))

x1eq = ((b2 + u2*uR)/a2)**(1.0/g2)
x2eq = ((b1 + u1*uR)/a1)**(1.0/g1)

print "\n"
print "STEADY-STATES (CANCER-FREE)"
print x1eq, x2eq, 0.0

r = a3/K
d = a1*a2*r + a1*s2*s3 + a2*s1*s4
x1can = ( (a1*(r*(b2 + u2*uR)+(b3 + uR)*s2-a3*s2)-s4*((b1 + u1*uR)*s2-(b2 + u2*uR)*s1))/d )**(1.0/g2)
x2can = ( (a2*(r*(b1 + u1*uR)+(b3 + uR)*s1-a3*s1)-s3*((b1 + u1*uR)*s2-(b2 + u2*uR)*s1))/d )**(1.0/g1)
x3can = ( a1*a2*a3 - a1*a2*(b3 + uR) + a1*s3*(b2 + u2*uR) + a2*s4*(b1 + u1*uR) )/d

print "\n"
print "STEADY-STATES (CANCER-INVASION)"
print x1can, x2can, x3can

#x1, x2, x3 = sp.symbols(('x1', 'x2', 'x3'), real=True)
#
## model
#dx1 = a1*x1*x2**g1 - (b1 + u1*uR)*x1 + s1*x1*x3
#dx2 = a2*x2*x1**g2 - (b2 + u2*uR)*x2 + s2*x2*x3
#dx3 = a3*x3*(1.0 - x3/K) - (b3 + uR)*x3 + s3*x1**g2*x3 + s4*x2**g1*x3
#
#
##---numeric steady-states
#ss = sp.solve((dx1,dx2,dx3),(x1,x2,x3), dict=True)
#
#print "\n"
#print "STEADY-STATES (NUMERICAL)"
#print ss

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
'uR': uR,
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
'x1': 'a1*x1*x2**g1 - (b1 + u1*uR)*x1 + s1*x1*x3',
'x2': 'a2*x2*x1**g2 - (b2 + u2*uR)*x2 + s2*x2*x3',
'x3': 'a3*x3*(1.0 - x3/K) - (b3 + uR)*x3 + s3*x1**g2*x3 + s4*x2**g1*x3'
}

# time
DSargs.tdomain = [0.,5.]
DSargs.pdomain = {'uR': [0.0, 0.95]}

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

#%%
#---Bifurcation Diagram

# CHECK RESULT: 2 cancer-free, 3 cancer colonization

DSargs.ics      = {
'x1': x1can, 'x2': x2can, 'x3': x3can
}

ode3  = PyDSTool.Generator.Vode_ODEsystem(DSargs)

PyCont = PyDSTool.ContClass(ode3)

# bifPar = 's4'
bifPar = 'uR'

PCargs = PyDSTool.args(name='EQ1', type='EP-C')
PCargs.freepars = [bifPar]
PCargs.StepSize = bifuStep
PCargs.MaxNumPoints = bifuSteps
PCargs.MaxStepSize = bifuStep
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True

print("Calculating EQ1 curve ...")
PyCont.newCurve(PCargs)
PyCont['EQ1'].forward()
# PyCont['EQ1'].backward()
#%%
#PCargs.name = 'LC1'
#PCargs.type = 'LC-C'
#PCargs.initpoint = 'EQ1:H1'
##PCargs.initpoint = 'EQ1:H2'
#PCargs.StepSize = 0.05
#PCargs.MinStepSize = 0.05
#PCargs.MaxStepSize = 0.05
#PCargs.force = True
#PCargs.NumSPOut = 1
#PCargs.verbosity = 1
#PCargs.SolutionMeasures = 'all'
#PCargs.LocBifPoints = 'all'
#PCargs.FuncTol = 1e-4
#PCargs.VarTol = 1e-4
#PCargs.TestTol = 1e-4
#PCargs.MaxNumPoints = 500
#PCargs.SaveEigen = True
#
#print("Calculating EQ1:LC1 curve ...")
#PyCont.newCurve(PCargs)
#PyCont['LC1'].backward()
#
#plt.figure()
#PyCont['EQ1'].display((bifPar,'x3'),axes=(1,1,1),stability=True)
#PyCont['LC1'].display((bifPar,'x3_min'),axes=(1,1,1),stability=True)
#PyCont['LC1'].display((bifPar,'x3_max'),axes=(1,1,1),stability=True)
#PyCont.plot.toggleLabels('off')
#PyCont.plot.togglePoints('off')
#%%

# PCargs.initpoint = {
# 'x1': x1eq, 'x2': x2eq, 'x3': 0.0
# }
#
# PCargs.name = 'EQ2'
# PCargs.force = True
# PCargs.type = 'EP-C'
# PCargs.freepars = [bifPar]
# PCargs.StepSize = 1e-4
# PCargs.MaxNumPoints = 100
# PCargs.MaxStepSize = 1e-4
# PCargs.LocBifPoints = 'all'
# PCargs.FuncTol = 1e-7
# PCargs.VarTol = 1e-7
# PCargs.TestTol = 1e-7
# PCargs.SaveEigen = True
#
# print("Calculating EQ2 curve ...")
# PyCont.newCurve(PCargs)
# # PyCont['EQ2'].forward()
# PyCont['EQ2'].backward()

#plt.figure()
#PyCont['EQ2'].display((bifPar,'x3'),axes=(1,1,1),stability=True)
#%%
plt.close('all')
plt.figure(figsize=(8,5))

PyCont['EQ1'].display((bifPar,'x3'),axes=(1,1,1),stability=True,linewidth=2)
# PyCont['EQ2'].display((bifPar,'x3'),axes=(1,1,1),stability=True,linewidth=2)

plt.title("")
plt.xlabel(r"Bifurcation Parameter $u_D$",fontsize=18)
plt.ylabel(r"Cancer Cells $T_I$",fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

PyCont.plot.toggleLabels('off')
PyCont.plot.togglePoints('off')

# PyCont.plot.togglePoints(visible='on', bylabel='H2')
# PyCont.plot.togglePoints(visible='on', bylabel='BP1')
#PyCont.plot.toggleLabels(visible='on', bylabel='H2')
#PyCont.plot.toggleLabels(visible='on', bylabel='BP1')

#-- sc2
# plt.ylim([-60,60])
# plt.xlim([-0.020,0.01])
fig = mpl.pyplot.gcf()
fig.set_size_inches(5, 3)
plt.tight_layout()

plt.show()
