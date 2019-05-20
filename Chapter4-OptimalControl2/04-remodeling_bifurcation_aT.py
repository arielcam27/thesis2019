# Optimal Control 2: Cellular-Molecular Level
# Bone remodeling model, bifurcation for alpha_T
#
# Ariel Camacho
# Doctorate Thesis
# Guanajuato, Mexico, 2019

import PyDSTool
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# name of system
DSargs = PyDSTool.args(name='ode')

# parameters
DSargs.pars = {
#'aC': 3.2e-1, #farhat (check: OBs)
'aC': 3.0e0, #komarova
#'aC': 1.0e-2, #---estimate---#
#'aC': 2.0e-1, #---estimate---#  

'bC': 3.0e-1, #farhat (same)
#'bC': 2.0e-1, #komarova (same)
#'bC': 5.0e-1, #--estimate--#
#'bC': 1.0e0, #--estimate--#

#'bCT': 1.2, #farhat (check: mass action)
'bCT': 1.3e-1, #ross
#'bCT': 1.0e-3, #--estimate--#
#'bCT': 5.0e-2, #--estimate--#

'aBW': 2.6e-1, #farhat (check: Wnt)
#'aBW': 1.0e-2, #--estimate--#
#'aBW': 2.0e0, #--estimate--#
#'aBW': 1.0e0, #--estimate--#

# bB up -> frequency up
#'bB': 3.0e-1, #farhat (same)
#'bB': 1.0e-2, #--estimate--#
#'bB': 1.0e-1, #--estimate--#
#'bB': 7.0e-1, #--estimate--#
'bB': 1.0e0, #--estimate--#

#'aT': 1.0e0, #pivonka2008 (same)
#'aT': 1.0e1, #--estimate--#
'aT': 1.0e2, #--estimate--#

# bT up -> OBs up
'bT': 499.1, #farhat (same)
#'bT': 2.0e2, #--estimate--#
#'bT': 1.0e1, #--estimate--#

#'aW': 5.0e-1, #--estimate--#
'aW': 1.0e0, #--estimate--#
#'aW': 1.0e2, #--estimate--#

#'bW': 2.0e0, #farhat, buenzli (same)
'bW': 1.0e0, #--estimate--#
#'bW': 5.0e0, #--estimate--#
#'bW': 1.0e1, #--estimate--#
}

# model equations                   
DSargs.varspecs = {
'xC': 'aC*xB^(-1.0) - bC*xC - bCT*xC*xT',
'xB': 'aBW*xB*xW - bB*xB',
'xT': 'aT*xC - bT*xT',
'xW': 'aW*xT*xC - bW*xW'}

#---Steady-states computations

xC,xB,xT,xW,xM = sp.symbols(('xC','xB','xT','xW','xM'))

# pars values
aC=DSargs.pars['aC'];
bC=DSargs.pars['bC'];
bCT=DSargs.pars['bCT'];

aBW=DSargs.pars['aBW'];
bB=DSargs.pars['bB'];

aT=DSargs.pars['aT'];
bT=DSargs.pars['bT'];

aW=DSargs.pars['aW'];
bW=DSargs.pars['bW'];

# model
f1 = aC/xB - bC*xC - bCT*xC*xT;
f2 = aBW*xB*xW - bB*xB;
fT = aT*xC - bT*xT;
fW = aW*xT*xC - bW*xW;

#---numeric steady-states
ss = sp.solve((f1,f2,fT,fW),(xC,xB,xT,xW),dict=True)

print "\n"
print "STEADY-STATES (NUMERICAL)"
print ss

#---Bifurcation Diagram


DSargs.ics      = {
'xC': ss[1][xC],
'xB': ss[1][xB],
'xT': ss[1][xT],
'xW': ss[1][xW]}

ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)

PyCont = PyDSTool.ContClass(ode)

bifPar = 'aT'

PCargs = PyDSTool.args(name='EQ1', type='EP-C')
PCargs.freepars = [bifPar]
PCargs.StepSize = 1e0
PCargs.MaxNumPoints = 200
PCargs.MaxStepSize = 1e0
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True

print("Calculating EQ1 curve ...")
PyCont.newCurve(PCargs)
PyCont['EQ1'].forward()
PyCont['EQ1'].backward()

PyCont['EQ1'].display((bifPar,'xC'),axes=(2,1,1),stability=True)
PyCont['EQ1'].display((bifPar,'xB'),axes=(2,1,2),stability=True)

PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ1:H1'
PCargs.StepSize = 1.0
PCargs.MinStepSize = 1.0
PCargs.MaxStepSize = 1.0
PCargs.force = True
PCargs.NumSPOut = 1
PCargs.verbosity = 1
PCargs.SolutionMeasures = 'all'
PCargs.LocBifPoints = 'all'
PCargs.FuncTol = 1e-5
PCargs.VarTol = 1e-5
PCargs.TestTol = 1e-5
PCargs.MaxNumPoints = 500
PCargs.SaveEigen = True

print("Calculating EQ1:LC1 curve ...")
PyCont.newCurve(PCargs)
PyCont['LC1'].backward()
#%%
PyCont['EQ1'].display((bifPar,'xC'),axes=(1,2,1),stability=True)
PyCont['LC1'].display((bifPar,'xC_min'),axes=(1,2,1),stability=True)
PyCont['LC1'].display((bifPar,'xC_max'),axes=(1,2,1),stability=True)
plt.title('')
plt.xlabel('')
plt.ylabel('$x_C$',fontsize=20)
plt.xlim([0.0,200.0])
plt.ylim([0,14])
plt.yticks([0,7,14])

plt.xlabel('$\\alpha_{T}$',fontsize=20)

PyCont['EQ1'].display((bifPar,'xB'),axes=(1,2,2),stability=True)
PyCont['LC1'].display((bifPar,'xB_min'),axes=(1,2,2),stability=True)
PyCont['LC1'].display((bifPar,'xB_max'),axes=(1,2,2),stability=True)
plt.title('')
plt.ylabel('$x_B$',fontsize=20)
plt.xlim([0.0,200.0])
plt.ylim([0,14])
plt.yticks([0,7,14])

plt.xlabel('$\\alpha_{T}$',fontsize=20)

PyCont.plot.toggleLabels('off')
PyCont.plot.togglePoints('off')
PyCont.plot.togglePoints(visible='on', bylabel='H1')
#PyCont.plot.toggleLabels('on','H1')
#PyCont.plot.setLabels('$H1$', bylabel='H1')

fig = plt.gcf()
fig.set_size_inches(8, 2)
plt.tight_layout()
