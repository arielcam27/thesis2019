# Optimal Control 2: Cellular-Molecular Level
# Bone metastasis model, bifurcation for beta_B
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

'KC': 0.5,
'KB': 0.2,

'aCM': 1.5,

#'aM' : 2.3e-3, #farhat (check: logistic)
'aM' : 1.0e-3,
'KM' : 1.0,
'aMT': 1.0e-1
}

# model equations                   
DSargs.varspecs = {
'xC': '(1.0 - KC*xM/KM)*aC*xB^(-1.0) - bC*xC - bCT*xC*xT + aCM*xM',
'xB': 'aBW*xB*xW - bB*xB',
'xT': 'aT*xC - bT*xT',
'xW': '(1.0 - KB*xM/KM)*aW*xT*xC - bW*xW',
'xM': 'xM*(aM + aMT*xT)*(1.0 - xM/KM)'}

#---Steady-states computations

xC,xB,xT,xW,xM = sp.symbols(('xC','xB','xT','xW','xM'))

# pars values
aC=DSargs.pars['aC'];
bC=DSargs.pars['bC'];
bCT=DSargs.pars['bCT'];
aCM=DSargs.pars['aCM'];

aBW=DSargs.pars['aBW'];
bB=DSargs.pars['bB'];

aT=DSargs.pars['aT'];
bT=DSargs.pars['bT'];

aW=DSargs.pars['aW'];
bW=DSargs.pars['bW'];

KC=DSargs.pars['KC'];
KB=DSargs.pars['KC'];
aM=DSargs.pars['aM'];
KM=DSargs.pars['KM'];
aMT=DSargs.pars['aMT'];

# model
f1 = (1.0-KC*xM/KM)*aC/xB - bC*xC - bCT*xC*xT + aCM*xM;
f2 = aBW*xB*xW - bB*xB;
fT = aT*xC - bT*xT;
fW = (1.0 - KB*xM/KM)*aW*xT*xC - bW*xW;
fM = (aM +  aMT*xT)*xM*(1.0 - xM/KM);

#---numeric steady-states
ss = sp.solve((f1,f2,fT,fW,fM),(xC,xB,xT,xW,xM),dict=True)

print "\n"
print "STEADY-STATES (NUMERICAL)"
print ss

#%%
#---Bifurcation Diagram

DSargs.ics      = {
'xC': ss[4][xC],
'xB': ss[4][xB],
'xT': ss[4][xT],
'xW': ss[4][xW],
'xM': ss[4][xM]}

ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)

PyCont = PyDSTool.ContClass(ode)

# bifurcation parameter
bifPar = 'bB'

PCargs = PyDSTool.args(name='EQ1', type='EP-C')
PCargs.freepars = [bifPar]
PCargs.StepSize = 1e-1
PCargs.MaxNumPoints = 200
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True

print("Calculating EQ1 curve ...")
PyCont.newCurve(PCargs)
PyCont['EQ1'].forward()
PyCont['EQ1'].backward()

#%%
PCargs.name = 'LC1'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ1:H1'
PCargs.StepSize = 0.5
PCargs.MinStepSize = 0.5
PCargs.MaxStepSize = 0.5
PCargs.force = True
PCargs.NumSPOut = 1
PCargs.verbosity = 1
PCargs.SolutionMeasures = 'all'
PCargs.LocBifPoints = 'all'
PCargs.FuncTol = 1e-5
PCargs.VarTol = 1e-5
PCargs.TestTol = 1e-5
PCargs.MaxNumPoints = 100
PCargs.SaveEigen = True

print("Calculating EQ1:LC1 curve ...")
PyCont.newCurve(PCargs)
PyCont['LC1'].backward()

## cancer-invasion bifurcation curve
#plt.figure(figsize=(5,2));
#
#PyCont['EQ1'].display((bifPar,'xC'),axes=(2,1,1),stability=True)
#PyCont['LC1'].display((bifPar,'xC_min'),axes=(2,1,1),stability=True)
#PyCont['LC1'].display((bifPar,'xC_max'),axes=(2,1,1),stability=True)
#plt.title('')
#plt.xlabel('')
#plt.ylabel('$x_C$',fontsize=12)
#plt.xlim([1,2])
#
#PyCont['EQ1'].display((bifPar,'xB'),axes=(2,1,2),stability=True)
#PyCont['LC1'].display((bifPar,'xB_min'),axes=(2,1,2),stability=True)
#PyCont['LC1'].display((bifPar,'xB_max'),axes=(2,1,2),stability=True)
#plt.title('')
#plt.ylabel('$x_B$',fontsize=12)
#plt.xlim([1,2])
#plt.ylim([-1,25])
#
#plt.xlabel('$\\beta_B$')
#
#PyCont.plot.toggleLabels('off')
#PyCont.plot.togglePoints('off')
#PyCont.plot.togglePoints(visible='on', bylabel='H1')
#PyCont.plot.toggleLabels('on','H1')
#PyCont.plot.setLabels('$H1$', bylabel='H1')
#
#fig = plt.gcf()
#fig.set_size_inches(4, 2)
#plt.tight_layout()

#%%
# CHECK RESULT: 2 cancer-free, 3 cancer colonization
PCargs.initpoint = {
'xC': ss[3][xC],
'xB': ss[3][xB],
'xT': ss[3][xT],
'xW': ss[3][xW],
'xM': ss[3][xM]}

PCargs.name = 'EQ2'
PCargs.type = 'EP-C'
PCargs.freepars = [bifPar]
PCargs.StepSize = 1e-1
PCargs.MaxNumPoints = 200
PCargs.MaxStepSize = 1e-1
PCargs.LocBifPoints = 'all'
PCargs.SaveEigen = True

print("Calculating EQ2 curve ...")
PyCont.newCurve(PCargs)
PyCont['EQ2'].forward()
PyCont['EQ2'].backward()

#%%
#PyCont['EQ2'].display((bifPar,'xC'),axes=(2,1,1),stability=True)
#PyCont['EQ2'].display((bifPar,'xB'),axes=(2,1,2),stability=True)
#%%
PCargs.name = 'LC2'
PCargs.type = 'LC-C'
PCargs.initpoint = 'EQ2:H1'
PCargs.StepSize = 0.5
PCargs.MinStepSize = 0.5
PCargs.MaxStepSize = 0.5
PCargs.force = True
PCargs.NumSPOut = 1
PCargs.verbosity = 1
PCargs.SolutionMeasures = 'all'
PCargs.LocBifPoints = 'all'
PCargs.FuncTol = 1e-2
PCargs.VarTol = 1e-2
PCargs.TestTol = 1e-2
PCargs.MaxNumPoints = 200
PCargs.SaveEigen = True

print("Calculating EQ2:LC2 curve ...")
PyCont.newCurve(PCargs)
PyCont['LC2'].backward()
#PyCont['LC2'].forward()

#%%
# bifurcation diagram
PyCont['EQ1'].display((bifPar,'xC'),axes=(1,2,1),stability=True)
PyCont['LC1'].display((bifPar,'xC_min'),axes=(1,2,1),stability=True)
PyCont['LC1'].display((bifPar,'xC_max'),axes=(1,2,1),stability=True)
PyCont['EQ2'].display((bifPar,'xC'),axes=(1,2,1),stability=True)
PyCont['LC2'].display((bifPar,'xC_min'),axes=(1,2,1),stability=True)
PyCont['LC2'].display((bifPar,'xC_max'),axes=(1,2,1),stability=True)
plt.title('')
plt.xlabel('')
plt.ylabel('$x_C$',fontsize=18)
plt.xlim([0.7,2.5])
plt.ylim([1.5,12])
plt.yticks([1.5,6,12])

plt.xlabel('$\\beta_B$',fontsize=18)

PyCont['EQ1'].display((bifPar,'xB'),axes=(1,2,2),stability=True)
PyCont['LC1'].display((bifPar,'xB_min'),axes=(1,2,2),stability=True)
PyCont['LC1'].display((bifPar,'xB_max'),axes=(1,2,2),stability=True)
PyCont['EQ2'].display((bifPar,'xB'),axes=(1,2,2),stability=True)
PyCont['LC2'].display((bifPar,'xB_min'),axes=(1,2,2),stability=True)
PyCont['LC2'].display((bifPar,'xB_max'),axes=(1,2,2),stability=True)
plt.title('')
plt.ylabel('$x_B$',fontsize=18)
plt.xlim([0.7,2.5])
plt.ylim([0.3,5])
plt.yticks([0.3,2.5,5])

plt.xlabel('$\\beta_B$',fontsize=18)

PyCont.plot.toggleLabels('off')
PyCont.plot.togglePoints('off')
PyCont.plot.togglePoints(visible='on', bylabel='H1')
#PyCont.plot.toggleLabels('on','H1')
#PyCont.plot.setLabels('$H1$', bylabel='H1')

fig = plt.gcf()
fig.set_size_inches(6, 1.75)
plt.tight_layout()
