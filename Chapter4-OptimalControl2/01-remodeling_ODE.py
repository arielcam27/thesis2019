# Optimal Control 2: Cellular-Molecular Level
# Bone remodeling model, base parameters
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

# baseline initial conditions
DSargs.ics      = {'xC': 5.0e0, 
                   'xB': 1.0e0, 
                   'xT': 0.0,
                   'xW': 0.0}

# time
DSargs.tdomain = [0,100]

# solve
ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)
traj = ode.compute('odeSol')
pd   = traj.sample(dt=0.1)

# plot
plt.figure(figsize=(3,1.75))
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)

plt.legend(['OCs','OBs'],
           ncol=5,fontsize=7,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.ylim([0,10])
plt.tight_layout()

plt.figure(figsize=(4,2))
plt.plot(pd['t'], pd['xT'],'#2ca02c')
plt.plot(pd['t'], pd['xW'],'#d62728')
plt.legend(['TGFb','Wnt'],
           ncol=5,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Mol. Density',fontsize=10)
plt.ylim([0,10])
plt.tight_layout()
plt.show()
