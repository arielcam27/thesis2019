# Optimal Control 2: Cellular-Molecular Level
# Bone metastasis model, sensitivity analysis
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

# baseline initial conditions
DSargs.ics      = {'xC': 5.0e0, 
                   'xB': 1.0e0, 
                   'xT': 0.0,
                   'xW': 0.0,
                   'xM': 1.0e-2}

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
plt.plot(pd['t'], pd['xM']*10,color='#9467bd',linewidth=1)

plt.legend(['OCs','OBs','CCs x10'],
           ncol=5,fontsize=7,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.ylim([0,15])
plt.tight_layout()

#plt.figure(figsize=(4,2))
#plt.plot(pd['t'], pd['xT'],'#2ca02c')
#plt.plot(pd['t'], pd['xW'],'#d62728')
#plt.legend(['TGFb','Wnt'],
#           ncol=5,fontsize=8,loc='upper center')
#plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Mol. Density',fontsize=10)
#plt.ylim([0,10])
#plt.tight_layout()

#%%
# KC analysis

parVals = [0.0, 0.5, 0.75]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'KC' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'KC' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'KC' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['KC=0.0','KC=0.5','KC=0.75'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0,8])
plt.yticks([0,4,8])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0.,10.])
plt.yticks([0.0,5.0,10.0])

plt.tight_layout()


ode.set(pars = {'KC': 0.5})

#%%
# aCM analysis

parVals = [0.0, 1.5, 2.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'aCM' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'aCM' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'aCM' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['aCM=0.0','aCM=1.5','aCM=2.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0,8])
plt.yticks([0,4,8])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0.,10.])
plt.yticks([0.0,5.0,10.0])

plt.tight_layout()


ode.set(pars = {'aCM': 1.5})

#%%
# KB analysis

parVals = [0.0, 0.5, 0.75]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'KB' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'KB' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'KB' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['KB=0.0','KB=0.5','KB=0.75'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0,10])
plt.yticks([0,5,10])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0.,10.])
plt.yticks([0.0,5.0,10.0])

plt.tight_layout()


ode.set(pars = {'KB': 0.2})

#%%
# aMT analysis

parVals = [0.0, 0.1, 1.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'aMT' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'aMT' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'aMT' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['aMT=0.0','aMT=0.1','aMT=1.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0,8])
plt.yticks([0,4,8])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([0,100])
plt.ylim([0.,10.])
plt.yticks([0.0,5.0,10.0])

plt.tight_layout()


ode.set(pars = {'aMT': 1.0e-1})
