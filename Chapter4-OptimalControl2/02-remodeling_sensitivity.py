# Optimal Control 2: Cellular-Molecular Level
# Bone remodeling model, sensitivity analysis
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
        
# aC up -> OCs down but controlled, OBs up, period same?
#'aC': 3.2e-1, #farhat (check: OBs)
'aC': 3.0e0, #komarova
#'aC': 1.0e-2, #---estimate---#
#'aC': 2.0e-1, #---estimate---#  

# bC up -> OCs flat, OBs down and flat, period down
'bC': 3.0e-1, #farhat (same)
#'bC': 2.0e-1, #komarova (same)
#'bC': 5.0e-1, #--estimate--#
#'bC': 1.0e0, #--estimate--#
 
# bCT up -> OCs down, OBs down, period down
#'bCT': 1.2, #farhat (check: mass action)
'bCT': 1.3e-1, #ross
#'bCT': 1.0e-3, #--estimate--#
#'bCT': 5.0e-2, #--estimate--#

# aBW up -> OCs down, OBs up, period up
'aBW': 2.6e-1, #farhat (check: Wnt)
#'aBW': 1.0e-2, #--estimate--#
#'aBW': 2.0e0, #--estimate--#
#'aBW': 1.0e0, #--estimate--#

# bB up -> OCs up, OBs down, period down
#'bB': 3.0e-1, #farhat (same)
#'bB': 1.0e-2, #--estimate--#
#'bB': 1.0e-1, #--estimate--#
#'bB': 7.0e-1, #--estimate--#
'bB': 1.0e0, #--estimate--#

# aT up -> OCs down, OBs down, period down
#'aT': 1.0e0, #pivonka2008 (same)
#'aT': 1.0e1, #--estimate--#
'aT': 1.0e2, #--estimate--#

# bT up -> OCs up, OBs up, period up
'bT': 499.1, #farhat (same)
#'bT': 2.0e2, #--estimate--#
#'bT': 1.0e1, #--estimate--#

# aW up -> OCs down, OBs up, period up 
#'aW': 5.0e-1, #--estimate--#
'aW': 1.0e0, #--estimate--#
#'aW': 1.0e2, #--estimate--#

# bW up -> OCs up, OBs down, period down
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
DSargs.tdomain = [0,200]

ode  = PyDSTool.Generator.Vode_ODEsystem(DSargs)
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)  

plt.figure(figsize=(3,1.75))
plt.plot(pd['t'], pd['xC'],'#1f77b4',linewidth=1)
plt.plot(pd['t'], pd['xB'],'#ff7f0e',linewidth=1)

plt.legend(['OCs','OBs'],
           ncol=2,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.ylim([0,10])
plt.tight_layout()

plt.figure(figsize=(3,1.75))
plt.plot(pd['t'], pd['xT'],'#2ca02c',linewidth=1)
plt.plot(pd['t'], pd['xW'],'#d62728',linewidth=1)

plt.legend(['TGFb','Wnt'],
           ncol=2,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.ylim([0,10])
plt.tight_layout()

#%%
#--- aC analysis
parVals = [1.0, 2.0, 3.0]

plt.figure(figsize=(6,1.75))

plt.hold(True)

ode.set(pars =  {'aC' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'aC' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'aC' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)

plt.subplot(121)
plt.legend(['aC=1.0','aC=2.0','aC=3.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([3.,6.])
plt.yticks([3.0,4.5,6.0])

plt.subplot(122)
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,4.])
plt.yticks([0.0,2.0,4.0])

plt.tight_layout()

ode.set(pars = {'aC': 3.0e0})

#%%
#--- bC analysis

parVals = [0.3, 0.5, 1.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(8,2))

plt.hold(True)
ode.set(pars =  {'bC' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e')
         
ode.set(pars =  {'bC' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--')
         
ode.set(pars =  {'bC' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4',linestyle=':')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e',linestyle=':')

plt.subplot(121)
plt.legend(['bC=0.3','bC=0.5','bC=1.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,6.])
plt.yticks([0.0,3.0,6.0])

plt.subplot(122)
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,4.])
plt.yticks([0.0,2.0,4.0])

plt.tight_layout()

ode.set(pars = {'bC': 3.0e-1})

#%%
#--- bCT analysis

parVals = [0.07, 0.1, 0.14]

plt.figure(figsize=(6,1.5))

plt.hold(True)

ode.set(pars =  {'bCT' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'bCT' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'bCT' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
         
plt.subplot(121)
plt.legend(['bCT=0.05','bCT=0.07','bCT=0.1'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([2.,8.])
plt.yticks([2.0,5.0,8.0])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,16.])
plt.yticks([0.0,8.0,16.0])

plt.tight_layout()

ode.set(pars = {'bCT': 1.3e-1})


#%%
#--- aBW analysis

parVals = [0.1, 0.2, 0.5]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'aBW' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'aBW' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'aBW' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['aBW=0.1','aBW=0.2','aBW=0.5'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([2.,8.])
plt.yticks([2.0,5.0,8.0])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,12.])
plt.yticks([0.0,6.0,12.0])

plt.tight_layout()

ode.set(pars = {'aBW': 2.6e-1})
#%%
parVals = [0.1, 0.5, 1.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(8,2))

plt.hold(True)
ode.set(pars =  {'bB' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e')
         
ode.set(pars =  {'bB' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--')
         
ode.set(pars =  {'bB' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4',linestyle=':')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e',linestyle=':')
plt.subplot(121)
plt.legend(['bB=0.1','bB=0.5','bB=1.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,8.])
plt.yticks([0.0,4.0,8.0])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,8.])
plt.yticks([0.0,4.0,8.0])

plt.tight_layout()

ode.set(pars = {'bB': 1.0e0})
#%%
parVals = [10.0, 50.0, 100.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(6,1.75))

plt.hold(True)
ode.set(pars =  {'aT' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linewidth=1)
         
ode.set(pars =  {'aT' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--',linewidth=1)
         
ode.set(pars =  {'aT' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle=':',linewidth=1)
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle=':',linewidth=1)
plt.subplot(121)
plt.legend(['aT=10.0','aT=50.0','aT=100.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([2.,28.])
plt.yticks([2.0,15.0,28.0])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,14.])
plt.yticks([0.0,7.0,14.0])

plt.tight_layout()


ode.set(pars = {'aT': 1.0e2})
#%%
parVals = [0.1, 1.0, 2.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(8,2))

plt.hold(True)
ode.set(pars =  {'aW' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e')
         
ode.set(pars =  {'aW' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--')
         
ode.set(pars =  {'aW' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4',linestyle=':')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e',linestyle=':')
         
plt.subplot(121)
plt.legend(['aW=0.1','aW=1.0','aW=2.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,15.])
plt.yticks([0.0,7.5,15.0])

plt.subplot(122)
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,14.])
plt.yticks([0.0,6.5,13.0])

plt.tight_layout()

ode.set(pars = {'aW': 1.0e0})
#%%
parVals = [1.0, 2.0, 5.0]
#np.linspace(0.01,4.0,4)
plt.figure(figsize=(8,2))

plt.hold(True)
ode.set(pars =  {'bW' : parVals[0]})
traj = ode.compute('odeSol1')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e')
         
ode.set(pars =  {'bW' : parVals[1]})
traj = ode.compute('odeSol2')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],color='#1f77b4',linestyle='--')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],color='#ff7f0e',linestyle='--')
         
ode.set(pars =  {'bW' : parVals[2]})
traj = ode.compute('odeSol3')
pd   = traj.sample(dt=0.1)

# plot
plt.subplot(121)
plt.plot(pd['t'], pd['xC'],'#1f77b4',linestyle=':')
plt.subplot(122)
plt.plot(pd['t'], pd['xB'],'#ff7f0e',linestyle=':')
plt.subplot(121)
plt.legend(['bW=1.0','bW=2.0','bW=5.0'],
           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,7.])
plt.yticks([0.0,3.5,7.0])

plt.subplot(122)
#plt.legend(['bCT=0.05','bCT=0.07','bC=0.1'],
#           ncol=3,fontsize=8,loc='upper center')
plt.xlabel('Time (days)',fontsize=10)
#plt.ylabel('Cell Density',fontsize=10)
plt.xlim([100,200])
plt.ylim([0.,4.])
plt.yticks([0.0,2.0,4.0])

plt.tight_layout()

ode.set(pars = {'bW': 1.0e0})


