# Optimal Control 2: Cellular-Molecular Level
# Bone metastasis model, optimal control plots
#
# Ariel Camacho
# Doctorate Thesis
# Guanajuato, Mexico, 2019

import matplotlib.pyplot as plt
import csv

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

import PyDSTool

plt.rcParams['text.usetex'] = True

#---ODE Simulation---#



#--- BONE METASTASIS ---#

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

xM0  = pd['t']
xCM0 = pd['xC']
xBM0 = pd['xB']
xMM0 = pd['xM']

#%%
plt.close('all')

def extractMetExp_4ctrl():
    x  = []
    u1 = []
    u2 = []
    u3 = []
    u4 = []
    
    folderName = './met_exp3/gauss_ALL/'
    
    with open(folderName + 'discretization_times.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            x.append(float(row[0]))
    
    with open(folderName + 'u1.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            u1.append(float(row[0]))
    
    with open(folderName + 'u2.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            u2.append(float(row[0]))
            
    with open(folderName + 'u3.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            u3.append(float(row[0]))
            
    with open(folderName + 'u4.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            u4.append(float(row[0]))
            
    xC = []
    
    with open(folderName + 'xC.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            xC.append(float(row[0]))
    
    xB = []
    
    with open(folderName + 'xB.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            xB.append(float(row[0]))
    
    xT = []
    
    with open(folderName + 'xT.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            xT.append(float(row[0]))
    
    xW = []
    
    with open(folderName + 'xW.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            xW.append(float(row[0]))

    xM = []
    
    with open(folderName + 'xM.export','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            xM.append(float(row[0]))
            
    return x, u1, u2, u3, u4, xC, xB, xT, xW, xM


x, u1, u2, u3, u4, xC, xB, xT, xW, xM = extractMetExp_4ctrl()

plt.figure(figsize=(8,3.5))

plt.subplot(2,4,1)
plt.plot(xM0,xCM0,':',color='#1f77b4',linewidth=1)
plt.plot(x,xC,'#1f77b4',linewidth=1)
plt.ylabel('OCs',fontsize=12)
plt.ylim([1,9])
plt.yticks([1,5,9])

plt.subplot(2,4,2)
plt.plot(xM0,xBM0,':',color='#ff7f0e',linewidth=1)
plt.plot(x,xB,'#ff7f0e',linewidth=1)
plt.ylabel('OBs',fontsize=12)
plt.ylim([0,9])
plt.yticks([0,4.5,9])

plt.subplot(2,4,3)
plt.plot(xM0,xMM0,':',color='#9467bd',linewidth=1)
plt.plot(x,xM,color='#9467bd',linewidth=1)
plt.ylabel('CCs',fontsize=12)

plt.subplot(2,4,4)
plt.plot(xCM0,xBM0,':',color='b',alpha=0.5,linewidth=1)
plt.plot(xC,xB,'b',linewidth=1)
plt.plot(xC[0],xB[0],'ro',linewidth=1)
plt.xlim([2,7])
plt.xticks([2,4.5,7])
plt.ylim([0,9])
plt.yticks([0,4.5,9])
plt.xlabel('OCs',fontsize=12)
plt.ylabel('OBs',fontsize=12)

plt.subplot(2,4,5)
plt.plot(x,u1,'k',linewidth=1)
plt.ylim([0,0.05])
plt.yticks([0.0, 0.025, 0.05])
plt.fill_between(x,u1,color='k',alpha=0.5)
plt.xlabel('Time (days)',fontsize=12)
plt.ylabel('Bisphosphonates',fontsize=12)
  
plt.subplot(2,4,6)
plt.plot(x,u2,'r',linewidth=1)
plt.ylim([0,200])
plt.yticks([0,100,200])
plt.fill_between(x,u2,color='r',alpha=0.5)
plt.xlabel('Time (days)',fontsize=12)
plt.ylabel('Anti-TGFb',fontsize=12)

plt.subplot(2,4,7)
plt.plot(x,u3,'m',linewidth=1)
plt.ylim([0,5])
plt.yticks([0,2.5,5])
plt.fill_between(x,u3,color='m',alpha=0.5)
plt.xlabel('Time (days)',fontsize=12)
plt.ylabel('Wnt input',fontsize=12)

plt.subplot(2,4,8)
plt.plot(x,u4,'g',linewidth=1)
plt.ylim([0,0.1])
plt.yticks([0.00, 0.05, 0.1])
plt.fill_between(x,u4,color='g',alpha=0.5)
plt.xlabel('Time (days)',fontsize=12)
plt.ylabel('Chemotherapy',fontsize=12)

plt.tight_layout()
