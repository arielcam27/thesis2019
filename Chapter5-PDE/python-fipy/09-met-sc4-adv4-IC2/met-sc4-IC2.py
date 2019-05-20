import matplotlib
matplotlib.use("Agg")
from fipy import *
import matplotlib.animation as manimation
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = True

plt.close('all')

FFMpegWriter = manimation.writers['ffmpeg']
# metadata = dict(title='Bone remodeling', artist='Matplotlib', comment='Movie support!')
metadata = dict(title='Bone metastasis')
writer = FFMpegWriter(fps=15, metadata=metadata)

# temporal parameters
a1 = 0.3
a2 = 0.1
b1 = 0.2
b2 = 0.02
g1 = -0.3
g2 = 0.5
# k1 = 0.07
# k2 = 0.0022
K = 300.0

# sc1
# a3 = 0.045
# b3 = 0.05
# s1 = 0.001
# s2 = -0.00005
# s3 = 0.005
# s4 = 0.0
# k1 = 0.07
# k2 = 0.0022
#y0 = np.array([10.0, 5.0, 20.0, 95.0])
#k1 = 0.08 # original
#k2 = 0.0015 # original
#y0 = np.array([10.0, 5.0, 50.0, 92.0]) # original

# sc4
a3 = 0.055
b3 = 0.05
s1 = 0.0005
s2 = -0.009
s3 = 0.001
s4 = 0.0
k1 = 0.02
k2 = 0.003
# y0 = np.array([10.0, 5.0, 1.0, 95.0])

# sc3
#a3 = 0.055
#b3 = 0.05
#s1 = 0.001
#s2 = -0.005
#s3 = 0.001
#s4 = 0.0
#k1 = 0.02
#k2 = 0.003
#y0 = np.array([10.0, 5.0, 1.0, 95.0])
#k1 = 0.023 # original
#k2 = 0.0023 # original

# diffusion parameters
# D1 = 1.e-7
# D2 = 10.0*D1

D1 = 1.e-6
D2 = D1*5.
D3 = D1*10.

# convection parameters
C1 = -2.0*1.e-3
C2 = 0.0
C3 = 0.0

convCoeff1 = (C1,)
convCoeff2 = (C2,)
convCoeff3 = (C3,)

# note: convection >= 0.1 is too high

# discretization parameters
L  = 1.
nx = 200

dx = L / float(nx)

# finalTime = 1000.0
# timeTics  = 2000

finalTime = 1000.0
timeTics  = int(finalTime * 2)

timeStep = finalTime / float(timeTics)
dt = timeStep

print("Initializing. NOTE: dx = %f, dt = %f" % (dx, dt))

# setup
mesh = Grid1D(dx=L / nx, nx=nx)

x = mesh.cellCenters[0]
# convCoeff = g*(x-L/2) * [[1.]]

u1eq = (b2/a2)**(1.0/g2)
u2eq = (b1/a1)**(1.0/g1)
u3eq = 0.0

print((u1eq,u2eq,u3eq))

u1 = CellVariable(name="OCs", mesh=mesh, value=u1eq, hasOld=True)
u2 = CellVariable(name="OBs", mesh=mesh, value=u2eq, hasOld=True)
u3 = CellVariable(name="CCs", mesh=mesh, value=u3eq, hasOld=True)

center = 0.3
radius = center/8.

initialOCs = 10.
initialOBs = 5.0
initialCCs = 50.0

u1.setValue(initialOCs, where=((center-radius < x) & (x < center+radius)))
u2.setValue(initialOBs, where=((center-radius < x) & (x < center+radius)))
# u3.setValue(initialCCs, where=((center-radius < x) & (x < center+radius)))

center += 4.*radius
u1.setValue(initialOCs, where=((center-radius < x) & (x < center+radius)))
u2.setValue(initialOBs, where=((center-radius < x) & (x < center+radius)))

u3.setValue(initialCCs, where=((center-radius/2. < x) & (x < center+radius/2.)))

center += 4.*radius
u1.setValue(initialOCs, where=((center-radius < x) & (x < center+radius)))
u2.setValue(initialOBs, where=((center-radius < x) & (x < center+radius)))
# u3.setValue(initialCCs, where=((center-9.*radius < x) & (x < center+radius)))

# epsi = 1
# u1.setValue(u1eq/50, where=((10 - 2*epsi) < x) & (x < (10 + 2*epsi)))
# u2.setValue(u2eq*20, where=((10 - epsi) < x) & (x < (10 + epsi)))

# BCs: v0(t,0) = 0, v0(t,1) = 1; v1(t,0) = 1, v1(t,1) = 0
u1.constrain(u1eq, mesh.facesLeft)
u1.constrain(u1eq, mesh.facesRight)

u2.constrain(u2eq, mesh.facesLeft)
u2.constrain(u2eq, mesh.facesRight)

u3.constrain(u3eq, mesh.facesLeft)
u3.constrain(u3eq, mesh.facesRight)

sourceCoeff1 = a1*u2**g1 + s1*u3
sourceCoeff2 = a2*u1**g2 + s2*u3
sourceCoeff3 = a3*(1.0 - u3/K) + s3*u1**g2 + s4*u2**g1

eq1 = (TransientTerm(var=u1)
       == DiffusionTerm(coeff=D1, var=u1)
        - ConvectionTerm(coeff=convCoeff1, var=u1)
        - ImplicitSourceTerm(coeff=b1, var=u1)
        + ImplicitSourceTerm(coeff=sourceCoeff1, var=u1))

eq2 = (TransientTerm(var=u2)
       == DiffusionTerm(coeff=D2, var=u2)
        - ConvectionTerm(coeff=convCoeff2, var=u2)
        - ImplicitSourceTerm(coeff=b2, var=u2)
        + ImplicitSourceTerm(coeff=sourceCoeff2, var=u2))

eq3 = (TransientTerm(var=u3)
       == DiffusionTerm(coeff=D3, var=u3)
        - ConvectionTerm(coeff=convCoeff3, var=u3)
        - ImplicitSourceTerm(coeff=b3, var=u3)
        + ImplicitSourceTerm(coeff=sourceCoeff3, var=u3))

# fig = plt.figure(figsize=(8,4))
# ax  = plt.axes()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))

#titleString = "Bone remodeling, D1=" + str(D1) + ", A1=" + str(convCoeff1)

eqn = eq1 & eq2 & eq3
# vi = MatplotlibViewer((u1, u2, u3), axes=ax, title=titleString)
v1 = MatplotlibViewer(u1, axes=ax1)
v2 = MatplotlibViewer(u2, axes=ax2)
v3 = MatplotlibViewer(u3, axes=ax3)

# timeStep = 1
# timeTics = 1000

actualTime = 0.
test_time  = [actualTime]
test_u1    = [u1.value.copy()]
test_u2    = [u2.value.copy()]
test_u3    = [u3.value.copy()]

with writer.saving(fig, "writer_simulation.mp4", dpi=200):
    for t in range(timeTics):
        u1.updateOld()
        u2.updateOld()
        u3.updateOld()

        res1 = res2 = res3 = 1e100
        ITER = 0

        while max(res1, res2, res3) > 0.01:
            print "Sweeping:", ITER
            ITER += 1
            res1 = eq1.sweep(var=u1, dt=timeStep)
            res2 = eq2.sweep(var=u2, dt=timeStep)
            res3 = eq3.sweep(var=u3, dt=timeStep)

        # eqn.solve(dt=timeStep)

        actualTime += timeStep
        titleString = r"Bone metastasis. Time $t$=" + str(actualTime)
        fig.suptitle(titleString)
	ax1.set_xlabel(r"Distance $x$", fontsize=16)
	ax1.set_ylabel(r"Density $u$", fontsize=16)
	ax2.set_xlabel(r"Distance $x$", fontsize=16)
	ax3.set_xlabel(r"Distance $x$", fontsize=16)
        test_time += [actualTime]
        test_u1   += [u1.value.copy()]
        test_u2   += [u2.value.copy()]
        test_u3   += [u3.value.copy()]

        # vi.plot()
        v1.plot()
        ax1.get_lines()[0].set_color("blue")
        ax1.get_legend().set_visible(False)

        v2.plot()
        ax2.get_lines()[0].set_color("orange")
        ax2.get_legend().set_visible(False)

        v3.plot()
        ax3.get_lines()[0].set_color("red")
        ax3.get_legend().set_visible(False)

        fig.tight_layout()
	fig.subplots_adjust(top=0.85)

        writer.grab_frame()

        if t % 10 == 0:
            print("Time Step #: %d; Time: %f" % (t, t*timeStep))

# raw_input("finite")

xAux = range(timeTics)
yAux = range(nx)

z1Aux = np.zeros((len(yAux),len(xAux)))
z2Aux = np.zeros((len(yAux),len(xAux)))
z3Aux = np.zeros((len(yAux),len(xAux)))

for xx in xAux:
    for yy in yAux:
        z1Aux[yy][xx] = test_u1[xx][yy]
        z2Aux[yy][xx] = test_u2[xx][yy]
        z3Aux[yy][xx] = test_u3[xx][yy]

Z1 = z1Aux
Z2 = z2Aux
Z3 = z3Aux

xAux = np.linspace(0., L, nx)
yAux = np.linspace(0., finalTime, timeTics)

Y, X = np.meshgrid(yAux, xAux)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
# raw_input("finite")

plt.figure(figsize=(9,3))
plt.subplot(131)
plt.pcolor(X, Y, Z1, cmap='Blues')
plt.xlabel(r"Distance $x$",fontsize=16)
plt.ylabel(r"Time $t$", fontsize=16)
plt.title("OCs", fontsize=16)
# plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()
# plt.show()

# plt.figure()
plt.subplot(132)
plt.pcolor(X, Y, Z2, cmap='Oranges')
plt.xlabel(r"Distance $x$",fontsize=16)
#plt.ylabel(r"Time $t$", fontsize=16)
plt.title("OBs", fontsize=16)
# plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()

plt.subplot(133)
plt.pcolor(X, Y, Z3, cmap='Reds')
plt.xlabel(r"Distance $x$",fontsize=16)
#plt.ylabel(r"Time $t$", fontsize=16)
plt.title("CCs", fontsize=16)
# plt.axis([x.min(), x.max(), y.min(), y.max()])
plt.colorbar()


plt.tight_layout()
plt.show()
plt.savefig("ocs-and-obs.png",dpi=200)

raw_input("finite")
