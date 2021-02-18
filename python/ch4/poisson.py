# found at https://github.com/pyHPC/pyhpc-tutorial/blob/master/notebooks/Appendix_02_PETSc4Py.ipynb
# seems to run fine

try: range = xrange
except: pass

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

class Poisson2D(object):

    def __init__(self, da):
        assert da.getDim() == 2
        self.da = da
        self.localX  = da.createLocalVec()

    def formRHS(self, B):
        b = self.da.getVecArray(B)
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                b[i, j] = 1*hx*hy

    def mult(self, mat, X, Y):
        #
        self.da.globalToLocal(X, self.localX)
        x = self.da.getVecArray(self.localX)
        y = self.da.getVecArray(Y)
        #
        mx, my = self.da.getSizes()
        hx, hy = [1.0/m for m in [mx, my]]
        (xs, xe), (ys, ye) = self.da.getRanges()
        for j in range(ys, ye):
            for i in range(xs, xe):
                u = x[i, j] # center
                u_e = u_w = u_n = u_s = 0
                if i > 0:    u_w = x[i-1, j] # west
                if i < mx-1: u_e = x[i+1, j] # east
                if j > 0:    u_s = x[i, j-1] # south
                if j < ny-1: u_n = x[i, j+1] # north
                u_xx = (-u_e + 2*u - u_w)*hy/hx
                u_yy = (-u_n + 2*u - u_s)*hx/hy
                y[i, j] = u_xx + u_yy

OptDB = PETSc.Options()

n  = OptDB.getInt('n', 16)
nx = OptDB.getInt('nx', n)
ny = OptDB.getInt('ny', n)

da = PETSc.DA().create([nx, ny], stencil_width=1)
pde = Poisson2D(da)

x = da.createGlobalVec()
b = da.createGlobalVec()
# A = da.createMat('python')
A = PETSc.Mat().createPython(
    [x.getSizes(), b.getSizes()], comm=da.comm)
A.setPythonContext(pde)
A.setUp()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
pc = ksp.getPC()
pc.setType('none')
ksp.setFromOptions()

pde.formRHS(b)
ksp.solve(b, x)

u = da.createNaturalVec()
da.globalToNatural(x, u)

from matplotlib import pylab
from numpy import mgrid

X, Y =  mgrid[0:1:1j*nx,0:1:1j*ny]
Z = x[...].reshape(nx,ny, order='F')
pylab.figure()
pylab.contourf(X,Y,Z)
pylab.plot(X.ravel(),Y.ravel(),'.k')
pylab.axis('equal')
pylab.colorbar()
pylab.show()

