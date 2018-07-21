#!/usr/bin/env python3

import argparse
from firedrake import *

parser = argparse.ArgumentParser(description="""
Use Firedrake's nonlinear solver for the Poisson problem
  -Laplace(u) = f        in the unit square
            u = g        on the boundary
Uses same manufactured exact solution as c/ch6/fish.c.
The PETSc solver prefix is 's_'.""",
                    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of mesh points in x-direction')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of mesh points in y-direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-order', type=int, default=1, metavar='X',
                    help='polynomial degree for elements')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral finite elements')
args, unknown = parser.parse_known_args()

# Create mesh and define function space
mesh = UnitSquareMesh(args.mx-1, args.my-1, quadrilateral=args.quad)
x,y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'Lagrange', args.order)

# Define exact solution and right-hand side
u_exact = Function(V).interpolate(- x * exp(y))
f_rhs = Function(V).interpolate(x * exp(y))  # note f = -(u_xx + u_yy)

# Define boundary conditions and weak form
boundary_ids = (1, 2, 3, 4)
bc = DirichletBC(V, u_exact, boundary_ids)
v = TestFunction(V)
u = Function(V)
F = (dot(grad(u), grad(v)) - f_rhs*v) * dx

# Set-up solver
u.interpolate(Constant(0.0, domain=mesh))   # initial iterate is zero
solve(F == 0, u,
      bcs = [bc], options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'cg'})

# Compute error in L_infty and L_2 norm
elementstr = '%s^%d' % (['P','Q'][args.quad],args.order)
error_Linf = max(abs(u.vector().array() - u_exact.vector().array()))
error_L2 = sqrt(assemble(dot(u - u_exact, u - u_exact) * dx))
print('done on %d x %d point 2D grid (mesh) with %s elements:' \
      % (args.mx,args.my,elementstr))
print('  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e' \
      % (error_Linf,error_L2))

# Optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    print('saving solution to %s ...' % args.o)
    u.rename("u")
    File(args.o).write(u)

