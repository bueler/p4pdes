#!/usr/bin/env python3

helpstr = """
Use nonlinear solver for Poisson equation.
  -Laplace(u) = f        in the unit square
            u = g        on the boundary
using polynomial manufactured exact solution.  PETSc solver prefix 's_'.
"""

import argparse
parser = argparse.ArgumentParser(description=helpstr,
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

from firedrake import *

# Create mesh and define function space
mesh = UnitSquareMesh(args.mx, args.my, quadrilateral=args.quad)
x,y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, 'Lagrange', args.order)
if args.quad:
    elementstr = 'Q^%d' % args.order
else:
    elementstr = 'P^%d' % args.order

# Define exact solution and right-hand side
u_exact = Function(V).interpolate(x*x * (1.0 - x*x) * y*y *(y*y - 1.0))
f_rhs = - Function(V).interpolate(2.0 * (1.0 - 6.0 * x*x) * y*y * (y*y - 1.0)
                                  + x*x * (1.0 - x*x) * 2.0 * (6.0 * y*y - 1.0))

# Define boundary conditions and weak form
boundary_ids = (1, 2, 3, 4)
bc = DirichletBC(V, u_exact, boundary_ids)
v = TestFunction(V)
u = Function(V)
F = (dot(grad(u), grad(v)) - f_rhs*v) * dx

# Set-up solver
u.interpolate(Constant(0.0, domain=mesh))   # initial iterate is zero
solve(F == 0, u, bcs = [bc], options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'cg'})

# Compute error in L_infty and L_2 norm
error_Linf = max(abs(u.vector().array() - u_exact.vector().array()))
error_L2 = sqrt(assemble(dot(u - u_exact, u - u_exact) * dx))
print('done on %d x %d mesh with %s elements:' % (args.mx,args.my,elementstr))
print('  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e' % (error_Linf,error_L2))

# Save to file viewable with Paraview
if len(args.o) > 0:
    print('saving solution to %s ...' % args.o)
    File(args.o).write(u)

