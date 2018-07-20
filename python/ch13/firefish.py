#!/usr/bin/env python3

#FIXME (1) solve same 2D problem as c/ch6/fish.c
#      (2) use only nonlinear solver

"""
Use nonlinear solver (or linear solver with -linprob) for Poisson equation.
  -Laplace(u) = f        # in the unit square
            u = u_D      # on the boundary
where
  u_D = exp(x - y^2)
    f = exp(x - y^2) * (1 - 4 y^2)     # manufactured
"""

# numerical error and KSP its are the same for these:
#   for MM in 2 4 8 16 32; do ./poissonF.py -m $MM -s_snes_converged_reason -s_ksp_converged_reason -linprob; done
#   for MM in 2 4 8 16 32; do ./poissonF.py -m $MM -s_snes_converged_reason -s_ksp_converged_reason; done

import argparse
parser = argparse.ArgumentParser(description='Use nonlinear solver for Poisson equation.')
parser.add_argument('-linprob', action='store_true')
parser.add_argument('-m', type=int, default=8, metavar='M',
                    help='number of mesh points in each dimension')
args, unknown = parser.parse_known_args()

from firedrake import *

# Create mesh and define function space
mesh = UnitSquareMesh(args.m, args.m)
V = FunctionSpace(mesh, "CG", 1)       # FIXME changing either causes seg faults ... what's going on?

# Define boundary condition and RHS
x,y = SpatialCoordinate(mesh)
u_D = exp(x - y*y)
boundary_ids = (1, 2, 3, 4)
bc = DirichletBC(V, u_D, boundary_ids)
f = Function(V).interpolate(exp(x - y*y) * (1.0 - 4.0 * y * y))

# Define variational problem and solve either using linear or nonlinear
v = TestFunction(V)
u = Function(V)
sps = {'ksp_type': 'cg',
       'pc_type': 'gamg',
       'ksp_rtol': 1.0e-7,
       'snes_rtol': 1.0e-5}
if args.linprob:
    print('using linear solver ...')
    utry = TrialFunction(V)
    a = dot(grad(utry), grad(v)) * dx
    L = f*v * dx
    solve(a == L, u, bcs = [bc], options_prefix='s',
          solver_parameters = sps)
else:
    print('using nonlinear solver ...')
    Fnl = (dot(grad(u), grad(v)) - f*v) * dx
    u.interpolate(Constant(0.0, domain=mesh))   # initial iterate
    solve(Fnl == 0, u, bcs = [bc], options_prefix='s',
          solver_parameters = sps)

# Compute error in L_2, L_infty norm
u_ex = Function(V).interpolate(u_D)
error_Linf = max(abs(u.vector().array() - u_ex.vector().array()))
error_L2 = sqrt(assemble(dot(u - u_ex, u - u_ex) * dx))

print('done on %d x %d mesh; error_L2 = %.3e, error_Linf = %.3e' \
      % (args.m,args.m,error_L2,error_Linf))
print('saving solution to usoln.pvd ...')
File("usoln.pvd").write(u)

