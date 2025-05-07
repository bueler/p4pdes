#!/usr/bin/env python3

# Read command-line options (in addition to PETSc solver options
# which use -s_ prefix; see below)
from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Use Firedrake's nonlinear solver for the Poisson problem
  -Laplace(u) = f        in the unit square
            u = g        on the boundary
Compare c/ch6/fish.c.  The prefix for PETSC solver options is 's_'.
Use -help for PETSc options and -fishhelp for options to fish.py.""",
    formatter_class=RawTextHelpFormatter,add_help=False)
parser.add_argument('-fishhelp', action='store_true', default=False,
                    help='print help for fish.py options and exit')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of grid points in x-direction')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of grid points in y-direction')
parser.add_argument('-o', metavar='NAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-k', type=int, default=1, metavar='K',
                    help='polynomial degree for elements')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral finite elements')
parser.add_argument('-refine', type=int, default=-1, metavar='X',
                    help='number of refinement levels (e.g. for GMG)')
args, passthroughoptions = parser.parse_known_args()
if args.fishhelp:  # -fishhelp is for help with fish.py
    parser.print_help()
    import sys
    sys.exit(0)

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc

# Create mesh, enabling GMG via refinement using hierarchy
mx, my = args.mx, args.my
mesh = UnitSquareMesh(mx-1, my-1, quadrilateral=args.quad)
if args.refine > 0:
    hierarchy = MeshHierarchy(mesh, args.refine)
    mesh = hierarchy[-1]     # the fine mesh
    mx, my = (mx-1) * 2**args.refine + 1, (my-1) * 2**args.refine + 1
x,y = SpatialCoordinate(mesh)
mesh.topology_dm.viewFromOptions('-dm_view')
# to print coordinates:  print(mesh.coordinates.dat.data)

# Define function space, right-hand side, and weak form.
W = FunctionSpace(mesh, 'Lagrange', degree=args.k)
f_rhs = Function(W).interpolate(x * exp(y))  # manufactured
u = Function(W)  # initialized to zero here
v = TestFunction(W)
F = (dot(grad(u), grad(v)) - f_rhs * v) * dx

# Define Dirichlet boundary conditions
g_bdry = Function(W).interpolate(- x * exp(y))  # = exact solution
bdry_ids = (1, 2, 3, 4)   # all four sides of boundary
bc = DirichletBC(W, g_bdry, bdry_ids)

# Solve system as though it is nonlinear:  F(u) = 0
solve(F == 0, u, bcs = [bc], options_prefix = 's',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'cg'})

# Print numerical error in L_infty and L_2 norm
elementstr = '%s_%d' % (['P','Q'][args.quad],args.k)
udiff = Function(W).interpolate(u - g_bdry)
with udiff.dat.vec_ro as vudiff:
    error_Linf = abs(vudiff).max()[1]
error_L2 = sqrt(assemble(dot(udiff, udiff) * dx))
PETSc.Sys.Print('done on %d x %d grid with %s elements:' \
      % (mx,my,elementstr))
PETSc.Sys.Print('  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e' \
      % (error_Linf,error_L2))

# Optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving solution to %s ...' % args.o)
    u.rename('u')
    VTKFile(args.o).write(u)
