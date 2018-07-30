#!/usr/bin/env python3

# generated from firedrake/demos/matrix_free/stokes.py.rst

#FIXME (1) check that read of Gmsh file works
#      (2) demo refinement in corners
#      (3) add computation of stream function
#      (4) show Moffat eddies

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Solve the linear Stokes problem for a lid-driven cavity.  Dirichlet velocity
conditions on all sides.  (The top lid has constant horizontal velocity.
Other sides have zero velocity.)  Mixed FE method, P^k x P^l or Q^k x Q^l;
defaults to Taylor-Hood P^2 x P^1.  An example of a saddle-point system.
The PETSc solver prefix is 's_'.""",
                    formatter_class=RawTextHelpFormatter)
parser.add_argument('-i', metavar='INNAME', type=str, default='',
                    help='input file for mesh in Gmsh format (.msh); ignors -mx,-my if given')
parser.add_argument('-lidvelocity', type=float, default=1.0, metavar='V0',
                    help='lid velocity (rightward is positive; default is 1.0)')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of grid points in x-direction (uniform case)')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of grid points in y-direction (uniform case)')
parser.add_argument('-o', metavar='OUTNAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-uorder', type=int, default=2, metavar='K',
                    help='polynomial degree for velocity')
parser.add_argument('-porder', type=int, default=1, metavar='L',
                    help='polynomial degree for pressure')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral finite elements')
parser.add_argument('-refine', type=int, default=0, metavar='X',
                    help='number of refinement levels (e.g. for GMG)')
parser.add_argument('-show_norms', action='store_true', default=False,
                    help='print solution norms (useful for testing)')
args, unknown = parser.parse_known_args()

# create mesh, either from file or uniform, enabling GMG using hierarchy
if len(args.i) > 0:
    # ignoring -mx,-my if given
    mesh = Mesh(args.i)
    meshstr = 'Gmsh mesh from %s' % args.i
else:
    mx, my = args.mx, args.my
    mesh = UnitSquareMesh(mx-1, my-1, quadrilateral=args.quad)
    mx, my = (mx-1) * 2**args.refine + 1, (my-1) * 2**args.refine + 1
    meshstr = '%d x %d grid' % (mx,my)
if args.refine > 0:
    hierarchy = MeshHierarchy(mesh, args.refine)
    mesh = hierarchy[-1]     # the fine mesh
    if len(args.i) > 0:
        meshstr += ' with %d levels uniform refinement' % args.refine
x,y = SpatialCoordinate(mesh)
mesh._plex.viewFromOptions('-dm_view')

# define Taylor-Hood elements (P^k-P^l or Q^k-Q^l)
V = VectorFunctionSpace(mesh, 'Lagrange', degree=args.uorder)
W = FunctionSpace(mesh, 'Lagrange', degree=args.porder)
Z = V * W

# define weak form
up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)
f = Constant((0.0, 0.0))  # no body force
F = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(f,v)) * dx
# FIXME compare bug in firedrake example which causes non-symmetry:
#a = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q) * dx

# boundary conditions are defined on the velocity space
noslip = Constant((0.0, 0.0))
lid_tangent = Constant((args.lidvelocity, 0.0))
othersides = (1,2,3)   # boundary indices from UnitSquareMesh
top = (4,)
bc = [ DirichletBC(Z.sub(0), noslip, othersides),
       DirichletBC(Z.sub(0), lid_tangent, top) ]

# no boundary conditions on the pressure space therefore set nullspace
ns = MixedVectorSpaceBasis(Z,[Z.sub(0), VectorSpaceBasis(constant=True)])

# solve
uFEstr = '%s^%d' % (['P','Q'][args.quad],args.uorder)
pFEstr = '%s^%d' % (['P','Q'][args.quad],args.porder)
print('solving on %s with %s x %s mixed elements ...' \
      % (meshstr,uFEstr,pFEstr))
solve(F == 0, up, bcs=bc, nullspace=ns, options_prefix='s',
      solver_parameters={'snes_type': 'ksponly',
                         'ksp_type': 'fgmres',  # or minres, gmres
                         'pc_type': 'fieldsplit',
                         'pc_fieldsplit_type': 'schur',
                         'pc_fieldsplit_schur_factorization_type': 'full',  # or diag
                         'fieldsplit_0_ksp_type': 'preonly', # FIXME why not CG+GMG
                         'fieldsplit_0_pc_type': 'lu',
                         'fieldsplit_1_ksp_type': 'gmres',
                         'fieldsplit_1_pc_type': 'none'})  # FIXME why?
u,p = up.split()

# ALSO can add these using -s_ prefix:
#    "ksp_type": "minres", "pc_type": "jacobi",
#    "mat_type": "aij"
#    "mat_type": "aij", "ksp_type": "preonly", "pc_type": "svd",  # fully-direct solver
#    "ksp_view_mat": ":foo.m:ascii_matlab"
#    "fieldsplit_0_ksp_converged_reason": True
#    "fieldsplit_1_ksp_converged_reason": True

# optionally print solution norms
if args.show_norms:
    uL2 = sqrt(assemble(dot(u, u) * dx))
    pL2 = sqrt(assemble(dot(p, p) * dx))
    PETSc.Sys.Print('  norms: |u|_h = %.3e, |p|_h = %.3e' % (uL2, pL2))

# optionally save to a .pvd file viewable with Paraview
if len(args.o) > 0:
    u.rename('velocity')
    p.rename('pressure')
    File(args.o).write(u,p)

