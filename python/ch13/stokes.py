#!/usr/bin/env python3

#FIXME
# (1) option to allow discontinuous for pressure
# (2) put in analytical solution
# (3) show Moffat eddies in paraview-generated figure
#     finds 2nd eddy: ./stokes.py -i lidbox.msh -s_ksp_rtol 1.0e-11 -o lidbox4_21.pvd -refine 4
# (4) parallel runs working?

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Solve the linear Stokes problem for a lid-driven cavity.  Dirichlet velocity
conditions on all sides.  (The top lid has quadratic horizontal velocity.
Other sides have zero velocity.)  Mixed FE method, P^k x P^l or Q^k x Q^l;
defaults to Taylor-Hood P^2 x P^1.  An example of a saddle-point system.
The PETSc solver prefix is 's_'.""",
                    formatter_class=RawTextHelpFormatter)
parser.add_argument('-i', metavar='INNAME', type=str, default='',
                    help='input file for mesh in Gmsh format (.msh); ignors -mx,-my if given')
parser.add_argument('-lidscale', type=float, default=1.0, metavar='GAMMA',
                    help='scale for lid velocity (rightward is positive; default is 1.0)')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of grid points in x-direction (uniform case)')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of grid points in y-direction (uniform case)')
parser.add_argument('-mu', type=float, default=1.0, metavar='MU',
                    help='dynamic viscosity (default is 1.0)')
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

# read or create mesh, either from file or uniform, and enable GMG using hierarchy
if len(args.i) > 0:
    PETSc.Sys.Print('reading mesh from %s ...' % args.i)
    mesh = Mesh(args.i)
    meshstr = ''
    other = (41,)  # FIXME use str names 'lid','other'
    lid = (40,)
else:
    mx, my = args.mx, args.my
    mesh = UnitSquareMesh(mx-1, my-1, quadrilateral=args.quad)
    mx, my = (mx-1) * 2**args.refine + 1, (my-1) * 2**args.refine + 1
    meshstr = ' on %d x %d grid' % (mx,my)
    other = (1,2,3)
    lid = (4,)
if args.refine > 0:
    hierarchy = MeshHierarchy(mesh, args.refine)
    mesh = hierarchy[-1]     # the fine mesh
    if len(args.i) > 0:
        meshstr += ' with %d levels uniform refinement' % args.refine
x,y = SpatialCoordinate(mesh)
mesh._plex.viewFromOptions('-dm_view')

# define mixed finite elements (P^k-P^l or Q^k-Q^l)
V = VectorFunctionSpace(mesh, 'Lagrange', degree=args.uorder)
W = FunctionSpace(mesh, 'Lagrange', degree=args.porder)
Z = V * W

# define weak form
up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)
f = Constant((0.0, 0.0))  # no body force
F = (args.mu * inner(grad(u), grad(v)) - p * div(v) - div(u) * q - inner(f,v)) * dx

# boundary conditions are defined on the velocity space
noslip = Constant((0.0, 0.0))
lidtangent = interpolate(as_vector([args.lidscale * x * (1.0 - x),0.0]),V)
bc = [ DirichletBC(Z.sub(0), noslip, other),
       DirichletBC(Z.sub(0), lidtangent, lid) ]

# no boundary conditions on the pressure space therefore set nullspace
ns = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# describe job and then solve
uFEstr = '%s^%d' % (['P','Q'][args.quad],args.uorder)
pFEstr = '%s^%d' % (['P','Q'][args.quad],args.porder)
PETSc.Sys.Print('solving%s with %s x %s mixed elements ...' \
                % (meshstr,uFEstr,pFEstr))
if len(args.i) > 0:
    PETSc.Sys.Print('  mesh has %d elements (2-cells) and %d vertices' \
                    % (mesh.num_cells(),mesh.num_vertices()))
if mesh.comm.size > 1:
    PETSc.Sys.syncPrint('    rank %d owns %d elements (cells) and can access %d vertices' \
                        % (mesh.comm.rank,mesh.num_cells(),mesh.num_vertices()), comm=mesh.comm)
    PETSc.Sys.syncFlush(comm=mesh.comm)
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

# optionally save to .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving to %s ...' % args.o)
    u.rename('velocity')
    p.rename('pressure')
    File(args.o).write(u,p)

