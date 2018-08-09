#!/usr/bin/env python3

#FIXME
# * generate small matrix and talk/check/show some details:
#      ./stokes.py -analytical -mx 2 -my 2 -s_mat_type aij -s_ksp_view_mat :foo.m:ascii_matlab
# * showing fixed number of iterations independent of LEV:
#      ./stokes.py -s_ksp_converged_reason -s_fieldsplit_1_ksp_converged_reason -refine LEV
# * consider detuning S block (-s_fieldsplit_1_ksp_rtol 1.0e-3)
# * show Moffat eddies in paraview-generated figure
# * finds 2nd eddy:
#      ./stokes.py -i lidbox.msh -dm_view -s_ksp_monitor -s_ksp_rtol 1.0e-10 -s_ksp_type fgmres -o lidbox3_21.pvd -refine 3

from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Solve a linear Stokes problem in 2D.  Three problem cases:
  1. (default) Lid-driven cavity with quadratic velocity on lid and
     Dirichlet conditions on all sides.  Null space = {constants}.
  2. (-nobase) Same but with stress free condition on bottom; null space = {0}.
  3. (-analytical) Analytical exact solution from Logg et al (2012).
Uses mixed FE method, either Taylor-Hood family (P^k x P^l or Q^k x Q^l)
or CD with discontinuous pressure; defaults to P^2 x P^1.  Uses either
uniform mesh or reads mesh.  Serves as an example of a saddle-point system.
The solver prefix for PETSc options is 's_'.""",
                    formatter_class=RawTextHelpFormatter)
parser.add_argument('-analytical', action='store_true', default=False,
                    help='use problem with exact solution')
parser.add_argument('-dpressure', action='store_true', default=False,
                    help='use discontinuous-Galerkin finite elements for pressure')
parser.add_argument('-i', metavar='INNAME', type=str, default='',
                    help='input file for mesh in Gmsh format (.msh)')
parser.add_argument('-lidscale', type=float, default=1.0, metavar='X',
                    help='scale for lid velocity (rightward positive; default=1.0)')
parser.add_argument('-mx', type=int, default=3, metavar='MX',
                    help='number of grid points in x-direction (uniform case)')
parser.add_argument('-my', type=int, default=3, metavar='MY',
                    help='number of grid points in y-direction (uniform case)')
parser.add_argument('-mu', type=float, default=1.0, metavar='MU',
                    help='dynamic viscosity (default=1.0)')
parser.add_argument('-nobase', action='store_true', default=False,
                    help='use problem with stress-free boundary condition on base')
parser.add_argument('-o', metavar='OUTNAME', type=str, default='',
                    help='output file name ending with .pvd')
parser.add_argument('-porder', type=int, default=1, metavar='L',
                    help='polynomial degree for pressure (default=1)')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral finite elements')
parser.add_argument('-refine', type=int, default=0, metavar='R',
                    help='number of refinement levels (e.g. for GMG)')
parser.add_argument('-rho', type=float, default=1.0, metavar='RHO',
                    help='constant fluid density (default=1.0)')
parser.add_argument('-show_norms', action='store_true', default=False,
                    help='print solution norms (useful for testing)')
parser.add_argument('-uorder', type=int, default=2, metavar='K',
                    help='polynomial degree for velocity (default=2)')
args, unknown = parser.parse_known_args()

# read general mesh or create uniform mesh
if len(args.i) > 0:
    assert (not args.analytical) and (not args.nobase)
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
    # boundary i.d.s:    4
    #                   ---
    #                 1 | | 2
    #                   ---
    #                    3
    if args.nobase:
        other = (1,2)
    else:
        other = (1,2,3)
    lid = (4,)

# enable GMG using hierarchy
if args.refine > 0:
    hierarchy = MeshHierarchy(mesh, args.refine)
    mesh = hierarchy[-1]     # the fine mesh
    if len(args.i) > 0:
        meshstr += ' with %d levels uniform refinement' % args.refine
x,y = SpatialCoordinate(mesh)
mesh._plex.viewFromOptions('-dm_view')

# define mixed finite elements (P^k-P^l or Q^k-Q^l)
V = VectorFunctionSpace(mesh, 'CG', degree=args.uorder)
if args.dpressure:
    W = FunctionSpace(mesh, 'DG', degree=args.porder)
    mixedname = 'CD'
else:
    W = FunctionSpace(mesh, 'CG', degree=args.porder)
    mixedname = 'Taylor-Hood'
Z = V * W

# define body force and Dir. boundary condition (on velocity only)
# note: UFL as_vector() takes UFL expressions and combines
if args.analytical:
    assert (len(args.i) == 0)  # require UnitSquareMesh
    assert (args.mu == 1.0 and args.rho == 1.0)
    g = as_vector([ 28.0 * pi*pi * sin(4.0*pi*x) * cos(4.0*pi*y), \
                   -36.0 * pi*pi * cos(4.0*pi*x) * sin(4.0*pi*y)])
    u_12 = Function(V).interpolate(as_vector([0.0,-sin(4.0*pi*y)]))
    u_34 = Function(V).interpolate(as_vector([sin(4.0*pi*x),0.0]))
    bc = [ DirichletBC(Z.sub(0), u_12, (1,2)),
           DirichletBC(Z.sub(0), u_34, (3,4)) ]
else:
    g = Constant((0.0, 0.0))  # no body force in lid-driven cavity
    u_noslip = Constant((0.0, 0.0))
    xlid = args.lidscale * x * (1.0 - x)
    u_lid = Function(V).interpolate(as_vector([xlid,0.0]))
    bc = [ DirichletBC(Z.sub(0), u_noslip, other),
           DirichletBC(Z.sub(0), u_lid,    lid)   ]

# define weak form
up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)
F = (args.mu * inner(grad(u), grad(v)) - p * div(v) - div(u) * q \
     - inner(args.rho * g,v)) * dx

# describe method
uFEstr = '%s^%d' % (['P','Q'][args.quad],args.uorder)
pFEstr = '%s^%d' % (['P','Q'][args.quad],args.porder)
PETSc.Sys.Print('solving%s with %s x %s %s elements ...' \
                % (meshstr,uFEstr,pFEstr,mixedname))

# solve
sparams = {'snes_type': 'ksponly',
           'ksp_type': 'minres',
           'pc_type': 'fieldsplit',
           'pc_fieldsplit_type': 'schur',
           'pc_fieldsplit_schur_factorization_type': 'diag',
           'fieldsplit_0_ksp_type': 'preonly', # -s_fieldsplit_0_ksp_converged_reason shows repeated application of KSP ... why?
           'fieldsplit_0_pc_type': 'mg',
           'fieldsplit_1_ksp_type': 'cg',  # why can https://www.firedrakeproject.org/demos/geometric_multigrid.py.html use preonly here?
           'fieldsplit_1_pc_type': 'jacobi'}
if not args.nobase:
    # Dirichlet-only boundary conds on velocity therefore set nullspace
    ns = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
    solve(F == 0, up, bcs=bc, nullspace=ns, options_prefix='s',
          solver_parameters=sparams)
else:
    solve(F == 0, up, bcs=bc, options_prefix='s',
          solver_parameters=sparams)
u,p = up.split()

# ALSO can add these using -s_ prefix:
#    "mat_type": "aij", "ksp_type": "preonly", "pc_type": "svd",  # fully-direct solver
#    "mat_type": "aij", "ksp_view_mat": ":foo.m:ascii_matlab"


# get numerical error if possible
if args.analytical:
    xexact = sin(4.0*pi*x) * cos(4.0*pi*y)
    yexact = -cos(4.0*pi*x) * sin(4.0*pi*y)
    u_exact = Function(V).interpolate(as_vector([xexact,yexact]))
    p_exact = Function(W).interpolate(pi * cos(4.0*pi*x) * cos(4.0*pi*y))
    uerr = sqrt(assemble(dot(u - u_exact, u - u_exact) * dx))
    perr = sqrt(assemble(dot(p - p_exact, p - p_exact) * dx))
    PETSc.Sys.Print('  numerical errors: |u-uexact|_h = %.3e, |p-pexact|_h = %.3e' % (uerr, perr))

# optionally print solution norms
if args.show_norms:
    uL2 = sqrt(assemble(dot(u, u) * dx))
    pL2 = sqrt(assemble(dot(p, p) * dx))
    PETSc.Sys.Print('  norms: |u|_h = %.5e, |p|_h = %.5e' % (uL2, pL2))

# optionally save to .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving to %s ...' % args.o)
    u.rename('velocity')
    p.rename('pressure')
    File(args.o).write(u,p)

