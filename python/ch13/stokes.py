#!/usr/bin/env python3

# * generate small matrix and talk/check/show some details (?):
#      ./stokes.py -analytical -s_ksp_type minres -s_pc_type none -mx 2 -my 2 -s_mat_type aij -s_ksp_view_mat :foo.m:ascii_matlab

# * showing fixed number of iterations independent of LEV and clear evidence of
#   optimality; note ksp_view_mat shows MatNest so get n_p,n_u;
#      for LEV in 1 2 3 4 5 6 7 8; do
#          timer ./stokes.py -s_ksp_converged_reason -pcpackage schur_lower_gmg -refine $LEV
#      done
#   + uses default -s_ksp_type gmres
#   + level 8 is 513x513 grid
#   + add for flops: -log_view |grep "Flop:  "
#   + add -s_ksp_view_mat to show MatNest to get n_p,n_u
#   + add -analytical -s_ksp_rtol 1.0e-10 to show convergence
#   + can go to level 9 (1025x1025 grid) which uses more than 16 Gb:

# * show Moffat eddies with paraview-generated streamlines; resolves 3rd eddy!:
#      ./lidbox.py lidbox.geo
#      gmsh -2 lidbox.geo -o lidbox.msh
#      ./stokes.py -i lidbox.msh -show_norms -dm_view -s_ksp_rtol 1.0e-12 -s_ksp_converged_reason -pcpackage schur_lower_gmg -refine 5 -o lid5.pvd
#   + uses default -s_ksp_type gmres
#   + -refine 6 works too and uses ~30Gb memory but does not get 4th eddy

# solver notes:
#   -s_pc_fieldsplit_type schur
#       is the ONLY viable fieldsplit type;
#       additive,multiplicative,symmetric_multiplicative all fail because
#       diagonal pressure block is identically zero and non-invertible
#   -s_pc_fieldsplit_schur_factorization_type diag
#       note main Murphy et al 2000 theorem applies to MINRES+(this schur diag);
#       the default -s_pc_fieldsplit_schur_scale -1.0 *is* what we want
#   with -pcpackage schur_lower_gmg_nomass, option -s_fieldsplit_0_ksp_converged_reason
#       shows extra applications of fieldsplit_0 KSP because it is applying (A00)^-1
#       in applying Schur factor

import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from firedrake import *
from firedrake.petsc import PETSc

parser = ArgumentParser(description="""
Solve a linear Stokes problem in 2D.  Three problem cases:
  1. (default) Lid-driven cavity with quadratic velocity on lid and
     Dirichlet conditions on all sides.  Null space = {constant pressure}.
  2. (-analytical) Analytical exact solution from Logg et al (2012).
  3. (-nobase) Same but with stress free condition on bottom; null space = {0}.
Uses mixed FE method, either Taylor-Hood family (P^k x P^l or Q^k x Q^l)
or CD with discontinuous pressure; defaults to P^2 x P^1.  Uses either
uniform mesh or reads mesh.  Serves as an example of a saddle-point system.
See code for PC packages:
  -pcpackage directlu|schur_lower_gmg|schur_lower_gmg_nomass|schur_diag_gmg
The prefix for PETSC solver options is 's_'.""",
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
parser.add_argument('-pcpackage', metavar='PKG', default='',
                    help='choose a PC solver package: see text for options')
parser.add_argument('-pdegree', type=int, default=1, metavar='L',
                    help='polynomial degree for pressure (default=1)')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral finite elements')
parser.add_argument('-refine', type=int, default=0, metavar='R',
                    help='number of refinement levels (e.g. for GMG)')
parser.add_argument('-verbose', action='store_true', default=False,
                    help='print solver and solution norms (useful for testing)')
parser.add_argument('-udegree', type=int, default=2, metavar='K',
                    help='polynomial degree for velocity (default=2)')
args, unknown = parser.parse_known_args()
assert not (args.analytical and args.nobase), 'conflict in problem choice options'

# read Gmsh mesh or create uniform mesh
if len(args.i) > 0:
    assert (not args.analytical), 'Gmsh file not allowed for -analytical problem'
    assert (not args.nobase), 'Gmsh file not allowed for -nobase problem'
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
V = VectorFunctionSpace(mesh, 'CG', degree=args.udegree)
if args.dpressure:
    W = FunctionSpace(mesh, 'DG', degree=args.pdegree)
    mixedname = 'CD'
else:
    W = FunctionSpace(mesh, 'CG', degree=args.pdegree)
    mixedname = 'Taylor-Hood'
Z = V * W

# define body force and Dir. boundary condition (on velocity only)
# note: UFL as_vector() takes UFL expressions and combines
if args.analytical:
    assert (len(args.i) == 0)  # require UnitSquareMesh
    assert (args.mu == 1.0)
    f_body = as_vector([ 28.0 * pi*pi * sin(4.0*pi*x) * cos(4.0*pi*y), \
                       -36.0 * pi*pi * cos(4.0*pi*x) * sin(4.0*pi*y)])
    u_12 = Function(V).interpolate(as_vector([0.0,-sin(4.0*pi*y)]))
    u_34 = Function(V).interpolate(as_vector([sin(4.0*pi*x),0.0]))
    bc = [ DirichletBC(Z.sub(0), u_12, (1,2)),
           DirichletBC(Z.sub(0), u_34, (3,4)) ]
else:
    f_body = Constant((0.0, 0.0))  # no body force in lid-driven cavity
    u_noslip = Constant((0.0, 0.0))
    ux_lid = args.lidscale * x * (1.0 - x)
    u_lid = Function(V).interpolate(as_vector([ux_lid,0.0]))
    bc = [ DirichletBC(Z.sub(0), u_noslip, other),
           DirichletBC(Z.sub(0), u_lid,    lid)   ]

# define weak form
up = Function(Z)
u,p = split(up)
v,q = TestFunctions(Z)
F = (args.mu * inner(grad(u), grad(v)) - p * div(v) - div(u) * q \
     - inner(f_body,v)) * dx

# reference for preconditioning Schur block using viscosity-weighted mass matrix: https://www.firedrakeproject.org/demos/geometric_multigrid.py.html
class Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = (1.0/args.mu) * inner(test, trial)*dx
        bcs = None
        return (a, bcs)

# solver packages; reference: https://www.firedrakeproject.org/demos/geometric_multigrid.py.html
pars = {'directlu':         # LU direct solver (serial only)
           {'pmat_type': 'aij',
            'pc_type': 'lu',
            'pc_factor_shift_type': 'inblocks'},
        'schur_lower_gmg':  # Schur(lower)+GMG with mass-matrix PC on Schur; use gmres
           {'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'lower',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'mg',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_1_pc_type': 'python',
            'fieldsplit_1_pc_python_type': '__main__.Mass',
            'fieldsplit_1_aux_pc_type': 'bjacobi',
            'fieldsplit_1_aux_sub_pc_type': 'icc'},
        'schur_lower_gmg_nomass':  # Schur(lower)+GMG w/o mass-matrix PC; use fgmres
           {'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'lower',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'mg',
            'fieldsplit_1_ksp_type': 'cg',
            'fieldsplit_1_pc_type': 'jacobi'},
        'schur_diag_gmg': # Schur(diag)+GMG with mass-matrix PC; use gmres
           # FIXME why doesn't this work with minres?
           {'pc_type': 'fieldsplit',
            'pc_fieldsplit_type': 'schur',
            'pc_fieldsplit_schur_fact_type': 'diag',
            'fieldsplit_0_ksp_type': 'preonly',
            'fieldsplit_0_pc_type': 'mg',
            'fieldsplit_1_ksp_type': 'preonly',
            'fieldsplit_1_pc_type': 'python',
            'fieldsplit_1_pc_python_type': '__main__.Mass',
            'fieldsplit_1_aux_pc_type': 'bjacobi',
            'fieldsplit_1_aux_sub_pc_type': 'icc'},
       }

sparams = {}
if len(args.pcpackage) > 0:
    try:
        sparams.update(pars[args.pcpackage])
    except KeyError:
        print('ERROR: invalid solver package choice')
        print('       choices are %s' % list(pars.keys()))
        sys.exit(1)
    PETSc.Sys.Print('PC package %s' % args.pcpackage)
sparams.update({'snes_type': 'ksponly'})  # applies to all

# optionally show solver dictionary
if args.verbose:
    PETSc.Sys.Print("  ", sparams)

# describe method
uFEstr = '%s^%d' % (['P','Q'][args.quad],args.udegree)
pFEstr = '%s^%d' % (['P','Q'][args.quad],args.pdegree)
PETSc.Sys.Print('solving%s with %s x %s %s elements ...' \
                % (meshstr,uFEstr,pFEstr,mixedname))

# actually solve
uu, pp = TrialFunctions(Z)
if args.nobase:
    ns = None
else:
    # Dirichlet-only boundary conds on velocity therefore set nullspace to constant pressure
    ns = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
solve(F == 0, up, bcs=bc, nullspace=ns,
      options_prefix='s', solver_parameters=sparams)
u,p = up.split()

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
if args.verbose:
    uL2 = sqrt(assemble(dot(u, u) * dx))
    pL2 = sqrt(assemble(dot(p, p) * dx))
    PETSc.Sys.Print('solution norms: |u|_h = %.5e, |p|_h = %.5e' % (uL2, pL2))

# optionally save to .pvd file viewable with Paraview
if len(args.o) > 0:
    PETSc.Sys.Print('saving to %s ...' % args.o)
    u.rename('velocity')
    p.rename('pressure')
    File(args.o).write(u,p)

