#!/usr/bin/env python3

#FIXME (1) allow read of Gmsh generated mesh e.g. refinement in corners
#      (2) how to show DMPlex data structures for simple mesh?
#      (3) show FIXME vortices in corners

# A simple example of a saddle-point system.  We set up the problem as a
# lid-driven cavity.  Zero Dirichlet conditions on the bottom and side walls,
# with a constant right-ward velocity condition on the top lid.
# This example was originally generated from demos/matrix_free/stokes.py.rst
# in firedrake source.

from firedrake import *

# define a mesh
N = 64
M = UnitSquareMesh(N, N)

# define Taylor-Hood elements
V = VectorFunctionSpace(M, "CG", 2)
W = FunctionSpace(M, "CG", 1)
Z = V * W
u,p = TrialFunctions(Z)
v,q = TestFunctions(Z)

# define weak form
a = (inner(grad(u), grad(v)) - p * div(v) - div(u) * q) * dx
# above line corrects bug in firedrake example which causes non-symmetry:
#a = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q) * dx  
f = Constant((0, 0))  # no body force
L = inner(f, v) * dx

# boundary conditions are defined on the velocity space
noslip = Constant((0, 0))
lid_tangent = Constant((1, 0))
othersides = (1,2,3)   # boundary indices
top = (4,)
bcs = [ DirichletBC(Z.sub(0), noslip, othersides),
        DirichletBC(Z.sub(0), lid_tangent, top) ]

# Since we have no boundary conditions on the pressure space it is only defined
# up to a constant.  We remove this component of the solution in the solver by
# providing the appropriate nullspace.
nullspace = MixedVectorSpaceBasis(Z,
                                  [Z.sub(0), VectorSpaceBasis(constant=True)])

# solve
up = Function(Z)   # put solution here
solve(a == L, up, bcs=bcs, nullspace=nullspace,
      options_prefix='stk',
      solver_parameters={#"ksp_view": True,
                         "ksp_converged_reason": True,
                         "ksp_monitor": True,
                         "ksp_type": "fgmres",  # or "gmres" or "minres"
                         "pc_type": "fieldsplit",
                         "pc_fieldsplit_type": "schur",
                         "pc_fieldsplit_schur_factorization_type": "full",  # or "diag"
                         "fieldsplit_0_ksp_type": "preonly",
                         "fieldsplit_0_pc_type": "lu",
                         #"fieldsplit_0_ksp_converged_reason": True,
                         "fieldsplit_1_ksp_converged_reason": True,
                         "fieldsplit_1_ksp_type": "gmres",
                         "fieldsplit_1_pc_type": "none"})

# note default solver already has -snes_type ksponly for this linear problem
# ALSO can add these using -stk_ prefix:
#    "ksp_type": "minres", "pc_type": "jacobi",
#    "mat_type": "aij"
#    "mat_type": "aij", "ksp_type": "preonly", "pc_type": "svd",  # fully-direct solver
#    "ksp_view_mat": ":foo.m:ascii_matlab"

# Write the solution to a file for visualisation with paraview.  We split the
# solution function, and name velocity and pressure parts.
u,p = up.split()
u.rename("velocity")
p.rename("pressure")
File("stokes.pvd").write(u,p)

