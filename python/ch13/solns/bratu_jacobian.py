from firedrake import *
from firedrake.petsc import PETSc

withJ = False   # optionally provide the Jacobian to solve

mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

x, y = SpatialCoordinate(mesh)
frhs = Function(V).interpolate(2.0*(x*(1.0-x) + y*(1.0-y)) + exp(x*(1.0-x)*y*(1.0-y)))

u = Function(V)
v = TestFunction(V)
F = (inner(grad(u), grad(v)) + exp(u) * v - frhs * v) * dx

bc = DirichletBC(V, Constant(0.0), (1, 2, 3, 4))
sp = {'snes_type': 'newtonls',
      'snes_linesearch_type': 'basic',
      'snes_monitor': None,
      'ksp_type': 'preonly',
      'pc_type': 'lu'}

if withJ:
    # this just (partly) exposes what the internals do, generating a Jacobian
    # by symbolic differentiation in UFL
    w = TrialFunction(V)
    J = derivative(F, u, du=w)
    solve(F == 0, u, solver_parameters=sp, bcs=bc, J=J)
else:
    solve(F == 0, u, solver_parameters=sp, bcs=bc)

uexact = Function(V).interpolate(x*(1.0-x)*y*(1.0-y)) # for comparison
u.rename('u (numerical soln)')
uexact.rename('uexact (exact soln)')
File("bratu.pvd").write(u,uexact)

diffu = Function(V).interpolate(u - uexact)
error_L2 = sqrt(assemble(dot(diffu, diffu) * dx))
PETSc.Sys.Print('done on mesh with %d nodes:  |u-uexact|_2 = %.3e' \
      % (len(u.dat.data), error_L2))
