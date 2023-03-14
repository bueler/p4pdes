from firedrake import *
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1)

u = Function(V)
v = TestFunction(V)

x, y = SpatialCoordinate(mesh)
frhs = Function(V).interpolate(2.0*(x*(1.0-x) + y*(1.0-y)) + exp(x*(1.0-x)*y*(1.0-y)))

u = Function(V)
F = (inner(grad(u), grad(v)) + exp(u) * v - frhs * v) * dx

bdry_ids = (1, 2, 3, 4)
bc = DirichletBC(V, Constant(0.0), bdry_ids)

sp = {'snes_type': 'newtonls',
      'snes_linesearch_type': 'basic',
      'snes_monitor': None,
      'ksp_type': 'preonly',
      'pc_type': 'lu'}
solve(F == 0, u, solver_parameters=sp, bcs=bc)

uexact = Function(V).interpolate(x*(1.0-x)*y*(1.0-y)) # for comparison
File("bratu.pvd").write(u,uexact)
