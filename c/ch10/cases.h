#ifndef CASES_H_
#define CASES_H_

// -----------------------------------------------------------------------------
// CASE 0
// LINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN
// USE: trap.poly

PetscReal a_lin(PetscReal u, PetscReal x, PetscReal y) {
    return 1.0;
}

// manufactured from a_lin(), uexact_lin():
PetscReal f_lin(PetscReal u, PetscReal x, PetscReal y) {
    return 2.0 * x + 3.0 * y * y;
}

PetscReal uexact_lin(PetscReal x, PetscReal y) {
    const PetscReal y2 = y * y;
    return 1.0 - x * y2 - 0.25 * y2 * y2;
}

// just evaluate exact u on boundary point:
PetscReal gD_lin(PetscReal x, PetscReal y) {
    return uexact_lin(x,y);
}

PetscReal gN_lin(PetscReal x, PetscReal y) {
    return 0.0;
}


// -----------------------------------------------------------------------------
// CASE 1
// NONLINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN
// USE: trap.poly

PetscReal a_nonlin(PetscReal u, PetscReal x, PetscReal y) {
    return 1.0 + u * u;
}

// manufactured from a_nonlin(), uexact_lin()
PetscReal f_nonlin(PetscReal udrop, PetscReal x, PetscReal y) {
    const PetscReal y2 = y * y,
                    y4 = y2 * y2,
                    u = 1.0 - x * y2 - 0.25 * y4,
                    v = 2.0 * x * y + y * y2;
    return - 2.0 * y4 * u - 2.0 * u * v * v
           + (1.0 + u * u) * (2.0 * x + 3.0 * y2);
}

// uexact_nonlin = uexact_lin
// gD_nonlin = gD_lin
// gN_nonlin = gN_lin


// -----------------------------------------------------------------------------
// CASE 2
// LINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, NONHOMOGENEOUS NEUMANN
// USE: trapneu.poly

// a_linneu = a_lin
// f_linneu = f_lin
// uexact_linneu = uexact_lin
// gD_linneu = gD_lin

// only valid on line  y = 2 - x
PetscReal gN_linneu(PetscReal x, PetscReal y) {
    return - y * (4.0 - y + y * y * y) / sqrt(2.0);
}


// -----------------------------------------------------------------------------
// CASE 3
// LINEAR PROBLEM, HOMOGENEOUS DIRICHLET ONLY, SQUARE DOMAIN
// SAME PROBLEM AS SOLVED BY DEFAULT BY c/ch6/fish.c
// USE: genstructured.py to generate meshes

PetscReal a_square(PetscReal u, PetscReal x, PetscReal y) {
    return 1.0;
}

// manufactured from a_square(), uexact_square():
PetscReal f_square(PetscReal u, PetscReal x, PetscReal y) {
    return x * exp(y);  // note  f = - (u_xx + u_yy) = - u
}

PetscReal uexact_square(PetscReal x, PetscReal y) {
    return - x * exp(y);
}

// just evaluate exact u on boundary point:
PetscReal gD_square(PetscReal x, PetscReal y) {
    return uexact_square(x,y);
}

// gN_fcn() = NULL in square case; want seg fault if called


// -----------------------------------------------------------------------------
// CASE 4
// LINEAR PROBLEM, HOMOGENEOUS DIRICHLET ONLY; NO EXACT SOLN
// USE: koch.poly from kochmesh.py (or genstructured.py)

PetscReal a_koch(PetscReal u, PetscReal x, PetscReal y) {
    return 1.0;
}

PetscReal f_koch(PetscReal u, PetscReal x, PetscReal y) {
    return 2.0;
}

PetscReal gD_koch(PetscReal x, PetscReal y) {
    return 0.0;
}

#endif

