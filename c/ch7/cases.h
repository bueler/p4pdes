#ifndef CASES_H_
#define CASES_H_

// -----------------------------------------------------------------------------
// CASE 0
// LINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN
// USE: trap.poly

double a_lin(double u, double x, double y) {
    return 1.0;
}

// manufactured from a_lin(), uexact_lin():
double f_lin(double u, double x, double y) {
    return 2.0 + 3.0 * y * y;
}

double uexact_lin(double x, double y) {
    const double y2 = y * y;
    return 1.0 - y2 - 0.25 * y2 * y2;
}

// just evaluate exact u on boundary point:
double gD_lin(double x, double y) {
    return uexact_lin(x,y);
}

double gN_lin(double x, double y) {
    return 0.0;
}


// -----------------------------------------------------------------------------
// CASE 1
// NONLINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN
// USE: trap.poly

double a_nonlin(double u, double x, double y) {
    return 1.0 + u * u;
}

// manufactured from a_nonlin(), uexact_lin()
double f_nonlin(double u, double x, double y) {
    const double dudy = - 2.0 * y - y * y * y;
    return - 2.0 * u * dudy * dudy + (1.0 + u * u) * (2.0 + 3.0 * y * y);
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

// only valid on line with slope -1
double gN_linneu(double x, double y) {
    return - y * (2.0 + y * y) / sqrt(2.0);
}


// -----------------------------------------------------------------------------
// CASE 3
// LINEAR PROBLEM, HOMOGENEOUS DIRICHLET ONLY, SQUARE DOMAIN, CHAPTER 3
// USE: genstructured.py to generate meshes

double a_square(double u, double x, double y) {
    return 1.0;
}

// manufactured from a_square(), uexact_square():
double f_square(double u, double x, double y) {
    const double x2 = x * x,
                 y2 = y * y;
    return 2.0 * (1.0 - 6.0 * x2) * y2 * (1.0 - y2)
           + 2.0 * (1.0 - 6.0 * y2) * x2 * (1.0 - x2);
}

double uexact_square(double x, double y) {
    const double x2 = x * x,
                 y2 = y * y;
    return (x2 - x2 * x2) * (y2 * y2 - y2);
}

// just evaluate exact u on boundary point:
double gD_square(double x, double y) {
    return uexact_square(x,y);
}

// gN_fcn() = NULL in square case; want seg fault if called


// -----------------------------------------------------------------------------
// CASE 4
// LINEAR PROBLEM, HOMOGENEOUS DIRICHLET ONLY; NO EXACT SOLN
// USE: koch.poly from kochmesh.py (or genstructured.py)

double a_koch(double u, double x, double y) {
    return 1.0;
}

double f_koch(double u, double x, double y) {
    return 1.0;
}

double gD_koch(double x, double y) {
    return 0.0;
}

#endif

