#ifndef SOLUTIONCASES_H_
#define SOLUTIONCASES_H_

// LINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN

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

// NONLINEAR PROBLEM, NONHOMOGENEOUS DIRICHLET, HOMOGENEOUS NEUMANN

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

// LINEAR PROBLEM, HOMOGENEOUS DIRICHLET, SQUARE DOMAIN, CHAPTER 3

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

#endif

