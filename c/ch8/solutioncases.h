#ifndef SOLUTIONCASES_H_
#define SOLUTIONCASES_H_

double a_lin(double u, double x, double y) {
    return 1.0;
}

double f_lin(double u, double x, double y) {
    return 2.0 + 3.0 * y * y;
}

double uexact_lin(double x, double y) {
    const double y2 = y * y;
    return 1.0 - y2 - 0.25 * y2 * y2;
}

double gD_lin(double x, double y) {
    return uexact_lin(x,y);
}

double a_nonlin(double u, double x, double y) {
    return 1.0 + u * u;
}

double f_nonlin(double u, double x, double y) {
    const double dudy = - 2.0 * y - y * y * y;
    return - 2.0 * u * dudy * dudy + (1.0 + u * u) * (2.0 + 3.0 * y * y);
}

// uexact_nonlin = uexact_lin

// gD_nonlin = gD_lin

#endif

