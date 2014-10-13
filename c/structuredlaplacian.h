#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

// assemble structured Laplacian matrix into A, using local stencil
// actually computes  A u = - hx * hy * (u_xx + u_yy)
PetscErrorCode formlaplacian(DM da, Mat A);

#endif

