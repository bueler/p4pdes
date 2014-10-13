#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

// assemble structured Laplacian matrix into A, using local stencil
PetscErrorCode formlaplacian(DM da, Mat A);

#endif

