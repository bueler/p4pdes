#ifndef POISSONTOOLS_H_
#define POISSONTOOLS_H_

// tools for working with linear system from FEM on unstructured triangulation
// to solve Poisson equation

// run through elements and find and assemble all Dirichlet rows
// uses INSERT_VALUES; may redundantly INSERT_VALUES many times given row
PetscErrorCode dirichletrows(MPI_Comm comm,
                             Vec E,
                             PetscScalar (*g)(PetscScalar, PetscScalar),
                             Mat A, Vec b);

// run through elements and add (INSERT_ADD) element stiffness contributions
// to A, and f, Neumann gamma, and Dirichlet g contributions to b
PetscErrorCode assembleadd(MPI_Comm comm,
                           Vec E,
                           PetscScalar (*f)(PetscScalar, PetscScalar),
                           PetscScalar (*g)(PetscScalar, PetscScalar),
                           PetscScalar (*gamma)(PetscScalar, PetscScalar),
                           Mat A, Vec b);

// assembly of linear system  A u = b for Poisson problem
//   - div(grad u) = f      in region,
//               f = g      on Dirichlet boundary
//           df/dn = gamma  on Neumann boundary
// note Mat A and Vec b must already be created;  A must either be preallocated
// or MatSetUp()
PetscErrorCode assemble(MPI_Comm comm,
                        Vec E,         // array of elementtype, as read by readmesh()
                        PetscScalar (*f)(PetscScalar, PetscScalar),
                        PetscScalar (*g)(PetscScalar, PetscScalar),
                        PetscScalar (*gamma)(PetscScalar, PetscScalar),
                        Mat A, Vec b);
#endif
