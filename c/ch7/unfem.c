
static char help[] = "Unstructured 2D FEM solution of nonlinear Poisson equation\n"
"    - div( a(u,x,y) grad u ) = f(u,x,y)\n"
"on arbitrary 2D polygonal domain, with boundary data g_D(x,y), g_N(x,y).\n"
"Functions a(), f(), g_D(), g_N(), and u_exact() are given as formulas.\n"
"(There are three different solution cases implemented for these functions.)\n"
"Input files in PETSc binary format contain node coordinates, elements, and\n"
"boundary flags.  Allows non-homogeneous Dirichlet and Neumann conditions\n"
"along subsets of boundary.\n\n";

#include <petsc.h>
#include "um.h"
#include "cases.h"

//STARTCTX
typedef struct {
    UM     *mesh;
    int    solncase,
           quaddeg;
    double (*a_fcn)(double, double, double);
    double (*f_fcn)(double, double, double);
    double (*gD_fcn)(double, double);
    double (*gN_fcn)(double, double);
    double (*uexact_fcn)(double, double);
    PetscLogStage readstage, setupstage, solverstage, resstage, jacstage;  //STRIP
} unfemCtx;
//ENDCTX

PetscErrorCode FillExact(Vec uexact, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const Node   *aloc;
    double       *auexact;
    int          i;
    ierr = UMGetNodeCoordArrayRead(ctx->mesh,&aloc); CHKERRQ(ierr);
    ierr = VecGetArray(uexact,&auexact); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh->N; i++) {
        auexact[i] = ctx->uexact_fcn(aloc[i].x,aloc[i].y);
    }
    ierr = VecRestoreArray(uexact,&auexact); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(ctx->mesh,&aloc); CHKERRQ(ierr);
    return 0;
}

//STARTFEM
double chi(int L, double xi, double eta) {
    if (L == 0)
        return 1.0 - xi - eta;
    else
        return (L == 1) ? xi : eta;
}

const double dchi[3][2] = {{-1.0,-1.0},{ 1.0, 0.0},{ 0.0, 1.0}};

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[3], double xi, double eta) {
    double sum = 0.0;
    int    L;
    for (L = 0; L < 3; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

double InnerProd(const double V[2], const double W[2]) {
    return V[0] * W[0] + V[1] * W[1];
}

// quadrature points and weights
const int    Q[3] = {1, 3, 4};  // number of quadrature points
const double w[3][4]   = {{1.0/2.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    1.0/6.0,   1.0/6.0,   NAN},
                          {-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0}},
             xi[3][4]  = {{1.0/3.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    2.0/3.0,   1.0/6.0,   NAN},
                          {1.0/3.0,    1.0/5.0,   3.0/5.0,   1.0/5.0}},
             eta[3][4] = {{1.0/3.0,    NAN,       NAN,       NAN},
                          {1.0/6.0,    1.0/6.0,   2.0/3.0,   NAN},
                          {1.0/3.0,    1.0/5.0,   1.0/5.0,   3.0/5.0}};
//ENDFEM

//STARTBDRYRESIDUALS
PetscErrorCode FormFunction(SNES snes, Vec u, Vec F, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx     *user = (unfemCtx*)ctx;
    const int    *abfn, *ae, *as, *abfs, *en, deg = user->quaddeg - 1;
    const Node   *aloc;
    const double *au;
    double       *aF, unode[3], gradu[2], gradpsi[3][2], uquad[4], aquad[4],
                 fquad[4], dx, dy, dx1, dx2, dy1, dy2, detJ,
                 ls, xmid, ymid, sint, xx, yy, sum;
    int          n, p, na, nb, k, l, q;

    PetscLogStagePush(user->resstage);  //STRIP
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecSet(F,0.0); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    ierr = UMGetNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);

    // Dirichlet node residuals
    ierr = ISGetIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    for (n = 0; n < user->mesh->N; n++) {
        if (abfn[n] == 2)  // node is Dirichlet
            aF[n] = au[n] - user->gD_fcn(aloc[n].x,aloc[n].y);
    }
    // Neumann segment contributions
    ierr = ISGetIndices(user->mesh->s,&as); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->bfs,&abfs); CHKERRQ(ierr);
    for (p = 0; p < user->mesh->P; p++) {
        if (abfs[p] == 1) {  // segment is Neumann
            na = as[2*p+0];  // nodes at end of segment
            nb = as[2*p+1];
            // length of segment
            dx = aloc[na].x-aloc[nb].x;
            dy = aloc[na].y-aloc[nb].y;
            ls = sqrt(dx * dx + dy * dy);
            // midpoint rule; psi_na=psi_nb=0.5 at midpoint of segment
            xmid = 0.5*(aloc[na].x+aloc[nb].x);
            ymid = 0.5*(aloc[na].y+aloc[nb].y);
            sint = 0.5 * user->gN_fcn(xmid,ymid) * ls;
            // nodes at end of segment could be Dirichlet
            if (abfn[na] != 2)
                aF[na] -= sint;
            if (abfn[nb] != 2)
                aF[nb] -= sint;
        }
    }
    ierr = ISRestoreIndices(user->mesh->s,&as); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bfs,&abfs); CHKERRQ(ierr);
//ENDBDRYRESIDUALS

//STARTELEMENTRESIDUALS
    // element contributions
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < user->mesh->K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        // geometry of element
        dx1 = aloc[en[1]].x - aloc[en[0]].x;
        dx2 = aloc[en[2]].x - aloc[en[0]].x;
        dy1 = aloc[en[1]].y - aloc[en[0]].y;
        dy2 = aloc[en[2]].y - aloc[en[0]].y;
        detJ = dx1 * dy2 - dx2 * dy1;
        // gradients of hat functions
        for (l = 0; l < 3; l++) {
            gradpsi[l][0] = ( dy2 * dchi[l][0] - dy1 * dchi[l][1]) / detJ;
            gradpsi[l][1] = (-dx2 * dchi[l][0] + dx1 * dchi[l][1]) / detJ;
        }
        // u and grad u on element
        gradu[0] = 0.0;
        gradu[1] = 0.0;
        for (l = 0; l < 3; l++) {
            if (abfn[en[l]] == 2)
                unode[l] = user->gD_fcn(aloc[en[l]].x,aloc[en[l]].y);
            else
                unode[l] = au[en[l]];
            gradu[0] += unode[l] * gradpsi[l][0];
            gradu[1] += unode[l] * gradpsi[l][1];
        }
        // function values at quadrature points on element
        for (q = 0; q < Q[deg]; q++) {
            uquad[q] = eval(unode,xi[deg][q],eta[deg][q]);
            xx = aloc[en[0]].x + dx1 * xi[deg][q] + dx2 * eta[deg][q];
            yy = aloc[en[0]].y + dy1 * xi[deg][q] + dy2 * eta[deg][q];
            aquad[q] = user->a_fcn(uquad[q],xx,yy);
            fquad[q] = user->f_fcn(uquad[q],xx,yy);
        }
        // residual contribution for each node of element
        for (l = 0; l < 3; l++) {
            if (abfn[en[l]] < 2) { // if NOT a Dirichlet node
                sum = 0.0;
                for (q = 0; q < Q[deg]; q++)
                    sum += w[deg][q]
                             * ( aquad[q] * InnerProd(gradu,gradpsi[l])
                                 - fquad[q] * chi(l,xi[deg][q],eta[deg][q]) );
                aF[en[l]] += fabs(detJ) * sum;
            }
        }
    }
    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP
    return 0;
}
//ENDELEMENTRESIDUALS

PetscErrorCode FormPicard(SNES snes, Vec u, Mat A, Mat P, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx     *user = (unfemCtx*)ctx;
    const int    *abfn, *ae, *en, deg = user->quaddeg - 1;
    const Node   *aloc;
    const double *au;
    double       unode[3], gradpsi[3][2],
                 uquad[4], aquad[4], v[9],
                 dx1, dx2, dy1, dy2, detJ, xx, yy, sum;
    int          n, k, l, m, q, cr, cv, row[3];

    PetscLogStagePush(user->jacstage);  //STRIP
    ierr = MatZeroEntries(P); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    for (n = 0; n < user->mesh->N; n++) {
        if (abfn[n] == 2) {
            v[0] = 1.0;
            ierr = MatSetValues(P,1,&n,1,&n,v,ADD_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = UMGetNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);
//STARTPICARD
    for (k = 0; k < user->mesh->K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        // geometry of element
        dx1 = aloc[en[1]].x - aloc[en[0]].x;
        dx2 = aloc[en[2]].x - aloc[en[0]].x;
        dy1 = aloc[en[1]].y - aloc[en[0]].y;
        dy2 = aloc[en[2]].y - aloc[en[0]].y;
        detJ = dx1 * dy2 - dx2 * dy1;
        // gradients of hat functions and u on element
        for (l = 0; l < 3; l++) {
            gradpsi[l][0] = ( dy2 * dchi[l][0] - dy1 * dchi[l][1]) / detJ;
            gradpsi[l][1] = (-dx2 * dchi[l][0] + dx1 * dchi[l][1]) / detJ;
            if (abfn[en[l]] == 2)
                unode[l] = user->gD_fcn(aloc[en[l]].x,aloc[en[l]].y);
            else
                unode[l] = au[en[l]];
        }
        // function values at quadrature points on element
        for (q = 0; q < Q[deg]; q++) {
            uquad[q] = eval(unode,xi[deg][q],eta[deg][q]);
            xx = aloc[en[0]].x + dx1 * xi[deg][q] + dx2 * eta[deg][q];
            yy = aloc[en[0]].y + dy1 * xi[deg][q] + dy2 * eta[deg][q];
            aquad[q] = user->a_fcn(uquad[q],xx,yy);
        }
        // generate 3x3 element stiffness matrix
        cr = 0; // count rows
        cv = 0; // count values
        for (l = 0; l < 3; l++) {
            if (abfn[en[l]] != 2) {
                row[cr] = en[l];
                cr++;
                for (m = 0; m < 3; m++) {
                    if (abfn[en[m]] != 2) {
                        sum = 0.0;
                        for (q = 0; q < Q[deg]; q++) {
                            sum += w[deg][q] * aquad[q]
                                       * InnerProd(gradpsi[l],gradpsi[m]);
                        }
                        v[cv] = fabs(detJ) * sum;
                        cv++;
                    }
                }
            }
        }
        // insert element stiffness matrix
        ierr = MatSetValues(P,cr,row,cr,row,v,ADD_VALUES); CHKERRQ(ierr);
    }
//ENDPICARD
    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (A != P) {
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    ierr = MatSetOption(P,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP
    return 0;
}

/* In this procedure, note that nnz[n] is the number of nonzeros in row n.
It is one for Dirichlet rows.  It is one more than the vertex degree for
interior points, and two more for Neumann nodes. */
//STARTPREALLOC
PetscErrorCode JacobianPreallocation(Mat J, unfemCtx *user) {
    PetscErrorCode ierr;
    const int    *abfn, *ae, *en;
    int          *nnz, n, k, l;

    nnz = (int *)malloc(sizeof(int)*(user->mesh->N));
    ierr = ISGetIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    for (n = 0; n < user->mesh->N; n++)
        nnz[n] = (abfn[n] == 1) ? 2 : 1;
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < user->mesh->K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        for (l = 0; l < 3; l++)
            if (abfn[en[l]] != 2)
                nnz[en[l]] += 1;
    }
    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bfn,&abfn); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(J,-1,nnz); CHKERRQ(ierr);
    free(nnz);
    return 0;
}
//ENDPREALLOC

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscBool   view = PETSC_FALSE, noprealloc = PETSC_FALSE;
    char        root[256] = "", nodesname[256], issname[256];
    UM          mesh;
    unfemCtx    user;
    SNES        snes;
    KSP         ksp;
    PC          pc;
    Mat         A;
    Vec         r, u, uexact;
    double      err, h_max;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscLogStageRegister("Read mesh      ", &user.readstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Set-up         ", &user.setupstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Solver         ", &user.solverstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Residual eval  ", &user.resstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Jacobian eval  ", &user.jacstage); CHKERRQ(ierr);  //STRIP

    user.quaddeg = 1;
    user.solncase = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-case",
           "exact solution cases: 0=linear, 1=nonlinear, 2=nonhomoNeumann, 3=chapter3",
           "unfem.c",user.solncase,&(user.solncase),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh stored in PETSc binary with .vec,.is extensions",
           "unfem.c",root,root,sizeof(root),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddeg",
           "quadrature degree (1,2,3)",
           "unfem.c",user.quaddeg,&(user.quaddeg),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view",
           "view loaded nodes and elements at stdout",
           "unfem.c",view,&view,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-noprealloc",
           "do not perform preallocation before matrix assembly",
           "unfem.c",noprealloc,&noprealloc,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // set parameters and exact solution
    user.a_fcn = &a_lin;
    user.f_fcn = &f_lin;
    user.uexact_fcn = &uexact_lin;
    user.gD_fcn = &gD_lin;
    user.gN_fcn = &gN_lin;
    switch (user.solncase) {
        case 0 :
            break;
        case 1 :
            user.a_fcn = &a_nonlin;
            user.f_fcn = &f_nonlin;
            break;
        case 2 :
            user.gN_fcn = &gN_linneu;
            break;
        case 3 :
            user.a_fcn = &a_square;
            user.f_fcn = &f_square;
            user.uexact_fcn = &uexact_square;
            user.gD_fcn = &gD_square;
            user.gN_fcn = NULL;  // seg fault if ever called
            break;
        default :
            SETERRQ(PETSC_COMM_WORLD,1,"other solution cases not implemented");
    }

    // determine filenames
    strcpy(nodesname, root);
    strncat(nodesname, ".vec", 4);
    strcpy(issname, root);
    strncat(issname, ".is", 3);

//STARTMAINREADMESH
    PetscLogStagePush(user.readstage);  //STRIP
    // read mesh object of type UM
    ierr = UMInitialize(&mesh); CHKERRQ(ierr);
    ierr = UMReadNodes(&mesh,nodesname); CHKERRQ(ierr);
    ierr = UMReadISs(&mesh,issname); CHKERRQ(ierr);
    ierr = UMStats(&mesh, &h_max, NULL, NULL, NULL); CHKERRQ(ierr);
    if (view) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UMView(&mesh,stdoutviewer); CHKERRQ(ierr);
    }
    user.mesh = &mesh;
    PetscLogStagePop();  //STRIP
//ENDMAINREADMESH

//STARTMAINMAT
    PetscLogStagePush(user.setupstage);  //STRIP
    // setup matrix for Picard iteration, including preallocation
    ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,mesh.N,mesh.N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
    if (noprealloc) {
        ierr = MatSetUp(A); CHKERRQ(ierr);
    } else {
        ierr = JacobianPreallocation(A,&user); CHKERRQ(ierr);
    }
//ENDMAINMAT

//STARTMAINSOLVER
    // configure SNES, including resetting default KSP and PC
    ierr = VecCreate(PETSC_COMM_WORLD,&r); CHKERRQ(ierr);
    ierr = VecSetSizes(r,PETSC_DECIDE,mesh.N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(r); CHKERRQ(ierr);
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,&user); CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,A,A,FormPicard,&user); CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCICC); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    // set initial iterate and solve
    ierr = VecDuplicate(r,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP
    PetscLogStagePush(user.solverstage);  //STRIP
    ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP
//ENDMAINSOLVER

    // measure error relative to exact solution
    ierr = VecDuplicate(r,&uexact); CHKERRQ(ierr);
    ierr = FillExact(uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "case %d result for N=%d nodes with h_max = %.3e :  |u-u_ex|_inf = %g\n",
               user.solncase,mesh.N,h_max,err); CHKERRQ(ierr);

    // clean-up
    SNESDestroy(&snes);
    MatDestroy(&A);
    VecDestroy(&u);  VecDestroy(&r);  VecDestroy(&uexact);
    UMDestroy(&mesh);
    PetscFinalize();
    return 0;
}

