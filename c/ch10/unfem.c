static char help[] =
"Unstructured 2D FEM solution of nonlinear Poisson problem.  Option prefix un_.\n"
"Equation is\n"
"    - div( a(u,x,y) grad u ) = f(u,x,y)\n"
"on arbitrary 2D polygonal domain, with boundary data g_D(x,y), g_N(x,y).\n"
"Functions a(), f(), g_D(), g_N(), and u_exact() are given as formulas.\n"
"Input files in PETSc binary format contain node coordinates, elements, Neumann\n"
"boundary segments and boundary flags.  Allows non-homogeneous Dirichlet and/or\n"
"Neumann boundary conditions.  Five different solution cases are implemented.\n\n";

#include <petsc.h>
#include "../quadrature.h"
#include "um.h"
#include "cases.h"

//STARTCTX
typedef struct {
    UM        *mesh;
    PetscInt  solncase,
              quaddegree;
    PetscReal (*a_fcn)(PetscReal, PetscReal, PetscReal);
    PetscReal (*f_fcn)(PetscReal, PetscReal, PetscReal);
    PetscReal (*gD_fcn)(PetscReal, PetscReal);
    PetscReal (*gN_fcn)(PetscReal, PetscReal);
    PetscReal (*uexact_fcn)(PetscReal, PetscReal);
    PetscLogStage readstage, setupstage, solverstage, resstage, jacstage;  //STRIP
} unfemCtx;
//ENDCTX

//STARTFEM
PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    const PetscReal z[3] = {1.0 - xi - eta, xi, eta};
    return z[L];
}

const PetscReal dchi[3][2] = {{-1.0,-1.0},{ 1.0, 0.0},{ 0.0, 1.0}};

// evaluate v(xi,eta) on reference element using local node numbering
PetscReal eval(const PetscReal v[3], PetscReal xi, PetscReal eta) {
    PetscReal  sum = 0.0;
    PetscInt   L;
    for (L = 0; L < 3; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}
//ENDFEM

extern PetscErrorCode FillExact(Vec, unfemCtx*);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
extern PetscErrorCode FormPicard(SNES, Vec, Mat, Mat, void*);
extern PetscErrorCode PreallocateAndSetNonzeros(Mat, unfemCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscMPIInt size;
    PetscBool   viewmesh = PETSC_FALSE,
                viewsoln = PETSC_FALSE,
                noprealloc = PETSC_FALSE,
                savepintbinary = PETSC_FALSE,
                savepintmatlab = PETSC_FALSE;
    char        root[256] = "", nodesname[256], issname[256], solnname[256],
                pintname[256] = "";
    PetscInt    savepintlevel = -1, levels;
    UM          mesh;
    unfemCtx    user;
    SNES        snes;
    KSP         ksp;
    PC          pc;
    PCType      pctype;
    Mat         A;
    Vec         r, u, uexact;
    PetscReal   err, h_max;

    PetscInitialize(&argc,&argv,NULL,help);

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
    if (size != 1) {
        SETERRQ(PETSC_COMM_WORLD,1,"unfem only works on one MPI process");
    }

    ierr = PetscLogStageRegister("Read mesh      ", &user.readstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Set-up         ", &user.setupstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Solver         ", &user.solverstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Residual eval  ", &user.resstage); CHKERRQ(ierr);  //STRIP
    ierr = PetscLogStageRegister("Jacobian eval  ", &user.jacstage); CHKERRQ(ierr);  //STRIP

    user.quaddegree = 1;
    user.solncase = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "un_", "options for unfem", ""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-case",
           "exact solution cases: 0=linear, 1=nonlinear, 2=nonhomoNeumann, 3=chapter3, 4=koch",
           "unfem.c",user.solncase,&(user.solncase),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-gamg_save_pint_binary",
           "filename under which to save interpolation operator (Mat) in PETSc binary format",
           "unfem.c",pintname,pintname,sizeof(pintname),&savepintbinary); CHKERRQ(ierr);
    ierr = PetscOptionsString("-gamg_save_pint_matlab",
           "filename under which to save interpolation operator (Mat) in ascii Matlab format",
           "unfem.c",pintname,pintname,sizeof(pintname),&savepintmatlab); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-gamg_save_pint_level",
           "saved interpolation operator is between L-1 and L where this option sets L; defaults to finest levels",
           "unfem.c",savepintlevel,&savepintlevel,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-mesh",
           "file name root of mesh stored in PETSc binary with .vec,.is extensions",
           "unfem.c",root,root,sizeof(root),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-noprealloc",
           "do not perform preallocation before matrix assembly",
           "unfem.c",noprealloc,&noprealloc,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddegree",
           "quadrature degree (1,2,3)",
           "unfem.c",user.quaddegree,&(user.quaddegree),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_mesh",
           "view loaded mesh (nodes and elements) at stdout",
           "unfem.c",viewmesh,&viewmesh,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_solution",
           "view solution u(x,y) to binary file; uses root name of mesh plus .soln\nsee petsc2tricontour.py to view graphically",
           "unfem.c",viewsoln,&viewsoln,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // determine filenames
    if (strlen(root) == 0) {
        SETERRQ(PETSC_COMM_WORLD,2,"no mesh name root given; rerun with '-un_mesh foo'");
    }
    strcpy(nodesname, root);
    strncat(nodesname, ".vec", 4);
    strcpy(issname, root);
    strncat(issname, ".is", 3);

    // set source/boundary functions and exact solution
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
        case 4 :
            user.a_fcn = &a_koch;
            user.f_fcn = &f_koch;
            user.uexact_fcn = NULL;  // seg fault if ever called
            user.gD_fcn = &gD_koch;
            user.gN_fcn = NULL;  // seg fault if ever called
            break;
        default :
            SETERRQ(PETSC_COMM_WORLD,3,"other solution cases not implemented");
    }

    PetscLogStagePush(user.readstage);
//STARTREADMESH
    // read mesh object of type UM
    ierr = UMInitialize(&mesh); CHKERRQ(ierr);
    ierr = UMReadNodes(&mesh,nodesname); CHKERRQ(ierr);
    ierr = UMReadISs(&mesh,issname); CHKERRQ(ierr);
    ierr = UMStats(&mesh, &h_max, NULL, NULL, NULL); CHKERRQ(ierr);
    user.mesh = &mesh;
//ENDREADMESH
    PetscLogStagePop();

    if (viewmesh) {
        PetscViewer stdoutviewer;
        ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&stdoutviewer); CHKERRQ(ierr);
        ierr = UMViewASCII(&mesh,stdoutviewer); CHKERRQ(ierr);
    }

    PetscLogStagePush(user.setupstage);
//STARTMAININITIAL
    // configure Vecs
    ierr = VecCreate(PETSC_COMM_WORLD,&r); CHKERRQ(ierr);
    ierr = VecSetSizes(r,PETSC_DECIDE,mesh.N); CHKERRQ(ierr);
    ierr = VecSetFromOptions(r); CHKERRQ(ierr);
    ierr = VecDuplicate(r,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);

    // configure SNES: reset default KSP and PC
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,&user); CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
    ierr = PCSetType(pc,PCICC); CHKERRQ(ierr);

    // setup matrix for Picard iteration, including preallocation
    ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,mesh.N,mesh.N); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A); CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
    // Preallocation and setting the nonzero (sparsity) pattern is
    //   recommended; setting the pattern allows finite difference
    //   approximation of the Jacobian using coloring.  Option
    //   -un_noprealloc reveals the poor performance otherwise.
    if (noprealloc) {
        ierr = MatSetUp(A); CHKERRQ(ierr);
    } else {
        ierr = PreallocateAndSetNonzeros(A,&user); CHKERRQ(ierr);
    }
    // The following call-back setting is ignored under option -snes_fd
    //   or -snes_fd_color; in the latter case
    //   SNESComputeJacobianDefaultColor() substitutes for FormPicard().
    ierr = SNESSetJacobian(snes,A,A,FormPicard,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP

    // solve
    PetscLogStagePush(user.solverstage);  //STRIP
    ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
//ENDMAININITIAL
    PetscLogStagePop();

    // report if PC is GAMG
    ierr = PCGetType(pc,&pctype); CHKERRQ(ierr);
    if (strcmp(pctype,"gamg") == 0) {
        ierr = PCMGGetLevels(pc,&levels); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  PC is GAMG with %d levels\n",levels); CHKERRQ(ierr);
    }

    // save Pint from GAMG if requested
    if (strlen(pintname) > 0) {
        Mat         pint;
        PetscViewer viewer;
        if (strcmp(pctype,"gamg") != 0) {
            SETERRQ(PETSC_COMM_WORLD,4,"option -un_gamg_save_pint set but PC is not of type PCGAMG");
        }
        if (savepintlevel >= levels) {
            SETERRQ(PETSC_COMM_WORLD,5,"invalid level given in -un_gamg_save_pint_level");
        }
        if (savepintbinary && savepintmatlab) {
            SETERRQ(PETSC_COMM_WORLD,6,"only one of -un_gamg_save_pint_binary OR -un_gamg_save_pint_matlab is allowed");
        }
        if (savepintlevel <= 0) {
            savepintlevel = levels - 1;
        }
        ierr = PCMGGetInterpolation(pc,savepintlevel,&pint); CHKERRQ(ierr);
        if (savepintbinary) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  saving interpolation operator on levels %d->%d in binary format to %s ...\n",
               savepintlevel-1,savepintlevel,pintname); CHKERRQ(ierr);
            ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,pintname,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
            ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
               "  saving interpolation operator on levels %d->%d in ascii format to %s ...\n",
               savepintlevel-1,savepintlevel,pintname); CHKERRQ(ierr);
            ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,pintname,&viewer); CHKERRQ(ierr);
            ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); CHKERRQ(ierr);
        }
        ierr = MatView(pint,viewer); CHKERRQ(ierr);
    }

    // if exact solution available, report numerical error
    if (user.uexact_fcn) {
        ierr = VecDuplicate(r,&uexact); CHKERRQ(ierr);
        ierr = FillExact(uexact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
        ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                   "case %d result for N=%d nodes with h = %.3e :  |u-u_ex|_inf = %.2e\n",
                   user.solncase,mesh.N,h_max,err); CHKERRQ(ierr);
        VecDestroy(&uexact);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                   "case %d result for N=%d nodes with h = %.3e ... done\n",
                   user.solncase,mesh.N,h_max); CHKERRQ(ierr);
    }

    // save solution in PETSc binary if requested
    if (viewsoln) {
        strcpy(solnname, root);
        strncat(solnname, ".soln", 5);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                   "writing solution in binary format to %s ...\n",solnname); CHKERRQ(ierr);
        ierr = UMViewSolutionBinary(&mesh,solnname,u); CHKERRQ(ierr);
    }

    // clean-up
    VecDestroy(&u);  VecDestroy(&r);
    MatDestroy(&A);  SNESDestroy(&snes);  UMDestroy(&mesh);
    return PetscFinalize();
}

PetscErrorCode FillExact(Vec uexact, unfemCtx *ctx) {
    PetscErrorCode ierr;
    const Node   *aloc;
    PetscReal    *auexact;
    PetscInt     i;
    ierr = UMGetNodeCoordArrayRead(ctx->mesh,&aloc); CHKERRQ(ierr);
    ierr = VecGetArray(uexact,&auexact); CHKERRQ(ierr);
    for (i = 0; i < ctx->mesh->N; i++) {
        auexact[i] = ctx->uexact_fcn(aloc[i].x,aloc[i].y);
    }
    ierr = VecRestoreArray(uexact,&auexact); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(ctx->mesh,&aloc); CHKERRQ(ierr);
    return 0;
}

PetscReal InnerProd(const PetscReal V[2], const PetscReal W[2]) {
    return V[0] * W[0] + V[1] * W[1];
}

//STARTRESIDUAL
PetscErrorCode FormFunction(SNES snes, Vec u, Vec F, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx         *user = (unfemCtx*)ctx;
    const Quad2DTri  q = symmgauss[user->quaddegree-1];
    const PetscInt   *ae, *ans, *abf, *en;
    const Node       *aloc;
    const PetscReal  *au;
    PetscInt         p, na, nb, k, l, r;
    PetscReal        *aF, unode[3], gradu[2], gradpsi[3][2], uquad[4],
                     aquad[4], fquad[4], dx, dy, dx1, dx2, dy1, dy2,
                     detJ, ls, xmid, ymid, sint, xx, yy, psi, ip, sum;

    PetscLogStagePush(user->resstage);  //STRIP
    ierr = VecSet(F,0.0); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    ierr = UMGetNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->bf,&abf); CHKERRQ(ierr);

    // Neumann boundary segment contributions (if any)
    if (user->mesh->P > 0) {
        ierr = ISGetIndices(user->mesh->ns,&ans); CHKERRQ(ierr);
        for (p = 0; p < user->mesh->P; p++) {
            na = ans[2*p+0];  nb = ans[2*p+1];  // end nodes of segment
            dx = aloc[na].x-aloc[nb].x;  dy = aloc[na].y-aloc[nb].y;
            ls = sqrt(dx * dx + dy * dy);  // length of segment
            // midpoint rule; psi_na=psi_nb=0.5 at midpoint of segment
            xmid = 0.5*(aloc[na].x+aloc[nb].x);
            ymid = 0.5*(aloc[na].y+aloc[nb].y);
            sint = 0.5 * ls * user->gN_fcn(xmid,ymid);
            // nodes could be Dirichlet
            if (abf[na] != 2)
                aF[na] -= sint;
            if (abf[nb] != 2)
                aF[nb] -= sint;
        }
        ierr = ISRestoreIndices(user->mesh->ns,&ans); CHKERRQ(ierr);
    }

    // element contributions and Dirichlet node residuals
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    for (k = 0; k < user->mesh->K; k++) {
        // element geometry and hat function gradients
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        dx1 = aloc[en[1]].x - aloc[en[0]].x;
        dx2 = aloc[en[2]].x - aloc[en[0]].x;
        dy1 = aloc[en[1]].y - aloc[en[0]].y;
        dy2 = aloc[en[2]].y - aloc[en[0]].y;
        detJ = dx1 * dy2 - dx2 * dy1;
        for (l = 0; l < 3; l++) {
            gradpsi[l][0] = ( dy2 * dchi[l][0] - dy1 * dchi[l][1]) / detJ;
            gradpsi[l][1] = (-dx2 * dchi[l][0] + dx1 * dchi[l][1]) / detJ;
        }
        // u and grad u on element
        gradu[0] = 0.0;
        gradu[1] = 0.0;
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] == 2)  // enforces symmetry
                unode[l] = user->gD_fcn(aloc[en[l]].x,aloc[en[l]].y);
            else
                unode[l] = au[en[l]];
            gradu[0] += unode[l] * gradpsi[l][0];
            gradu[1] += unode[l] * gradpsi[l][1];
        }
        // function values at quadrature points on element
        for (r = 0; r < q.n; r++) {
            uquad[r] = eval(unode,q.xi[r],q.eta[r]);
            xx = aloc[en[0]].x + dx1 * q.xi[r] + dx2 * q.eta[r];
            yy = aloc[en[0]].y + dy1 * q.xi[r] + dy2 * q.eta[r];
            aquad[r] = user->a_fcn(uquad[r],xx,yy);
            fquad[r] = user->f_fcn(uquad[r],xx,yy);
        }
        // residual contribution for each node of element
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] == 2) { // set Dirichlet residual
                xx = aloc[en[l]].x;   yy = aloc[en[l]].y;
                aF[en[l]] = au[en[l]] - user->gD_fcn(xx,yy);
            } else {
                sum = 0.0;
                for (r = 0; r < q.n; r++) {
                    psi = chi(l,q.xi[r],q.eta[r]);
                    ip  = InnerProd(gradu,gradpsi[l]);
                    sum += q.w[r] * ( aquad[r] * ip - fquad[r] * psi );
                }
                aF[en[l]] += PetscAbsReal(detJ) * sum;
            }
        }
    }

    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bf,&abf); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    PetscLogStagePop();  //STRIP
    return 0;
}
//ENDRESIDUAL


//STARTPICARD
PetscErrorCode FormPicard(SNES snes, Vec u, Mat A, Mat P, void *ctx) {
    PetscErrorCode ierr;
    unfemCtx         *user = (unfemCtx*)ctx;
    const Quad2DTri  q = symmgauss[user->quaddegree-1];
    const PetscInt   *ae, *abf, *en;
    const Node       *aloc;
    const PetscReal  *au;
    PetscReal        unode[3], gradpsi[3][2], uquad[4], aquad[4], v[9],
                     dx1, dx2, dy1, dy2, detJ, xx, yy, sum;
    PetscInt         n, k, l, m, r, cr, cv, row[3];

    PetscLogStagePush(user->jacstage);  //STRIP
    ierr = MatZeroEntries(P); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->bf,&abf); CHKERRQ(ierr);
    for (n = 0; n < user->mesh->N; n++) {
        if (abf[n] == 2) {
            v[0] = 1.0;
            ierr = MatSetValues(P,1,&n,1,&n,v,ADD_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = UMGetNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);
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
            if (abf[en[l]] == 2)
                unode[l] = user->gD_fcn(aloc[en[l]].x,aloc[en[l]].y);
            else
                unode[l] = au[en[l]];
        }
        // function values at quadrature points on element
        for (r = 0; r < q.n; r++) {
            uquad[r] = eval(unode,q.xi[r],q.eta[r]);
            xx = aloc[en[0]].x + dx1 * q.xi[r] + dx2 * q.eta[r];
            yy = aloc[en[0]].y + dy1 * q.xi[r] + dy2 * q.eta[r];
            aquad[r] = user->a_fcn(uquad[r],xx,yy);
        }
        // generate 3x3 element stiffness matrix (may be smaller)
        cr = 0;  cv = 0;  // cr = count rows; cv = entry counter
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] != 2) {
                row[cr++] = en[l];
                for (m = 0; m < 3; m++) {
                    if (abf[en[m]] != 2) {
                        sum = 0.0;
                        for (r = 0; r < q.n; r++) {
                            sum += q.w[r] * aquad[r]
                                   * InnerProd(gradpsi[l],gradpsi[m]);
                        }
                        v[cv++] = PetscAbsReal(detJ) * sum;
                    }
                }
            }
        }
        ierr = MatSetValues(P,cr,row,cr,row,v,ADD_VALUES); CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bf,&abf); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = UMRestoreNodeCoordArrayRead(user->mesh,&aloc); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (A != P) {
        ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    PetscLogStagePop();  //STRIP
    return 0;
}
//ENDPICARD


/* The following procedure is accomplishes essentially the same actions
as DMCreateMatrix() when a DM is present.  It first preallocates storage
for the sparse matrix by providing a count of the entries.  Then it
actually sets the sparsity pattern by inserting zeros--ironically--where
there will be nonzero entries.  This means that -snes_fd_color can be used.

Note that nnz[n] is the number of nonzeros in row n.  In our case it
equals one for Dirichlet rows, it is one more than the number of incident
triangles for an interior point, and it is two more than the number of
incident triangles for Neumann boundary nodes. */
//STARTPREALLOC
PetscErrorCode PreallocateAndSetNonzeros(Mat J, unfemCtx *user) {
    PetscErrorCode ierr;
    const PetscInt  *ae, *abf, *en;
    PetscInt        *nnz, n, k, l, cr, row[3];
    PetscReal       zero = 0.0,
                    v[9] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

    // preallocate: set number of nonzeros per row
    ierr = ISGetIndices(user->mesh->bf,&abf); CHKERRQ(ierr);
    ierr = ISGetIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = PetscMalloc1(user->mesh->N,&nnz); CHKERRQ(ierr);
    for (n = 0; n < user->mesh->N; n++)
        nnz[n] = (abf[n] == 1) ? 2 : 1;
    for (k = 0; k < user->mesh->K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        for (l = 0; l < 3; l++)
            if (abf[en[l]] != 2)
                nnz[en[l]] += 1;
    }
    ierr = MatSeqAIJSetPreallocation(J,-1,nnz); CHKERRQ(ierr);
    ierr = PetscFree(nnz); CHKERRQ(ierr);

    // set nonzeros: put values (=zeros) in allocated locations
    for (n = 0; n < user->mesh->N; n++) {
        if (abf[n] == 2) {
            ierr = MatSetValues(J,1,&n,1,&n,&zero,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    for (k = 0; k < user->mesh->K; k++) {
        en = ae + 3*k;  // en[0], en[1], en[2] are nodes of element k
        // a 3x3 element stiffness matrix (at most) for each element
        cr = 0;  // cr = count rows
        for (l = 0; l < 3; l++) {
            if (abf[en[l]] != 2) {
                row[cr++] = en[l];
            }
        }
        ierr = MatSetValues(J,cr,row,cr,row,v,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    // the assembly routine FormPicard() will generate an error if
    //   it tries to put a matrix entry in the wrong place
    ierr = MatSetOption(J,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->e,&ae); CHKERRQ(ierr);
    ierr = ISRestoreIndices(user->mesh->bf,&abf); CHKERRQ(ierr);
    return 0;
}
//ENDPREALLOC

