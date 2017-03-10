p4pdes/c/ch7/ on branch plap-not-weird
======================================

I think this is a failed experiment.

In the `master` branch, `plap.c` uses only the interior points as unknowns.  This means that every vector of unknowns represents an element of W_g^{1,p}(\Omega) where \Omega is the (open) unit square and g is the boundary value.  Note that the Poincare inequality applies, and so the functional I[u] is coercive even without a zeroth-order term.  Also, g can be chosen so that a manufactured solution u can be found so that |grad u| is never zero in the interior.  (This relates to the value of parameter alpha.)  These facts mean that the objective function is coercive and the minimizer satisfies (for many alpha) a uniformly elliptic PDE.  However, because grid refinement requires expanding the domain where the unknowns define the function, multigrid does not apply.  (Presumably one could write new prolongation/restriction operators to fix this.)

In this `plap-not-weird` branch, `plap.c` has periodic boundary conditions, i.e. the solution is in W^{1,p}(T^2) not W_g^{1,p}(\Omega).  Again every vector of unknowns represents an element of W^{1,p}.  Because there is no Poincare we have added a zeroth-order term to make the functional coercive.  However, any nontrivial solution u in W^{1,p} now has both a minimum and an maximum in the domain, so that |grad u|=0 always somewhere in the domain.  Thus the PDE satisfied by the minimizer is _not_ uniformly elliptic.  Every possible example is hard to solve for numerically, whether using objective-only or residual verion.  On the other hand, PETSc's standard multigrid  works.

