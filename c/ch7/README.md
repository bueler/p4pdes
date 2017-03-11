p4pdes/c/ch7/ on branch plap-not-weird
======================================

I think this is a failed experiment.  I was misguided in trying to change the semantics of the version of `plap.c` in the `master` branch.

In the `master` branch we use only the interior points as unknowns.  This means that every vector of unknowns represents an element of W_g^{1,p}(\Omega) where \Omega is the (open) unit square and g is the boundary value.  Note that the Poincare inequality applies, and so the functional I[u] is coercive even without a zeroth-order term.  Also, g can be chosen so that a manufactured solution u can be found so that |grad u| is never zero in the interior.  (This relates to the value of parameter alpha.  If alpha is positive, or less than -1, then we have this property.)  These facts mean that the objective function is coercive and the minimizer usually satisfies a uniformly-elliptic PDE for any p.

In this `plap-not-weird` branch we have periodic boundary conditions, i.e. the solution is in W^{1,p}(T^2) not W_g^{1,p}(\Omega).  Again every vector of unknowns represents an element of W^{1,p}.  However, because there is no Poincare inequality, we have added a zeroth-order term to make the functional coercive.  Furthermore, any nontrivial solution u in W^{1,p} now has both a minimum and an maximum in the domain, so that |grad u|=0 always applies somewhere in the domain.  Thus the PDE satisfied by the minimizer is _not_ uniformly elliptic if p > 2.  (The cases p < 2 are even harder to understand.)  Every possible example is hard to solve numerically, whether using objective-only or with a residual.

Earlier I thought that the grid usage in the `master` version of `plap.c` made multigrid hopeless.  Not so; it works fine in the `master` version, because recent commits replace storage of fields on the fine grid with correct pointwise evaluation functions.  (See 1ee9f7d08564a198282b4beeeedc5f01551b50d7.)

