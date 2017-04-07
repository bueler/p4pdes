best practices for codes in "PETSc for PDEs"
============================================

My goal for myself is to use consistent best practices.  This goal is not yet
achieved, of course.

  * A feature without a regression test is broken.
  * A code which is not valgrind-clean is broken.
  * Any important idea in a code should have at least one associated exercise,
    with a solution to that exercise in `solns/`.
  * All functions should be declared with a prototype (e.g.
    `extern int foo(double);`) before `main()`, and then put after it.  Thus
    the reader of a code should see high-level stuff first, namely the `help`
    string and the context variables, and then `main()`, and then further
    details.
  * If you do `./code -help |head` then you should see the main idea _and_ the
    option prefix, so that `./code -help |grep prefix_` is immediately available.
  * When in doubt the style should follow the Style Guide in the
    [PETSc Developer's Guide](http://www.mcs.anl.gov/petsc/developers/developers.pdf),
    but a collection of examples is not a library API, so this is not rigid.

