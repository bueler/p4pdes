best practices for C programs in `p4pdes/c/`
--------------------------------------------

The following best practices are goals for Bueler, the author.  Who does not
always succeed.

  1. `./program -help intro` should produce a description of what the program
     does _and_ its option prefix `xxx_`.
  2. `./program -help |grep xxx_` should produce program-specific options.
  3. A feature without a regression test is broken.
  4. A program which is not valgrind-clean is broken.
  5. Any important code idea should have at least one associated exercise.
     * If the exercise asks for a program, the solution is in `solns/`.
  6. Each program should declare high-level structure first, then `main()`,
     and then function implementations (definitions).
     * All non-static functions should be declared with a prototype (e.g.
       `extern int foo(double);`) just before `main()`.
     * `static` helper functions should be defined before `main()`.
  7. When in doubt the style should follow the Style Guide in the PETSc
     [Developer's Documentation](https://petsc.org/release/developers/).
     * A collection of examples is not a library API, so rigid conformance
       would be foolish.
