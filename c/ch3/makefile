include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

poisson: poisson.o
	-${CLINKER} -o poisson poisson.o ${PETSC_LIB}
	${RM} poisson.o

poisson1D: poisson1D.o
	-${CLINKER} -o poisson1D poisson1D.o ${PETSC_LIB}
	${RM} poisson1D.o

poisson_modify: poisson_modify.o
	-${CLINKER} -o poisson_modify poisson_modify.o ${PETSC_LIB}
	${RM} poisson_modify.o

poisson_dirichlet: poisson_dirichlet.o
	-${CLINKER} -o poisson_dirichlet poisson_dirichlet.o ${PETSC_LIB}
	${RM} poisson_dirichlet.o

# testing
runpoisson_1:
	-@../testit.sh poisson "-da_grid_x 5 -da_grid_y 7" 1 1

runpoisson_2:
	-@../testit.sh poisson "-da_grid_x 4 -da_grid_y 3 -mat_view" 1 2

runpoisson_3:
	-@../testit.sh poisson "-da_grid_x 10 -da_grid_y 10" 4 3

test_poisson: runpoisson_1 runpoisson_2 runpoisson_3

test: test_poisson

# etc

.PHONY: clean distclean runpoisson_1 runpoisson_2 runpoisson_3 test test_poisson

distclean: clean

clean::
	@rm -f *~ poisson poisson1D *tmp poisson_modify poisson_dirichlet

