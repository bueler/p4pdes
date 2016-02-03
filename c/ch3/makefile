include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

poisson: poisson.o chkopts
	-${CLINKER} -o poisson poisson.o ${PETSC_LIB}
	${RM} poisson.o

# testing
runpoisson_1:
	-@../testit.sh poisson "-da_grid_x 5 -da_grid_y 7" 1 1

runpoisson_2:
	-@../testit.sh poisson "-da_grid_x 4 -da_grid_y 3 -a_mat_view" 1 2

runpoisson_3:
	-@../testit.sh poisson "-da_grid_x 10 -da_grid_y 10" 4 3

# etc

.PHONY: distclean

distclean:
	@rm -f *~ poisson

