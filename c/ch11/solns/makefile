include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

layer1: layer1.o  chkopts
	-${CLINKER} -o layer1 layer1.o  ${PETSC_LIB}
	${RM} layer1.o

glaze: glaze.o  chkopts
	-${CLINKER} -o glaze glaze.o  ${PETSC_LIB}
	${RM} glaze.o

# text
runlayer1_1:
	-@../../testit.sh layer1 "-lay_eps 0.1 -da_refine 1 -ksp_view_mat ::ascii_dense -snes_monitor_solution" 1 1

test_layer1: runlayer1_1

test: test_layer1

# etc

.PHONY: distclean runlayer1_1 test_layer1 test

distclean:
	@rm -f *~ glaze layer1

