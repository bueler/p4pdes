include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

layer1: layer1.o  chkopts
	-${CLINKER} -o layer1 layer1.o  ${PETSC_LIB}
	${RM} layer1.o

glaze: glaze.o  chkopts
	-${CLINKER} -o glaze glaze.o  ${PETSC_LIB}
	${RM} glaze.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ glaze layer1

