include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

glaze: glaze.o  chkopts
	-${CLINKER} -o glaze glaze.o  ${PETSC_LIB}
	${RM} glaze.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ glaze

