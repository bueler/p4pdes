include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

e: e.o  chkopts
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

# exercise solutions

e1balanced: e1balanced.o  chkopts
	-${CLINKER} -o e1balanced e1balanced.o  ${PETSC_LIB}
	${RM} e1balanced.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ e e1balanced

