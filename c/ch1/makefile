include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

e: e.o  chkopts
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

shuffle: shuffle.o  chkopts
	-${CLINKER} -o shuffle shuffle.o  ${PETSC_LIB}
	${RM} shuffle.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ e shuffle

