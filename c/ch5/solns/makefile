include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

stiff: stiff.o
	-${CLINKER} -o stiff stiff.o  ${PETSC_LIB}
	${RM} stiff.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ stiff *.pyc *.dat *.dat.info

