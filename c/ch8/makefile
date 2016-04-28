include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


unfem: unfem.o  chkopts
	-${CLINKER} -o unfem unfem.o  ${PETSC_LIB}
	${RM} unfem.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ *.pyc unfem *.dat *.dat.*

