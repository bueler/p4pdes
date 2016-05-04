include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


unfem: unfem.o um.o chkopts
	-${CLINKER} -o unfem unfem.o um.o  ${PETSC_LIB}
	${RM} unfem.o um.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ *.pyc unfem *.dat *.dat.*

