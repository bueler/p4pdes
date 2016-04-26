include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


FIXMEconvert: convert.o readmesh.o  chkopts
	-${CLINKER} -o convert convert.o readmesh.o  ${PETSC_LIB}
	${RM} convert.o readmesh.o


# etc

.PHONY: distclean

distclean:
	@rm -f *~ FIXME

