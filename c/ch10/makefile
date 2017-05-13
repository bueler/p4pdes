include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

well: well.o  chkopts
	-${CLINKER} -o well well.o  ${PETSC_LIB}
	${RM} well.o

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_LIB}
	${RM} stokes.o

# copy of this broken example from petsc commit 55cee6f09eb989b4d30b8ed1c133ceefb891d332
ex7: ex7.o  chkopts
	-${CLINKER} -o ex7 ex7.o  ${PETSC_LIB}
	${RM} ex7.o

# etc

.PHONY: distclean

distclean:
	@rm -f *~ stokes ex7 well

