include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bratu2D: bratu2D.o ../poissonfunctions.o chkopts
	-${CLINKER} -o bratu2D bratu2D.o ../poissonfunctions.o ${PETSC_LIB}
	${RM} bratu2D.o ../poissonfunctions.o

# etc

.PHONY: distclean

distclean:
	@rm -f bratu2D

