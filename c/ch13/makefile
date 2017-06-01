include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

tiny: tiny.o plexview.o chkopts
	-${CLINKER} -o tiny tiny.o plexview.o ${PETSC_LIB}
	${RM} tiny.o plexview.o

# testing

#FIXME add one or two tests for tiny

# etc

.PHONY: distclean

distclean:
	@rm -f *~ tiny

