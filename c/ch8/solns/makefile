include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

lntwo: lntwo.o
	-${CLINKER} -o lntwo lntwo.o  ${PETSC_LIB}
	${RM} lntwo.o

.PHONY: clean distclean

distclean: clean

clean::
	@rm -f *~ lntwo
