include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

e: e.o
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

# testing
rune_1:
	-@../testit.sh e "" 1 1

test_e: rune_1

test: test_e

.PHONY: distclean rune_1 test_e test

distclean:
	@rm -f *~ e *tmp

