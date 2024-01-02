include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

e: e.o
	-${CLINKER} -o e e.o  ${PETSC_LIB}
	${RM} e.o

expx: expx.o
	-${CLINKER} -o expx expx.o  ${PETSC_LIB}
	${RM} expx.o

balanced: balanced.o
	-${CLINKER} -o balanced balanced.o  ${PETSC_LIB}
	${RM} balanced.o

# testing
rune_1:
	-@../testit.sh e "" 1 1

test_e: rune_1

test: test_e

.PHONY: clean distclean rune_1 test_e test

distclean: clean

clean::
	@rm -f *~ e *tmp expx balanced
