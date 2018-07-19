include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

.PHONY: test

TESTDIRS = ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch9 ch10 ch11 ch12   # FIXME: ch8 ch13
test:
	@echo "PETSC_DIR=${PETSC_DIR}"
	@echo "PETSC_ARCH=${PETSC_ARCH}"
	for DIR in $(TESTDIRS); do \
	     (cd $$DIR && ${MAKE} -s test); \
	done

# etc

.PHONY: distclean

distclean:
	for DIR in $(TESTDIRS); do \
	     (cd $$DIR; rm -f maketmp tmp difftmp; ${MAKE} distclean; cd solns/; ${MAKE} distclean); \
	done
	@rm -f *~ *.o *.pyc

