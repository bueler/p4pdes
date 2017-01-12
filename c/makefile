include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

.PHONY: test

TESTDIRS = ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch10 ch11
test:
	for DIR in $(TESTDIRS); do \
	     (cd $$DIR && ${MAKE} -s test); \
	done

# etc

.PHONY: distclean

distclean:
	for DIR in $(TESTDIRS); do \
	     (cd $$DIR; ${MAKE} distclean; rm -f output/*~ solns/*~; rm -f maketmp tmp difftmp); \
	done
	@rm -f *~

