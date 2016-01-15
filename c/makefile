include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

.PHONY: test

test:
	@(cd ch1/ && make -s rune_1)
	@(cd ch2/ && make -s runvecmatksp_1 runtri_1 runtri_2)
	@(cd ch3/ && make -s runpoisson_1 runpoisson_2 runpoisson_3)
	@(cd ch4/ && make -s runexpcircle_1 runexpcircle_2 runecjacobian_1 runecjacobian_2 runreaction_1 runreaction_2 runreaction_3 runreaction_4)
	@(cd ch5/ && make -s runplap_1 runplap_2)


# extra codes not in chapters yet

heat: heat.o ch3/structuredpoisson.o chkopts
	-${CLINKER} -o heat heat.o ch3/structuredpoisson.o ${PETSC_KSP_LIB}
	${RM} heat.o ch3/structuredpoisson.o

riemann: riemann.o  chkopts
	-${CLINKER} -o riemann riemann.o  ${PETSC_KSP_LIB}
	${RM} riemann.o

stokes: stokes.o  chkopts
	-${CLINKER} -o stokes stokes.o  ${PETSC_SNES_LIB}
	${RM} stokes.o

testprealloc: testprealloc.o ch4/readmesh.o chkopts
	-${CLINKER} -o testprealloc testprealloc.o ch4/readmesh.o ${PETSC_KSP_LIB}
	${RM} testprealloc.o ch4/readmesh.o

testts: testts.o  chkopts
	-${CLINKER} -o testts testts.o  ${PETSC_KSP_LIB}
	${RM} testts.o


# etc

.PHONY: distclean rdistclean

CHAPTERS = ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch9
rdistclean:
	for DIR in $(CHAPTERS); do \
	     (cd $$DIR; rm -f output/*~ solns/*~; ${MAKE} distclean); \
	done

distclean: rdistclean
	rm -f maketmp tmp difftmp
	rm -f *~ heat stokes riemann testprealloc testts

