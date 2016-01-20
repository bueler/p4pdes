include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

.PHONY: test

test:
	@(cd ch1/ && make -s rune_1)
	@(cd ch2/ && make -s runvecmatksp_1 runtri_1 runtri_2)
	@(cd ch3/ && make -s runpoisson_1 runpoisson_2 runpoisson_3)
	@(cd ch4/ && make -s runexpcircle_1 runexpcircle_2 runecjacobian_1 runecjacobian_2 runreaction_1 runreaction_2 runreaction_3 runreaction_4)
	@(cd ch5/ && make -s runplap_1 runplap_2 runplap_3 runplap_4 runplap_5 runplap_6)

# etc

.PHONY: distclean rdistclean

CHAPTERS = ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch9
rdistclean:
	for DIR in $(CHAPTERS); do \
	     (cd $$DIR; rm -f output/*~ solns/*~; ${MAKE} distclean; rm -f maketmp tmp difftmp); \
	done

distclean: rdistclean
	@rm -f *~

