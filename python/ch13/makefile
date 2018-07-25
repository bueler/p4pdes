# testing

runfirefish_1:
	-@../../c/testit.sh firefish.py "-mx 5 -my 5 -order 2 -s_ksp_type preonly -s_pc_type lu" 1 1

test_firefish: runfirefish_1

test: test_firefish

# etc

.PHONY: clean runfirefish_1 test test_firefish

.PHONY: clean

clean:
	@rm -f *.pyc *.pvd *.vtu

