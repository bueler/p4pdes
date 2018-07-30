# testing

runfish_1:
	-@../../c/testit.sh fish.py "-mx 5 -my 5 -order 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfish_2:
	-@../../c/testit.sh fish.py "-refine 2 -s_ksp_converged_reason -s_pc_type mg" 1 2

test_fish: runfish_1 runfish_2

test: test_fish

# etc

.PHONY: clean runfish_1 runfish_2 test test_fish

.PHONY: clean

clean:
	@rm -f *.pyc *.pvd *.vtu

