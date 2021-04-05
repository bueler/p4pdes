# testing

runfish_1:
	-@../../c/testit.sh fish.py "-mx 5 -my 5 -k 2 -s_ksp_type preonly -s_pc_type lu" 1 1

runfish_2:
	-@../../c/testit.sh fish.py "-refine 2 -dm_view -s_ksp_converged_reason -s_pc_type mg -s_mg_levels_ksp_max_it 3" 2 2

runfish_3:
	-@../../c/testit.sh fish.py "-refine 1 -dm_view -s_snes_type newtonls -s_snes_converged_reason -s_ksp_converged_reason -s_pc_type gamg" 1 3

runfish_4:
	-@../../c/testit.sh fish.py "-fishhelp" 1 4

runfish_5:
	-@../../c/testit.sh fish.py "-dm_view" 10 5 # zero elements on some ranks

runfish_6:
	-@../../c/testit.sh fish.py "-refine 1 -quad -s_ksp_converged_reason -s_pc_type mg -s_mg_levels_ksp_type richardson -s_mg_levels_pc_type icc" 1 6

test_fish: runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6

test: test_fish

# etc

.PHONY: clean runfish_1 runfish_2 runfish_3 runfish_4 runfish_5 runfish_6 test_fish test

clean:
	@rm -f *.pyc *.geo *.msh *.pvd *.pvtu *.vtu *.m maketmp tmp difftmp

