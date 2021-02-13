# testing
COUTPUT := ../../c/ch4/output/

output:
	@ln -sf $(COUTPUT)

runexpcircle_1: output
	-@../ptestit.sh expcircle "-snes_fd" 1 1

runexpcircle_2: output
	-@../ptestit.sh expcircle "-snes_mf" 1 2

runecjac_1: output
	-@../ptestit.sh ecjac "" 1 1

runecjac_2: output
	-@../ptestit.sh ecjac "-snes_mf_operator" 1 2

test_expcircle: runexpcircle_1 runexpcircle_2

test_ecjac: runecjac_1 runecjac_2

test: test_expcircle test_ecjac

# etc

.PHONY: clean runexpcircle_1 runexpcircle_2 runecjac_1 runecjac_2 test_expcircle test_ecjac test

clean:
	@rm -f *tmp
	@rm -rf output

