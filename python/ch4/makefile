# testing
COUTPUT := ../../c/ch4/output/

output:
	@ln -sf $(COUTPUT)

runexpcircle_1: output
	-@../ptestit.sh expcircle "-snes_fd" 1 1

runexpcircle_2: output
	-@../ptestit.sh expcircle "-snes_mf" 1 2

test_expcircle: runexpcircle_1 runexpcircle_2

test: test_expcircle

# etc

.PHONY: clean runexpcircle_1 runexpcircle_2 test_expcircle test

clean:
	@rm -f *tmp
	@rm -rf output

