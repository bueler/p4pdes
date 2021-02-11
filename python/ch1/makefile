# testing
COUTPUTS := ../../c/ch1/output/

output:
	@ln -sf $(COUTPUTS)

rune_1: output
	-@../ptestit.sh e "" 1 1

test_e: rune_1

test: test_e

# etc

.PHONY: clean 

clean:
	@rm -f *tmp
	@rm -rf output

