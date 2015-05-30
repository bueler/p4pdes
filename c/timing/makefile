all:



timing_c1tri: c1tri
	make c1tri
	

cstrip/%.c: ../c/%.c
	rm -rf $@.tmp

.PHONY: timing_c1tri clean

clean:
	@rm -f *~

