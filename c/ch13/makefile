include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

tiny: tiny.o plexview.o chkopts
	-${CLINKER} -o tiny tiny.o plexview.o ${PETSC_LIB}
	${RM} tiny.o plexview.o

# testing

runtiny_1:
	-@../testit.sh tiny "-dm_refine 1 -plex_view_ranges -plex_view_use_height -plex_view_cell_cones" 1 1

runtiny_2:
	-@../testit.sh tiny "-tny_by_hand -plex_view_ranges -plex_view_coords" 2 2

runtiny_3:
	-@../testit.sh tiny "-section_view" 1 3

test_tiny: runtiny_1 runtiny_2 runtiny_3

test: test_tiny

# etc

.PHONY: distclean runtiny_1 runtiny_2 runtiny_3 test test_tiny

distclean:
	@rm -f *~ tiny *tmp

