all: structure.pdf

texinputs := linear.tex

cinputs := c1matvec.c c2triangle.c
cstrip := $(addprefix cstrip/, $(cinputs))

# TO BUILD PDF:
#     ice-bib.bib  needs to be a link to   pism/doc/ice-bib.bib

structure.pdf: structure.tex structure.aux $(texinputs) $(cstrip)
	pdflatex structure

structure.aux: structure.tex $(texinputs) $(cstrip)
	pdflatex structure
	bibtex structure.aux

# The purpose of the following nonsense is to put only the "meat" of codes
# into the book.  We remove ierr stuff.
cstrip/%.c: ../c/%.c
	sed -e 's/ierr = //' $< > $@.tmp
	sed -e 's/ CHKERRQ(ierr);//' $@.tmp > $@
	rm -rf $@.tmp

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
