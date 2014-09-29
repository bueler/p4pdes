all: structure.pdf

tikzinputs := bump.1.tikz bump.2.tikz
texinputs := getstarted.tex linear.tex

cinputs := c1matvec.c c2triangle.c
cstrip := $(addprefix cstrip/, $(cinputs))

# TO BUILD PDF:
#     ice-bib.bib  needs to be a link to   pism/doc/ice-bib.bib

structure.pdf: structure.tex structure.aux $(texinputs) $(cstrip) $(tikzinputs)
	pdflatex structure
	pdflatex structure

structure.aux: structure.tex $(texinputs) $(cstrip) $(tikzinputs)
	pdflatex structure
	bibtex structure.aux

# The purpose of the following nonsense is to put only the "meat" of codes
# into the book.  We remove ierr stuff.
cstrip/%.c: ../c/%.c
	sed -e 's/ierr = //' $< > $@.tmp
	sed -e 's/ CHKERRQ(ierr);//' $@.tmp > $@
	rm -rf $@.tmp

bump.1.tikz: ../c/bump.poly
	(cd ../c/ && triangle -pqa1.0 bump)
	rm -rf bump.1.node bump.1.ele bump.1.poly
	ln -s ../c/bump.1.node
	ln -s ../c/bump.1.ele
	ln -s ../c/bump.1.poly
	python tri2tikz.py --labelnodes --scale 2.0 --labeloffset 0.25 bump.1 bump.1.tikz

bump.2.tikz: ../c/bump.poly ../c/bump.1.poly
	(cd ../c/ && triangle -rpqa0.25 bump.1)
	rm -rf bump.2.node bump.2.ele bump.2.poly
	ln -s ../c/bump.2.node
	ln -s ../c/bump.2.ele
	ln -s ../c/bump.2.poly
	python tri2tikz.py --scale 2.0 bump.2 bump.2.tikz

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi cstrip/*.c
