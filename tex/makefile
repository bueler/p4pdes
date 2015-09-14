all: book.pdf

texroots := preface getstarted linearsystem structured nonlinear optifem unstructured multigrid scaling stokes dmplex constrained
texinputs := $(addsuffix .tex, $(texroots))

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.poly.tikz mesh.1.tikz

pdfinputs := poisson-conv.pdf poisson-cg-scale.pdf richpolyplot.pdf expcirclebasic.pdf newtonconvbasic.pdf newtonconvdelayed.pdf reaction-conv.pdf

hinputs :=

# for these, see strip.sh
cinputs := vecmatksp.c tri.c structuredpoisson.c poisson.c expcircle.c expcircleJAC.c reaction.c fish2.c readmesh.c poissontools.c poissonfem.c obstacle.c
cstrip := $(addprefix cstrip/, $(cinputs))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib $(texinputs) $(hinputs) $(cstrip) $(tikzinputs) $(pdfinputs)
	pdflatex book
	bibtex book.aux

$(cstrip):
	./strip.sh

samplepoly.1.tikz: samplepoly.poly
	triangle -pqa4.0 samplepoly
	python tri2tikz.py --labelnodes --nodesize 2.0 --nodeoffset -0.3 --labelelements --scale 0.6 samplepoly.1 $@

squarefour.tikz: squarefour.poly
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 squarefour $@

bump.1.tikz: ../c/ch8/bump.poly
	(cd ../c/ch8/ && triangle -pqa1.0 bump)
	python tri2tikz.py --labelnodes --nodeoffset 0.25 --labelelements --scale 1.7 ../c/ch8/bump.1 $@

bump.poly.tikz: ../c/ch8/bump.poly
	python tri2tikz.py --scale 0.85 --polyonly ../c/ch8/bump --nodesize 3.0 bump.poly.tikz

mesh.1.tikz: ../c/ch8/genstructured.py
	(cd ../c/ch8/ && python genstructured.py mesh.1 5)
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 ../c/ch8/mesh.1 $@

richpolyplot.pdf: richpolyplot.py
	python richpolyplot.py

poisson-conv.pdf: poisson-conv.py
	python poisson-conv.py

poisson-cg-scale.pdf: poisson-cg-scale.py
	python poisson-cg-scale.py

expcirclebasic.pdf newtonconvbasic.pdf newtonconvdelayed.pdf: expcircle.py
	python expcircle.py
	pdfcrop expcirclebasic.pdf expcirclebasic.pdf
	pdfcrop newtonconvbasic.pdf newtonconvbasic.pdf
	pdfcrop newtonconvdelayed.pdf newtonconvdelayed.pdf

reaction-conv.pdf: reaction-conv.py
	python reaction-conv.py

.PHONY: clean

clean:
	@rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	@rm -rf cstrip/
	@rm -f bump.?.* samplepoly.?.* bump.poly* squarefour.tikz mesh.?.*
	@rm -f poisson-conv.pdf poisson-cg-scale.pdf richpolyplot.pdf expcirclebasic.pdf newtonconv*pdf reaction-conv.pdf
	@rm -f timing/*~ timing/poisson/*~ timing/tri/*~

