all: book.pdf

texroots := macros preface getstarted linearsystem structured nonlinear optifem unstructured multigrid scaling stokes dmplex constrained
texinputs := $(addsuffix .tex, $(texroots))

# for these, see strip.sh
croots := vecmatksp tri structuredpoisson poisson expcircle expcircleJAC reaction fish2 readmesh poissontools poissonfem obstacle
cstrip := $(addprefix cstrip/, $(addsuffix .c, $(croots)))

tikzinputs := samplepoly.1.tikz squarefour.tikz bump.1.tikz bump.poly.tikz mesh.1.tikz

pdfroots := petscghostvalues
pdfinputs := $(addprefix figs/, $(addsuffix .pdf, $(pdfroots)))

pngroots := dmM5N7 line-graph-tri matM5N7
pnginputs := $(addprefix figs/, $(addsuffix .png, $(pngroots)))

book.pdf: book.aux
	pdflatex book
	pdflatex book

book.aux: book.tex book.bib $(texinputs) $(cstrip) $(tikzinputs) $(pdfinputs) $(pnginputs)
	(cd figs/ && make)
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

.PHONY: clean

clean:
	rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	rm -rf cstrip/
	(cd figs/ && make clean)
	rm -f bump.?.* samplepoly.?.* bump.poly* squarefour.tikz mesh.?.*
	rm -f timing/*~ timing/poisson/*~ timing/tri/*~

