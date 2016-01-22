all: book.pdf

texroots := macros preface getstarted linearsystem structured nonlinear optifem advdif precond scaling unstructured dmplex stokes constrained
texinputs := $(addsuffix .tex, $(texroots))

# for these, see strip.sh
croots := vecmatksp tri poisson expcircle expcircleJAC reaction plap fish2 readmesh poissontools poissonfem obstacle
cstrip := $(addprefix cstrip/, $(addsuffix .c, $(croots)))

tikzinputs := blob.tikz blob.1.elenum.tikz blob.1.nodenum.tikz squarefour.tikz mesh.1.tikz

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

blob.tikz blob.1.elenum.tikz blob.1.nodenum.tikz: ../c/ch9/blob.poly
	python tri2tikz.py --polyonly --nodesize 1.0 --scale 0.6 ../c/ch9/blob blob.tikz
	triangle -pqa4.0 ../c/ch9/blob
	python tri2tikz.py --labelelements --nodesize 1.0 --scale 0.6 ../c/ch9/blob.1 blob.1.elenum.tikz
	python tri2tikz.py --labelnodes --nodesize 3.0 --nodeoffset -0.5 --scale 0.6 ../c/ch9/blob.1 blob.1.nodenum.tikz

squarefour.tikz: squarefour.poly
	python tri2tikz.py --labelnodes --nodesize 0.5 --nodeoffset -0.1 --scale 3.0 squarefour $@

mesh.1.tikz: ../c/ch9/genstructured.py
	(cd ../c/ch9/ && python genstructured.py mesh.1 5)
	python tri2tikz.py --labelnodes --nodeoffset 0.05 --nodesize 0.5 --labelelements --scale 5.0 ../c/ch9/mesh.1 $@

.PHONY: clean

clean:
	rm -f *~ *.out *.aux *.log *.bbl *.blg *.synctex.gz *.idx *.ilg *.ind *.toc *.dvi
	rm -rf cstrip/
	(cd figs/ && make clean)
	rm -f blob*.tikz squarefour.tikz mesh.?.*
	rm -f timing/*~ timing/poisson/*~ timing/tri/*~

