# list the targets for which we run a script to generate
# (note there are both .png and .pdf figs which are not re-generated)
pdfroots := poisson-conv poisson-cg-scale richpolyplot expcirclebasic newtonconvbasic newtonconvdelayed reaction-conv
pdfgens := $(addsuffix .pdf, $(pdfroots))

all: $(pdfgens)

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
	@rm -f richpolyplot.pdf poisson-conv.pdf poisson-cg-scale.pdf expcirclebasic.pdf newtonconv*pdf reaction-conv.pdf

