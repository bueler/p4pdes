# list the targets for which we run a script to generate
# (note there are both .png and .pdf figs which are not re-generated)
pdfroots := poisson-conv poisson-cg-scale richpolyplot expcirclebasic newtonconvbasic newtonconvdelayed reaction-conv minsurf plap-conv plap-perrs plap-piters plap-conditionnumber ode unfem-snesfd unfem-error
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

minsurf.pdf: minsurf.tex
	pdflatex minsurf
	pdfcrop minsurf.pdf minsurf.pdf

plap-conv.pdf plap-perrs.pdf plap-piters.pdf: plap-conv.py
	python plap-conv.py

plap-conditionnumber.pdf: plap-conditionnumber.py
	python plap-conditionnumber.py

ode.pdf: ode-plot.py
	python ode-plot.py

#FIXME unfem-snesfd.pdf unfem-error.pdf unfem-times.pdf: unfem-graphs.py
unfem-snesfd.pdf unfem-error.pdf: unfem-graphs.py
	python unfem-graphs.py

.PHONY: clean

clean:
	@rm -f $(pdfgens)
	@rm -f *~ *.aux *.log

