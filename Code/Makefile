D = ../Data
O = ../Output

crsp: $(O)/crsp.h5

compustat: $(O)/compustat.h5

merged: $(O)/merged.h5

$(O)/crsp.h5: $(D)/crsp.txt write_crsp.py
	python write_crsp.py

$(O)/compustat.h5: $(D)/compustat-merged.txt write_compustat.py
	python write_compustat.py

$(O)/merged.h5: $(O)/crsp.h5 $(O)/compustat.h5
	python merge_crsp_compustat.py