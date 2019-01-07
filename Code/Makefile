D = ../Data
O = ../Output
L = ../Logs

crsp: $(O)/crsp.h5

compustat: $(O)/compustat.h5

merged: $(O)/merged.h5

$(L)/crsp.log: $(D)/crsp.txt write_crsp.py
	python write_crsp.py

$(O)/crsp.h5: $(L)/crsp.log features_crsp.py
	python features_crsp.py

$(L)/compustat.log: $(D)/compustat-merged.txt write_compustat.py
	python write_compustat.py

$(O)/compustat.h5: $(L)/compustat.log write_compustat.py
	python features_compustat.py

$(O)/merged.h5: $(O)/crsp.h5 $(O)/compustat.h5
	python merge_crsp_compustat.py