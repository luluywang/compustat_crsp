from utils import *

"""
Description: Merges the CRSP and compustat databases. Final database is "resampled"
to the monthly CRSP frequency. The fundamental data on a given date is the most recent
non-null fundamental data from Compustat. Therefore the data lag changes over the
course of the year.

Author: Lulu

Last Updated: Jan 5, 2018
"""

crsp = pd.read_hdf('../Output/crsp.h5')
compustat = pd.read_hdf('../Output/compustat.h5')

compustat_variables = compustat.columns.tolist()

print_message('Merging')
merged = crsp.join(compustat, how = 'left', lsuffix = '.crsp', rsuffix = '.comp')

print_message('Filling NAs')
merged = merged.groupby(by = ['Permco']).fillna(method = 'ffill')
pd.to_hdf('../Output/merged.h5', key = 'merged')