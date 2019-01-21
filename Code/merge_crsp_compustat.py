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
merged = crsp.join(compustat, how = 'left', rsuffix = '.comp') # Use crsp data by default

# Combine some variables
# missing = lambda s: pd.isnull(merged[s])
# merged.loc[missing('Company Name'), 'Company Name'] = merged['Company Name.crsp']
# merged.safe_drop(['Company Name.crsp'], inplace = True)

print('Final Variables')
print(merged.columns.tolist())

print_message('Filling NAs')
merged = merged.groupby(by = ['Permco']).fillna(method = 'ffill')
merged.to_hdf('../Output/merged.h5', key = 'merged')