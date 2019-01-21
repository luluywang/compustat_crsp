from utils import *
import numpy as np
import time
from copy import copy

NUM_CORES = 4

print_message('Loading Data')
crsp = pd.read_hdf('../Output/crsp.h5', 'crsp')
crsp = crsp.safe_index(['Permco', 'datadate'])
print(crsp.head())

################ Generate Features ################
print_message('Building features')
crsp['Price Volume (Billions)'] = crsp['Price'] * crsp['Volume'] / 1e9
crsp['Log Return'] = np.log(1 + crsp['Return'])
crsp['Cumulative Return'] = crsp['Log Return'].groupby(['Permco'], group_keys = False).cumsum()
crsp['Price Volume (3mma)'] = crsp['Price Volume (Billions)'].groupby(['Permco'], group_keys = False).rolling(3).mean()
crsp['Market Cap (3mma)'] = crsp['Market Cap (Billions, CRSP)'].groupby(['Permco'], group_keys = False).rolling(3).mean()
crsp['Volume (% of Market Cap, 3mma)'] = crsp['Price Volume (3mma)'] / crsp['Market Cap (3mma)']

################ Writing ################
crsp.to_hdf('../Output/crsp.h5', 'crsp')

