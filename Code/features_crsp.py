from utils import *
import numpy as np

crsp = pd.read_hdf('../Output/crsp.h5', 'crsp')

################ Generate Features ################
crsp['Price Volume'] = crsp['Price'] * crsp['Volume']
# todo: make it generate cumulative returns

crsp.to_hdf('../Output/crsp.h5', 'crsp')