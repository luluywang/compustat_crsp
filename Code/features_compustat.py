from utils import *

compustat = pd.read_hdf('../Output/compustat.h5', 'compustat')

################ Creating Features ################

# Make some variables
print('Making features')
compustat['Market Cap (Compustat)'] = compustat['Price (Compustat)'] * \
                                      compustat['Shares Outstanding (Compustat)'] / 1e3
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Common Equity, Total'] + compustat['Preferred Equity, Total'], compustat['Shareholder Equity, Total'])
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Assets, Total'] - compustat['Liabilities, Total'], compustat['Shareholder Equity, Total'])
compustat['Book Equity'] = compustat['Shareholder Equity, Total'] + compustat['Deferred Tax Assets'] - compustat['Preferred Equity, Total']

compustat.to_hdf('../Output/compustat.h5', 'compustat')