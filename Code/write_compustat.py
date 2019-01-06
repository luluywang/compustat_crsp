"""
Description: Cleans the compustat database. I rename some variables, and collapse
share classes down so that each company is uniquely identified by Permco-datadate.
I also construct some basic features (mkt cap, book equity). In building out the
features, I also "quarterly-ize" the cash flow numbers, which are all provided on a
fiscal year to date basis from Compustat

Author: Lulu

Last Updated: Jan 5, 2018
"""

# Custom functions
from utils import *
import numpy as np
from copy import copy
from multiprocessing import Pool, cpu_count

################ Import Data ################
print_message('Loading Data')
compustat_raw = pd.read_csv('../Data/compustat-merged.txt', sep = '\t', low_memory = False)#, nrows = 10000)

################ Types ################
print_message('Defining Types')
compustat_datatypes = {'date_vars': ['datadate'],
                 'float_vars': ['cogsq', 'cshopq', 'cshoq', 'dpq', 'oiadpq', 'oibdpq', 'prcraq', 'saleq', 'txpq', 'xintq', 'xsgaq', 'aqcy', 'capxy', 'dvy', 'fincfy', 'oancfy', 'prccq', 'seqq', 'ceqq', 'pstkrq', 'atq', 'ltq', 'cheq'],
                 'int_vars': ['GVKEY', 'LPERMNO', 'LPERMCO', 'fyearq', 'fqtr', 'cusip', 'exchg', 'cik', 'naics']}
compustat = clean_data(copy(compustat_raw), compustat_datatypes)

################ Renaming Concepts ################
print_message('Renaming Features')
compustat_names = \
    {  # Identifiers
        'GVKEY': 'Gvkey',
        'LPERMNO': 'Permno',
        'LPERMCO': 'Permco',
        'fyearq': 'Fiscal Year',
        'fqtr': 'Fiscal Quarter',
        'conm': 'Company Name',
        'curcdq': 'Currency',
        'rdq': 'Report Date',
        'naics': 'NAICS Sector Code',
        'exchg': 'Exchange Code',
        'datadate': 'datadate',

        # Balance Sheet
        'atq': 'Assets, Total',
        'ceqq': 'Common Equity, Total',
        'seqq': 'Shareholder Equity, Total',
        'pstkrq': 'Preferred Equity, Total',
        'ltq': 'Liabilities, Total',
        'txditcq': 'Deferred Tax Assets',
        'dlttq': 'Long Term Debt',
        'dlcq': 'Short Term Debt',
        'cheq': 'Cash',

        # Income Statement
        'saleq': 'Sales',
        'cogsq': 'COGS',
        'xsgaq': 'SG&A',
        'dpq': 'Depreciation and Amortization',
        'oibdpq': 'EBITDA',
        'oiadpq': 'EBIT',
        'xintq': 'Interest Expense',
        'txpq': 'Taxes Payable',
        'niq': 'Net Income',
        'epsfiq': 'Diluted EPS, Raw',
        'epsfxq': 'Diluted EPS, Adjusted',

        # Cash Flow Statement
        'oancfy': 'Operating Cash Flow',
        'fincfy': 'Financing Activities',
        'dltisy': 'Long Term Debt, Gross Issuance',
        'dltry': 'Long Term Debt, Retired',
        'dvy': 'Cash Dividends',
        'capxy': 'Capex',
        'aqcy': 'M&A',
        'cshopq': 'Total Repurchased Shares',
        'prcraq': 'Repurchase Price',

        # Market Data
        'cshoq': 'Shares Outstanding (Compustat)',
        'prccq': 'Price (Compustat)',
        'cshfdq': 'Shares Outstanding for EPS'}
compustat = compustat.rename(index=str, columns=compustat_names)
compustat = compustat[list(compustat_names.values())]
compustat = compustat.set_index(['Permco', 'datadate'])

################ Dropping to Unique Identifiers ################
# At this level, data should be identified by Permco - datadate. Count the number of non-valid variables for
# each combination, pick the one with the fewer null entries

print_message('Generating Unique Identifiers')
def select_least_missing(dataframe):
    if dataframe['Group Number'].values[0] % 10000 == 0:
        print('Group: ' + str(dataframe['Group Number'].values[0]) + ' | ' + str(dataframe.index[0]))

    dataframe['Missing Count'] = dataframe.isnull().sum(axis = 1)
    dataframe.sort_values(by = ['Missing Count'], ascending = True, inplace = True)
    return dataframe.iloc[0, :]

compustat['Group Number'] = compustat.groupby(['Permco', 'datadate']).ngroup()
print('Maximum group number: ' + str(compustat['Group Number'].max()))
compustat = compustat.groupby(['Permco', 'datadate']).apply(select_least_missing)
compustat = compustat.drop(['Group Number'], axis = 1)

# Check now that Permco + datadate uniquely identify each observation
duplicated_values = check_unique(compustat, ['Permco', 'datadate'])
print(duplicated_values)
assert(duplicated_values.shape[0] == 0)

################ Creating Features ################
print_message('Converting yearly to quarterly data')
yearly_variables = sorted([compustat_names[x] for x in list(compustat_names.keys()) if x[-1] == 'y'])

compustat = compustat.reset_index().set_index(['Permco', 'Fiscal Year'])
cashflow_groups = compustat[yearly_variables].groupby(['Permco', 'Fiscal Year'])

def take_diffs(x):
    """
    Takes a Series of a "year to date" variable and converts it into a quarterly variable

    param x: a Series of a year to date variable, e.g. capex year to date
    :return: a Series of the quarterly variable
    """
    if x.name[1] == 2000:
        print(x.name)

    if all(np.isnan(x)):
        return x

    x = x.fillna(0)
    ret = x.diff()
    ret[0] = x[0]
    return ret

def cash_flow(variable):
    print(variable)
    return cashflow_groups[variable].transform(take_diffs)

with Pool(4) as p:
    new_dataframes = p.map(cash_flow, yearly_variables)

print_message('Finished parallel process')
# Assign the new variables
for d in new_dataframes:
    compustat[d.name] = d

# Make some variables
print('Making features')
compustat['Market Cap (Compustat)'] = compustat['Price (Compustat)'] * \
                                      compustat['Shares Outstanding (Compustat)'] / 1e3
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Common Equity, Total'] + compustat['Preferred Equity, Total'], compustat['Shareholder Equity, Total'])
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Assets, Total'] - compustat['Liabilities, Total'], compustat['Shareholder Equity, Total'])
compustat['Book Equity'] = compustat['Shareholder Equity, Total'] + compustat['Deferred Tax Assets'] - compustat['Preferred Equity, Total']

################ Data Integrity ################
compustat = compustat.reset_index().set_index(['Permco', 'datadate'])

# Check that dates are contiguous
valid_returns = compustat.groupby(by = ['Permco']).apply(continuous_index)
discontinuous_returns = valid_returns.loc[~valid_returns]
print('Companies without continuous dates')
print(discontinuous_returns)
assert(discontinuous_returns.shape[0] == 0)

################ Outputting Data ################
print_message('Outputting Data')
compustat.to_hdf('../Output/compustat.h5', 'compustat')
