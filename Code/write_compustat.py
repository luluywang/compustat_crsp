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
    dataframe.safe_drop(['_Group'], inplace = True) # Addition to make it work well with the parallel process
    dataframe['Missing Count'] = dataframe.isnull().sum(axis = 1)
    dataframe.sort_values(by = ['Missing Count'], ascending = True, inplace = True)
    return dataframe.iloc[0, :]
compustat = parallel_apply(compustat, ['Permco', 'datadate'], select_least_missing, 2, 10000)

################ Reshaping the Annual Data ################
print_message('Converting yearly to quarterly data')
yearly_variables = sorted([compustat_names[x] for x in list(compustat_names.keys()) if x[-1] == 'y'])

def take_diffs(x):
    """
    Takes a Series of a "year to date" variable and converts it into a quarterly variable

    param x: a Series of a year to date variable, e.g. capex year to date
    :return: a Series of the quarterly variable
    """

    if all(np.isnan(x)):
        return x

    x = x.fillna(0)
    ret = x.diff()
    ret[0] = x[0]
    return ret

def take_diff_across_columns(dataframe):
    dataframe.safe_drop(['_Group'], inplace = True) # Addition to make it work well with the parallel process
    ret = dataframe.transform(take_diffs)
    return ret

compustat = compustat.safe_index(['Permco', 'datadate', 'Fiscal Year'])
compustat_yearly = compustat.loc[:, yearly_variables]
compustat_yearly = parallel_apply(compustat_yearly, ['Permco', 'Fiscal Year'], take_diff_across_columns, 2, None)
compustat[yearly_variables] = compustat_yearly
print_message('Finished parallel process')

################ Data Integrity ################

compustat = compustat.safe_index(['Permco', 'datadate'])

# Check now that Permco + datadate uniquely identify each observation
duplicated_values = check_unique(compustat, ['Permco', 'datadate'])
print('Duplicated Permco - datadate pairs')
print(duplicated_values)
assert(duplicated_values.shape[0] == 0)

# Check that dates are contiguous
valid_returns = parallel_apply(compustat, ['Permco'], continuous_index, 2, None)
discontinuous_returns = valid_returns.loc[~valid_returns]
print('Companies without continuous dates')
print(discontinuous_returns)
assert(discontinuous_returns.shape[0] == 0)

################ Outputting Data ################
print_message('Outputting Data')
compustat.to_hdf('../Output/compustat.h5', 'compustat')

with open('../Logs/compustat.log', 'w') as f:
    f.write('Raw Compustat file has been written')
