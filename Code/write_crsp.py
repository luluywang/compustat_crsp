"""
Description: Cleans the crsp database to only include entries with valid returns and
so that every stock is uniquely identified by Permco-datadate.


I rename some variables, and collapse share classes. In the combined share classes, I
add up the market cap of each share class to get to the total market cap of that
company at that time. Returns are also computed as market cap weighted returns of the
individual share classes.
"""

# Custom functions for data cleaning
from utils import *
import numpy as np
from copy import copy
from multiprocessing import Pool, cpu_count

# Proper data cleaning
################ Import Data ################
print_message('Loading Data')
crsp_raw = pd.read_csv('../Data/crsp.txt', sep = '\t', low_memory = False)#, nrows = 10000)

# Filter down to only common shares
crsp_raw = crsp_raw.loc[np.floor(crsp_raw['SHRCD'] / 10) == 1]

print('Data Size:')
print(crsp_raw.shape)
print(str(crsp_raw.shape[0] * crsp_raw.shape[1]) + ' observations')

################ Setting Types ################
print_message('Setting Types')
crsp_datatypes = {'date_vars': ['date'],
                 'float_vars': ['BIDLO', 'ASKHI', 'PRC', 'RET', 'DLRET', 'BID', 'ASK', 'RETX', 'CFACPR', 'CFACSHR'],
                 'int_vars': ['PERMNO', 'PERMCO', 'HSICCD', 'CUSIP', 'SHROUT', 'VOL', 'EXCHCD']}
crsp = clean_data(copy(crsp_raw), crsp_datatypes)

################ Setting Types ################
print_message('Renaming Variables')
crsp_names = {'RET': 'Return',
              'SHROUT': 'Shares Outstanding on Trading Day',
              'COMNAM': 'Company Name',
              'EXCHCD': 'Exchange Code',
              'TICKER': 'Ticker',
              'date': 'datadate',
              'PERMNO': 'Permno',
              'PERMCO': 'Permco',
              'PRC': 'Price with Flag',
              'BID': 'Bid',
              'ASK': 'Ask',
              'VOL': 'Volume on Trading Day',
              'SHRCLS': 'Share Class',
              'CFACPR': 'Price Adjustment Factor',
              'CFACSHR': 'Share Adjustment Factor'}

crsp = crsp.rename(index = str, columns = crsp_names)
crsp = crsp[list(crsp_names.values())]


# Make a few more useful variables
print_message('Create new features')
crsp['Imputed Price'] = (crsp['Price with Flag'] < 0)
crsp['Price on Trading Day'] = np.abs(crsp['Price with Flag'])
crsp['Price'] = crsp['Price on Trading Day'] / crsp['Price Adjustment Factor']
crsp['Shares Outstanding'] = crsp['Shares Outstanding on Trading Day'] * crsp['Share Adjustment Factor']
crsp['Volume'] = crsp['Volume on Trading Day'] * crsp['Share Adjustment Factor']
crsp['Market Cap (Billions, CRSP)'] = crsp['Shares Outstanding'] * crsp['Price'] / 1e6
crsp = crsp.drop(['Price on Trading Day', 'Shares Outstanding on Trading Day', 'Volume on Trading Day'], axis = 1)

################
print_message('Aggregating share classes')
AGG_VAR_TYPES = {'First': ['Company Name', 'Permno', 'Ticker', 'Price', 'Bid', 'Ask', 'Exchange Code'],
                 'Add': ['Market Cap (Billions, CRSP)'],
                 'Weighted Sum': ['Volume', 'Return']}
ALL_CRSP_VAR = [item for sublist in AGG_VAR_TYPES.values() for item in sublist]

def agg_share_classes(company_on_date):
    """
    Given a dataframe grouped on Permco and datadate, aggregates up the key variables from the individual
    share classes. THe operation done on each variable is defined by the global dictionary AGG_

    :param company_on_date - a dataframe from a pandas groupby object with all the variables inside ALL_CRSP_VAR
    :return a dataframe with one row that has all the variables in ALL_CRSP_VAR, agrgregated across the share classes
    """

    if ('Loop Number' in company_on_date.columns) and company_on_date['Loop Number'].values[0] % 1000 == 0:
        print('Group: ' + str(company_on_date['Loop Number'].values[0]) + ' | ' + str(company_on_date.index[0]))

    # If only one share class
    if company_on_date.shape[1] <= 1:
        return company_on_date[ALL_CRSP_VAR]

    # Then for the remainign cases
    company_on_date['Market Cap (Billions, CRSP)'].fillna(0)
    company_on_date.sort_values(by = ['Market Cap (Billions, CRSP)'], ascending = False, inplace = True)
    new_frame = company_on_date.iloc[0, :][AGG_VAR_TYPES['First']]

    for v in AGG_VAR_TYPES['Add']:
        new_frame[v] = company_on_date[v].fillna(0).sum()

    for v in AGG_VAR_TYPES['Weighted Sum']:
        valid = ~company_on_date[v].isnull()
        denominator = company_on_date.loc[valid, 'Market Cap (Billions, CRSP)'].sum()

        if denominator > 0:
            new_frame[v] = (company_on_date.loc[valid, v] * company_on_date.loc[
                valid, 'Market Cap (Billions, CRSP)']).sum() / denominator
        else:
            new_frame[v] = company_on_date[v][0]

    return new_frame

# First, pull every permco-date that isn't paired
crsp = crsp.set_index(['Permco', 'datadate'])
crsp_counts = check_unique(crsp, ['Permco', 'datadate'])

# Then isolate the problem observations
crsp_merge = crsp_counts.join(crsp, how = 'outer')
problem_children = crsp_merge.loc[~pd.isnull(crsp_merge['Count']), ALL_CRSP_VAR]

if problem_children.shape[0] > 0:
    good_children = crsp_merge.loc[pd.isnull(crsp_merge['Count']), ALL_CRSP_VAR]

    # Apply the function only on the problem observations
    problem_children['Loop Number'] = list(range(problem_children.shape[0]))
    print('Total Groups: ' + str(problem_children.shape[0]))
    merged_problems = problem_children.groupby(by = ['Permco', 'datadate']).apply(agg_share_classes)
    print(merged_problems.head())
    merged_problems = merged_problems[ALL_CRSP_VAR]

    # Combine the dataframes together again
    crsp_merge = pd.concat([good_children[ALL_CRSP_VAR], merged_problems])

print_message('Dropping Null Returns')
crsp_merge = crsp_merge.loc[~pd.isnull(crsp_merge['Return'])]

#######
print_message('Verifying data integrity')
crsp_merge = crsp_merge.reset_index().set_index(['Permco', 'datadate'])

# Check final assumptions on the data
# 1. Permco + datadate uniquely identify each observation
# 2. For each permco, there is a continuous stream of returns without interruption

# Unique identification
duplicated_values = check_unique(crsp_merge, ['Permco', 'datadate'])
print(duplicated_values)
assert(duplicated_values.shape[0] == 0)

# Continuity of returns
valid_returns = crsp_merge.groupby(by = ['Permco']).apply(continuous_index)
discontinuous_returns = valid_returns.loc[~valid_returns]
print('Companies without continuous returns')
print(discontinuous_returns)
assert(discontinuous_returns.shape[0] == 0)

################
print_message('Outputting Data')
crsp_merge.to_hdf('../Output/crsp.h5', 'crsp')
