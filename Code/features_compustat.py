################ Creating Features ################
print_message('Converting yearly to quarterly data')
yearly_variables = sorted([compustat_names[x] for x in list(compustat_names.keys()) if x[-1] == 'y'])

compustat = compustat.safe_index(['Permco', 'Fiscal Year'])
compustat['Group Number'] = compustat.groupby(['Permco', 'Fiscal Year']).ngroup()
print('Maximum Group Number: ' + str(compustat['Group Number'].max()))
compustat = compustat.safe_index(['Permco', 'Fiscal Year', 'Group Number'])
cashflow_groups = compustat[yearly_variables].groupby(['Permco', 'Fiscal Year', 'Group Number'])

def take_diffs(x, var_type):
    """
    Takes a Series of a "year to date" variable and converts it into a quarterly variable

    param x: a Series of a year to date variable, e.g. capex year to date
    :return: a Series of the quarterly variable
    """

    group_number = x.name[2]
    if group_number % 10000 == 0:
        print(var_type + ': ' + str(group_number))

    if all(np.isnan(x)):
        return x

    x = x.fillna(0)
    ret = x.diff()
    ret[0] = x[0]
    return ret

def cash_flow(variable):
    return cashflow_groups[variable].transform(take_diffs, var_type = variable)

with Pool(4) as p:
    new_dataframes = p.map(cash_flow, yearly_variables)

print_message('Finished parallel process')
# Assign the new variables
for d in new_dataframes:
    compustat[d.name] = d
compustat = compustat.safe_index(['Permco', 'datadate'])
compustat = compustat.drop(['Group Number'], axis = 1)

# Make some variables
print('Making features')
compustat['Market Cap (Compustat)'] = compustat['Price (Compustat)'] * \
                                      compustat['Shares Outstanding (Compustat)'] / 1e3
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Common Equity, Total'] + compustat['Preferred Equity, Total'], compustat['Shareholder Equity, Total'])
compustat['Shareholder Equity, Total'] = np.select(compustat['Shareholder Equity, Total'].isnull(), \
                                                   compustat['Assets, Total'] - compustat['Liabilities, Total'], compustat['Shareholder Equity, Total'])
compustat['Book Equity'] = compustat['Shareholder Equity, Total'] + compustat['Deferred Tax Assets'] - compustat['Preferred Equity, Total']

