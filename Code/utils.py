import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib
from datetime import datetime
matplotlib.use("TkAgg")
from multiprocess import Pool, cpu_count
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import scipy as sci
# from copy import copy


def clean_data(df, type_dict):
    """
    Coerces the columns of df to match the definitions laid out in type_dict.

    :param df: a pandas dataframe
    :param type_dict: a dictionary with three keys that each contain a list of variable name

    type_dict['date_vars'] - contains a list of the names of variables that should be coerced
    to datetimes. The format should be '2000-01-31'.

    type_dict['float_vars'] - contains a list of the names of variables that should be coerced
    into floats.

    type_dict['int_vars'] - contains a list of the names of variables that should be coerced
    into ints.

    If the coercion is unsuccessful, a NaN is placed instead.

    :return:the dataframe with the new coerced values
    """
    print('Cleaning date variables:')
    for v in type_dict['date_vars']:
        print(v)
        df[v] = pd.to_datetime(df[v], format='%Y/%m/%d', errors='coerce', cache=True).dt.tz_localize(None) + MonthEnd(0)

    print('Cleaning numeric variables:')
    for v in type_dict['float_vars']:
        print(v)
        df[v] = pd.to_numeric(df[v], errors='coerce')

    print('Cleaning integer variables:')
    for v in type_dict['int_vars']:
        print(v)
        df[v] = pd.to_numeric(df[v], downcast='signed', errors='coerce')

    print('Final data types:')
    print(df.dtypes)

    return df

def check_unique(dataframe, identifier_list):
    """
    Verifies that dataframe is uniquely identified by the variables in identifier_list
    :param dataframe: a pandas dataframe
    :param identifier_list: a list of variable names. Must be contained in either the index
    or the variables of dataframe
    :return: a dataframe of identifiers that are not uniquely identified along with the counts of
    how many times they occurred. If the dataframe is uniquely identified, the function returns
    an empty dataframe
    """
    unique_identifier = dataframe.groupby(by = identifier_list).count().iloc[:, 0]
    unique_identifier.name = 'Count'
    unique_identifier = unique_identifier[unique_identifier > 1]
    return unique_identifier.to_frame()

def continuous_index(company_dataframe, num_months = 1):
    """
    Detects if a company has continuous return data. Returns "True" if the returns are continuous. False otherwise
    :param company_dataframe -- a dataframe from a groupby object, indexed on Permco and the time variable at level 2
    :param num_months -- the maximum number of months between timestamps
    :returns True or False, depending on whether the return sequence is continuous
    """
    times = company_dataframe.index.get_level_values(1)
    diffs = times.shift(1, freq = 'M') - times
    return (diffs.max().total_seconds() < 32 * num_months * 24 * 60 * 60)

def print_message(text):
    print('=====' + text + ' (' + str(datetime.now()) + ') =====')

def safe_index(self, variable_list):
    """
    Function that makes setting an index easier. Often times the existing index
    already contains some variables.
    :param self: a Pandas dataframe
    :param variable_list: a list of strings containing the new index names
    :return:
    """
    return self.reset_index().set_index(variable_list)
pd.DataFrame.safe_index = safe_index

def safe_drop(self, variable_list, **kwargs):
    safe_variables = list(set(self.columns) & set(variable_list))
    return self.drop(columns = safe_variables, **kwargs)
pd.DataFrame.safe_drop = safe_drop


def parallel_apply(df, group_list, f, num_cores, print_every_n=1000):
    """
    A lightweight version of apply using the multiprocess library

    :param df: the pandas dataframe that needs to be grouped
    :param group_list: a list of variable names to group by
    :param f: the function to apply to each group
    :param num_cores: the number of cores to use
    :param print_every_n: a message will be printed every print_every_n groups processed by
    each core. If print_every_n = None, then no messages will be printed

    :return: a dataframe that has had the apply function done on it

    Usage:

    NUM_GROUPS = 10000
    SIZE_OF_GROUP = 4
    N = NUM_GROUPS * SIZE_OF_GROUP
    df_one_group = pd.DataFrame({'g1': np.random.randint(low = 1, high = NUM_GROUPS, size = N),
                                'data': np.random.random(N) - 0.5})

    def slow_max(d):
        ret = d.sort_values(['data'], ascending = False)
        return ret.iloc[0, :]

    par = parallel_apply(df_one_group, ['g1'], slow_max, 4).head()
    serial = df_one_group.groupby(['g1']).apply(slow_max).head()
    assert(np.all(par['data'].values == serial['data'].values))
    """


    # Cut up the dataframes into a list
    num_cores = int(num_cores)
    group_numbers = df.groupby(by=group_list).ngroup()
    group_cuts = group_numbers.quantile(np.linspace(0, 1, num = num_cores + 1), interpolation='nearest').values
    df['_Group'] = group_numbers

    if print_every_n != None:
        print('Total of ' + str(group_cuts[-1]) + ' groups')

    cuts = []
    for ind in range(num_cores - 1):
        cuts.append(
            df.loc[(group_numbers >= group_cuts[ind]) & (group_numbers < group_cuts[ind + 1])].groupby(group_list))
    cuts.append(df.loc[(group_numbers >= group_cuts[num_cores - 1]) & (group_numbers <= group_cuts[num_cores])].groupby(
        group_list))

    # Define functions to be passed to parallel process
    def verbose_function(dataframe):
        curr = dataframe['_Group'].values[0]
        if curr % print_every_n == 0:
            print('Group: ' + str(curr))
        return f(dataframe)

    def verbose_func_to_apply(group_by_object):
        return group_by_object.apply(verbose_function)

    def silent_func_to_apply(group_by_object):
        return group_by_object.apply(f)

    with Pool(num_cores) as p:
        if print_every_n != None:
            parallel_results = p.map(verbose_func_to_apply, cuts)
        else:
            parallel_results = p.map(silent_func_to_apply, cuts)

    ret = pd.concat(parallel_results)

    df.safe_drop(['_Group'], inplace = True)
    ret.safe_drop(['_Group'], inplace = True)

    return ret

