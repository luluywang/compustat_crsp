import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib
from datetime import datetime
matplotlib.use("TkAgg")
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import scipy as sci
# from copy import copy
# from multiprocessing import Pool, cpu_count


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