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
    unique_identifier = dataframe.groupby(by = identifier_list).count().iloc[:, 0]
    unique_identifier.name = 'Count'
    unique_identifier = unique_identifier[unique_identifier > 1]
    return unique_identifier.to_frame()

def continuous_index(company_dataframe, num_months = 1):
    """
    Detects if a company has continuous return data. Returns "True" if the returns are continuous. False otherwise
    :param company_dataframe -- a dataframe from a groupby object, indexed on Permco
    :returns True or False, depending on whether the return sequence is continuous
    """
    times = company_dataframe.index.get_level_values(1)
    diffs = times.shift(1, freq = 'M') - times
    return (diffs.max().total_seconds() < 32 * num_months * 24 * 60 * 60)


def print_message(text):
    print('=====' + text + ' (' + str(datetime.now()) + ') =====')