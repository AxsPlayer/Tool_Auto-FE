# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package is used for data cleaning, containing several kind of data cleaning methods.
"""
# Import necessary libraries.
import numpy as np
from sklearn import preprocessing as pp


def fill_na(data, column_list):
    """Fill 'NA' in columns.

    Fill 'NA' in category columns with 'missing' value, and fill 'NA' in numeric columns
    with mean or median, as well as creating flag column recording whether the value is
    'NA' or not.

    :param data: Dataframe. The Pandas dataframe to be analyzed.
    :param column_list: List. The list contains column names which would be converted in function.

    :return: Dataframe. The converted Pandas dataframe.
    """
    # Loop through column list.
    for column in column_list:
        # Check whether column contains 'NA' or not.
        if not df.isnull().any():
            continue
        # If contain 'NA', check the type of column, category or numeric.
        if data.dtypes[column] not in ['float', 'int']:
            # Fill 'NA' with 'missing' value in category column.
            data[column] = data[column].fillna('missing')
        else:
            # Fill 'NA' with mean value in numeric column.
            imp = pp.Imputer(missing_values='NaN', strategy='mean', axis=0)
            data[column] = imp.fit_transform(data[column])

            # Create flag column, '1' for missing value, '0' for not.
            data[column + '_flag'] = np.zeros(shape=len(data[column]))
            data[column + '_flag'][data[column][data[column].isnull().values is True].index.tolist()] = 1

    return data


def wash_data(data, outlier=False):
    """Clean unusual data rows.

    Remove repetitive rows, rows with too many missing values, and report rows which
    should be considered as outliers.

    :param data: Dataframe. The Pandas dataframe to be washed.
    :param outlier: Boolean value. Whether to discard or not.

    :return: Dataframe. The converted Pandas dataframe.
    """
    # Remove repetitive data.
    data = data.drop_duplicates(keep='first')

    # Remove rows with too many missing values.
    percent = 0.6
    data = data.dropna(thresh=int(data.shape[1]*percent))

    # Detect outliers and report or remove.
    


