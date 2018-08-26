# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This package is used for data cleaning, containing several kind of data cleaning methods.
"""
# Import necessary libraries.
import numpy as np
from sklearn import preprocessing as pp
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector(object):
    """
    The class of several methods to detect outliers.
    """
    def __init__(self, data):
        """Initialize outlier detector with dataframe.

        :param data: Dataframe. Input Pandas dataframe, without target column.
        """
        self.data = data

    def

class OverSampler(object):
    """
    The class of several methods for oversampling.
    """
    def __init__(self, data, target_column):
        """Initialize sampler with dataframe, as well as target column name.

        :param data: Dataframe. Input Pandas dataframe, with target column.
        :param target_column: String. The target column name.
        """
        self.data = data
        self.target_column = target_column




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


def wash_data(data, target_column, outlier=False):
    """Clean unusual data rows.

    Remove repetitive rows, rows with too many missing values, and report rows which
    should be considered as outliers.

    :param data: Dataframe. The Pandas dataframe to be washed.
    :param target_column: String. The target column name.
    :param outlier: Boolean value. [default: False]. Whether to discard or not.

    :return: Dataframe. The converted Pandas dataframe.
    """
    # Remove repetitive data.
    data = data.drop_duplicates(keep='first')

    # Remove rows with too many missing values.
    percent = 0.6
    data = data.dropna(thresh=int(data.shape[1]*percent))

    # Detect outliers and report or remove.


def sample_data(data, target_column, method='both'):
    """Sample the data to be balanced.

    Down-sampling majority class and over-sampling minority class to have the ratio of 1:1.
    This version only supports for binary classification.

    :param data: Dataframe. The Pandas dataframe to be sampled.
    :param target_column: String. The target column name.
    :param method: String. [default: 'both']. Choose from list ['both', 'down-sampling', 'over-sampling']. When
                set to 'both', both methods will be applied to dataframe, or when set to ether
                of other two methods, the corresponding method will be applied to dataframe alone.

    :return: Dataframe. The converted Pandas dataframe.
    """
    # Check the target column to decide which are majority label and minority label.
    major_label = data[target_column].value_counts().keys()[0]
    minor_label = data[target_column].value_counts().keys()[1]

    # Down-sampling the majority class data.
    



