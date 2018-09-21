# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed to store some kind of feature engineering methods.
"""
# Import necessary libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import warnings
import logging

from scipy.stats import kstest
from scipy.stats import shapiro
import scipy.stats as spstats
from sklearn.preprocessing import *

import data_clean as dc
from config import output_log
from util import PowerTransformer

warnings.filterwarnings('ignore')
# Create logger for debugging.
output_log.init_log('./log/crawl_html')


class CategoryCombiner(object):
    """The class to combine categories with little instance.

    """
    def __init__(self, cate_columns, discard_ratio=0.01):
        """Initialize class with given parameters.

        :param cate_columns: List. The list of category columns, which would be processed.
        :param discard_ratio: The ratio set to filter out categories which
            should be combined.
        """
        # Assign values of parameters.
        self.cate_columns = cate_columns
        self.discard_ratio = discard_ratio
        # Create dictionary to store list of discard categories for each column.
        self.discard_cate_dic = {}

    def fit_transform(self, data):
        """Combine categories whose instance number is small, in train data.

        Combine categories whose instance number is small, and replace
        them with 'Others' value.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The result dataframe after processing.
        """
        # Combine categories whose ratio are under discard_ratio into one 'Others' category.
        for column in self.cate_columns:
            total_num = float(sum(data[column].value_counts()))
            discard_cate = []
            for key in data[column].value_counts().keys():
                if data[column].value_counts()[key] / total_num < self.discard_ratio:
                    discard_cate.append(key)
            data[column] = data[column].replace(discard_cate, 'Others')
            # Store discard categories for each column.
            self.discard_cate_dic[column] = discard_cate

        return data

    def transform(self, data):
        """Combine categories for each column in test data.

        Apply the method in fit() to transform test data in the same way.

        :param data: Dataframe. The Pandas dataframe to be transformed.

        :return: Dataframe. The dataframe after transforming.
        """
        # Combine categories whose ratio are under discard_ratio into one 'Others' category.
        for column in self.cate_columns:
            discard_cate = self.discard_cate_dic[column]
            data[column] = data[column].replace(discard_cate, 'Others')

        return data


class CategoryFeatureEngineer(object):
    """The class to apply feature engineering automatically.

    """
    def __init__(self, cate_columns):
        """Initialize class with given parameters.

        :param cate_columns: List. The list consists of category column names.
        """
        # Assign parameters.
        self.cate_columns = cate_columns
        # Create convert dictionary.
        self.cate_encoding_dic = {}
        self.cate_combiner = None
        self.cate_label_dic = {}
        self.cate_na_filler = None

    def fit_transform(self, dataset):
        """Feature engineering for category columns.

        Conduct feature engineering to category columns.
        Including several kind of methods, as followings:
            1. Fill NA wil 'missing'.
            2. Combine small categories into one same category.
            3. One-hot encoding for category columns.

        :param dataset: Dataframe. The input dataframe.

        :return: Dataframe. The output dataframe with converted category columns.
        """
        # Fill None with 'missing' for category columns.
        data = dataset.copy()
        na_filler = dc.NaFiller()
        data = na_filler.fit_transform(data, self.cate_columns)
        self.cate_na_filler = na_filler

        # Combine categories whose ratio are under 0.01 into one 'Others' category.
        cate_combiner = CategoryCombiner(self.cate_columns)
        data = cate_combiner.fit_transform(data)
        self.cate_combiner = cate_combiner
        # Label encoder to convert values in category column into numeric values.
        result = pd.DataFrame()
        for column in self.cate_columns:
            gen_le = LabelEncoder()
            result[column] = gen_le.fit_transform(data[column])
            # Store label encoder into dictionary.
            self.cate_label_dic[column] = gen_le

            # Encode category columns with One-hot Encoding method.
            gen_ohe = OneHotEncoder()
            gen_ohe.fit(result[[column]])
            gen_feature_arr = gen_ohe.transform(result[[column]]).toarray()
            gen_feature_labels = [column + '_' + str(cls_label)
                                  for cls_label in gen_le.classes_]
            gen_features = pd.DataFrame(gen_feature_arr,
                                        columns=gen_feature_labels)
            result = pd.concat([result, gen_features], axis=1)
            # Store encoders into dictionary.
            self.cate_encoding_dic[column] = gen_ohe

        data = data.reset_index()
        # Add other columns into result.
        for column in data.columns:
            if column not in self.cate_columns and column != 'index':
                result = pd.concat([result, data[column]], axis=1)

        return result

    def transform(self, data):
        """The feature engineering for category columns.

        The feature engineering method applied on test data, using the same in
        fit_transform() function.

        :param data: Dataframe. The input Pandas dataframe, to be processed.

        :return: Dataframe. The processed dataframe.
        """
        # Fill None with 'missing' for category columns.
        data = self.cate_na_filler.transform(data, self.cate_columns)

        # Combine categories whose ratio are under 0.01 into one 'Others' category.
        data = cate_combiner.transform(data)

        # Label encoder to convert values in category column into numeric values.
        for column in self.cate_columns:
            # Apply label encoder to transform category columns.
            data[column] = gen_le.transform(data[column])

            # Encode category columns with One-hot Encoding method.
            gen_feature_arr = gen_ohe.transform(data[[column]]).toarray()
            gen_feature_labels = [column + '_' + str(cls_label)
                                  for cls_label in gen_le.classes_]
            gen_features = pd.DataFrame(gen_feature_arr,
                                        columns=gen_feature_labels)
            data = pd.concat([data, gen_features], axis=1)

        return data


def column_type_detection(data, id_col, target_col):
    """Detect column type and collect according to column type.

    Given id column name and target column name, divide columns into
    ID_col, target_col, numeric_col and category_col.

    :param data: Dataframe. The input dataframe in pandas form.
    :param id_col: List. The list of ID column names.
    :param target_col: List. The list of Target column names.

    :return: List. Numeric_col, Category_col.
    """
    # Loop through features and divide into two parts according to data types.
    cate_columns = []
    num_columns = []
    for key in data.dtypes.keys():
        # Skip id and target columns.
        if key in id_col or key in target_col:
            continue
        # If data type is not in ['float', 'int'], column is collected by cate_col.
        if data.dtypes[key] not in ['float', 'int']:
            cate_columns.append(key)
            # Convert all the cat_col into 'object' data type.
            data[key] = data[key].astype('object')
        else:
            num_columns.append(key)

    return num_columns, cate_columns


class NumericFeatureEngineer(object):
    """The class for numeric feature engineering.

    """
    def __init__(self, num_columns):
        """Initialize class with given parameters.

        :param num_columns: List. The list consists of numeric column names.
        """
        # Assign parameters.
        self.num_columns = num_columns
        # Create dictionaries for converters.
        self.num_transform_dic = {}
        self.num_na_filler = None

    @staticmethod
    def check_normal_distribution(data):
        """Function to test whether the data is normal distribution or not.

        Use some statistic methods to test normal distribution, numerically.

        :param data: Dataframe. One column dataframe, waited to be tested.

        :return:
            stat: Some statistic result from test.
            p: P-value to reject Null hypothesis, which means it's not normal distribution.
        """
        # normality test
        stat, p = shapiro(data)

        return stat, p

    def fit_transform(self, dataset):
        """Feature engineering for numeric columns.

        Conduct feature engineering to numeric columns.
        Including several kind of methods, as followings:
            1. Fill NA wil mean.
            3. Detect and convert to normal distribution.
            4. Standardization.
            5. Round to float3.

        :param dataset: Dataframe. The input dataframe.

        :return: Dataframe. The output dataframe with converted numeric columns.
        """
        # Convert numerical columns whose distributions are not normal to normal distribution.
        # Check whether the distribution is normal or not.
        data = dataset.copy()
        for column in self.num_columns:
            print column
            # Normality test.
            stat, p = self.check_normal_distribution(data[column])
            print(column, ': Statistics=%.3f, p=%.3f' % (stat, p)),

            # When p-value is under 0.05, it means the distribution is different to normal distribution.
            alpha = 0.05  # Set cutoff to reject the Null hypothesis.
            if p < alpha:
                print('Sample does not look Gaussian (reject H0)'),
                # Calculate skewness of distribution.
                skewness = data[column].skew(axis=0)

                # Check whether there are outliers or not.
                outlier_detector = dc.OutlierDetector(data, [], [])
                outlier_index = outlier_detector.mean_detection(data[column])
                # todo check outlier function.
                if True:
                    # Check whether there are negative values.
                    print '\nThere is no outlier.'
                    if sum(data[column] < 0) == 0:
                        # If there is none of negative values, apply Box-cox transformation.
                        power_transformer = PowerTransformer(method='box-cox')
                        data[column] = power_transformer.fit_transform(data[column].reshape(-1, 1))
                    else:
                        # If there are some negative values, apply yeo-johnson method.
                        power_transformer = PowerTransformer(method='yeo-johnson')
                        data[column] = power_transformer.fit_transform(data[column].reshape(-1, 1))
                    # Store power transformer into dictionary.
                    self.num_transform_dic[column] = power_transformer
                else:
                    # If there are some outliers, apply quantile transformer to normal distribution.
                    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=1021)
                    data[column] = quantile_transformer.fit_transform(data[column].reshape(-1, 1))
                    # Store quantile transformer into dictionary.
                    self.num_transform_dic[column] = quantile_transformer
            else:
                print('Sample looks Gaussian (fail to reject H0)')
                # If the column is normal distribution, assign 'None' to transformer dictionary.
                self.num_transform_dic[column] = None

        # Round the number into .3float, to lower running time.
        data = data.round(3)

        # Fill None with 'mean' for numerical columns.
        na_filler = dc.NaFiller()
        data = na_filler.fit_transform(data, self.num_columns)
        # Store imputer into dictionary.
        self.num_na_filler = na_filler

        return data

    def transform(self, data):
        """Transform numeric column, especially for test data.

        Apply same method in fit_transform() function to transform target dataframe.

        :param data: Dataframe. The target Pandas dataframe to be transformed.

        :return: Dataframe. The processed dataframe.
        """
        # Convert numerical columns whose distributions are not normal to normal distribution.
        for column in self.num_transform_dic.keys():
            transformer = self.num_transform_dic[column]
            data[column] = transformer.transform(data[column].reshape(-1, 1))

        # Round the number into .3float, to lower running time.
        data = data.round(3)

        # Fill None with 'mean' for numerical columns.
        na_filler = self.num_na_filler
        data = na_filler.transform(data, self.num_columns)

        return data


class TargetConverter(object):
    """The class to convert target class from string into numeric.

    """
    def __init__(self, target_column):
        """Initialize class with given parameters.

        :param target_column: String. The name of target column.
        """
        # Assign parameters.
        self.target_column = target_column
        # Create variable to store transformer.
        self.label_encoder = None

    def fit_transform(self, data):
        """Fit and transform target variable.

        Fit and convert target variable into numbers if it's string type.

        :param data: Dataframe. The Pandas dataframe, which to be processed.

        :return: Dataframe. The transformed dataframe.
        """
        # Convert target variable into numerical if currently not.
        if data[self.target_column].dtype not in ['int', 'float']:
            gen_le = LabelEncoder()
            data[self.target_column] = gen_le.fit_transform(data[self.target_column])
            # Store label encoder for target column.
            self.label_encoder = gen_le

        return data

    def transform(self, data):
        """Transform target variable to numeric.

        Apply the same method in fit_transform() function to convert target
        column into numeric if not numeric.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The transformed dataframe.
        """
        # Convert target variable into numerical using the same method ,if currently not.
        if not self.label_encoder:
            label_encoder = self.label_encoder
            data[self.target_column] = label_encoder.transform(data[self.target_column])

        return data


def convert_sparse(data):
    """Convert columns with sparse data into sparse matrix.

    Use sparse related function in Scipy to convert sparse data columns into
    sparse matrix.

    :param data: Dataframe. The Pandas dataframe to be processed.

    :return: Dataframe. The converted dataframe.
    """
    # Check whether it's sparse data column or not.
    if scipy.sparse.issparse(data):
        data = scipy.sparse.csr_matrix(data)

    return data
