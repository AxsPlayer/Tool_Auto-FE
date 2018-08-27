# !/usr/bin/python
# -*- coding: utf-8 -*-
# Import necessary libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from scipy.stats import kstest
from scipy.stats import shapiro
import scipy.stats as spstats
from sklearn.preprocessing import *

warnings.filterwarnings('ignore')


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


def num_fe(data, num_columns):
    """Feature engineering for numeric columns.

    Conduct feature engineering to numeric columns.
    Including several kind of methods, as followings:
        1. Fill NA wil mean.
        2. Translate value to none-negative.
        3. Detect and convert to normal distribution.
        4. Standardization.
        5. Round to float3.

    :param data: Dataframe. The input dataframe.
    :param num_columns: List. The list containing numeric column names in dataframe.

    :return: Dataframe. The output dataframe with converted numeric columns.
    """
    # Create parameter dictionary to store feature converter for test data.
    para_dic = {}
    # Fill None with 'mean' for numerical columns.
    imputer = Imputer()
    imputer.fit(data[num_columns])
    data[num_columns] = imputer.transform(data[num_columns])
    para_dic['imputer'] = imputer  # Add imputer into para_dic.

    # Convert values in numerical columns to positive values, if there are some negative values.
    para_dic['translate'] = {}
    for column in num_columns:
        if sum(data[column] <= 0) != 0:
            if min(data[column]) == 0:
                data[column] = data[column] + max(data[column]) * 0.01
            else:
                data[column] = data[column] - min(data[column]) * 1.01
        para_dic['translate'][column] = [max(data[column]), min(data[column])]

    # Convert numerical columns whose distributions are not normal to normal distribution.
    # Check whether the distribution is normal or not.
    para_dic['normal_distribution'] = {}
    for column in num_columns:
        # normality test
        stat, p = shapiro(data[column])
        print(column, ': Statistics=%.3f, p=%.3f' % (stat, p)),
        # interpret
        alpha = 0.05
        # When p-value is under 0.05, it means the distribution is different to normal distribution.
        if p < alpha:
            print('Sample does not look Gaussian (reject H0)'),
            # Calculate skewness of distribution.
            skewness = data[column].skew(axis=0)
            # get optimal lambda value from non null income values
            middle = np.array(data[column])
            middle_clean = middle[~np.isnan(middle)]
            l, opt_lambda = spstats.boxcox(middle_clean)
            print('Optimal lambda value:', opt_lambda)
            data[column] = spstats.boxcox(data[column], lmbda=opt_lambda)
            para_dic['normal_distribution'][column] = opt_lambda
        else:
            print('Sample looks Gaussian (fail to reject H0)')
            para_dic['normal_distribution'][column] = None

    # Standardization.
    scaler = StandardScaler()
    scaler.fit(data[num_columns])
    data[num_columns] = scaler.transform(data[num_columns])
    para_dic['scaler'] = scaler

    # Round the number into .3float, to lower running time.
    data = data.round(3)

    return data


def cate_fe(data, cate_columns):
    """Feature engineering for category columns.

    Conduct feature engineering to category columns.
    Including several kind of methods, as followings:
        1. Fill NA wil 'missing'.
        2. Combine small categories into one same category.
        3. One-hot encoding for category columns.

    :param data: Dataframe. The input dataframe.
    :param cate_columns: List. The list containing category column names in dataframe.

    :return: Dataframe. The output dataframe with converted category columns.
    """
    # Fill None with 'missing' for category columns.
    data[cate_columns] = data[cate_columns].fillna('missing')

    # Combine categories whose ratio are under 0.01 into one 'Others' category.
    cate_combiner = CategoryCombiner(cate_columns)
    data = cate_combiner.fit_transform(data)

    # Encode category columns with One-hot Encoding method.
    # data = pd.get_dummies(data, columns=cate_columns)
    para_dic = {}
    result = pd.concat([data[id_columns], data[target_column], data[num_columns]], axis=1)
    for column in cate_columns:
        gen_le = LabelEncoder()
        gen_le.fit(data[column])
        data[column] = gen_le.transform(data[column])

        gen_ohe = OneHotEncoder()
        gen_ohe.fit(data[[column]])
        gen_feature_arr = gen_ohe.transform(data[[column]]).toarray()
        gen_feature_labels = [column + '_' + str(cls_label)
                              for cls_label in gen_le.classes_]
        gen_features = pd.DataFrame(gen_feature_arr,
                                    columns=gen_feature_labels)
        result = pd.concat([result, gen_features], axis=1)

        para_dic[column] = [gen_le, gen_ohe]

    return result


def train_converter(data, id_col, target_col):
    """Detect column types, as well as feature engineering.

    Firstly, detect column types. And then conduct feature engineering on
    numeric columns, and category columns.

    :param data: Dataframe. The input dataframe in pandas form.
    :param id_col: List. The list of ID column names.
    :param target_col: List. The list of Target column names.

    :return: Dataframe. The output dataframe after feature engineering.
    """
    # Detect column types and divide them into corresponding lists.
    num_columns, cate_columns = column_type_detection(data, id_col, target_col)

    # Feature engineering with numeric columns.
    data = num_fe(data, num_columns)

    # Feature engineering with category columns.
    data = cate_fe(data, cate_columns)

    # Convert target variable into numerical if currently not.
    if result[target_col].dtype not in ['int', 'float']:
        gen_le = LabelEncoder()
        gen_le.fit(data[target_col])
        data[target_col] = gen_le.transform(data[target_col])

    return data,


