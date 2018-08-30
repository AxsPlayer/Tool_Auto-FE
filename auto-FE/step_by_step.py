# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed to apply feature engineering methods step-by-step, automatically.
"""
# Import necessary libraries.
import sklearn

import data_clean as dc
import feature_engineering as fe
import feature_filtering as ff
import evaluation


class FeatureEngineering(object):
    """
    The class is created to perform feature engineering automatically, without
    any human labor. What you need to do is feeding into raw data with fixed format
    and some necessary information, the class and functions will handle hard feature
    engineering problems for you.
    """
    def __init__(self, id_columns, target_column):
        """Initialization with given parameters.

        :param id_columns: List. The list of ID column names.
        :param target_column: String. The target column name.
        """
        # Assign parameters.
        self.id_columns = id_columns
        self.target_column = target_column
        # Create dictionary or variable to store converters.

    def fit_transform(self, data):
        """Fit and transform train data, with automatic feature engineering methods.

        Automatically clean train data and perform feature engineering methods, return
        the clean data which is ready to be fed into model.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe which is suitable as model input.
        """
        #




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