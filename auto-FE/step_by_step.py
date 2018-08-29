# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed to apply feature engineering methods step-by-step, automatically.
"""
# Import necessary libraries.
import sklearn

import data_clean as dc
import feature_engineering as fe


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