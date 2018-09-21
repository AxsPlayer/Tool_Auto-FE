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
        self.cate_fe = None
        self.num_fe = None
        self.target_converter = None
        self.feature_filter = None
        self.feature_embedded = None
        self.feature_decomposition = None

    def fit_transform(self, dataset):
        """Fit and transform train data, with automatic feature engineering methods.

        Automatically clean train data and perform feature engineering methods, return
        the clean data which is ready to be fed into model.

        :param dataset: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe which is suitable as model input.
        """
        # Data clean process.
        # -------------------
        # Detect column types.
        data = dataset.copy()
        num_columns, cate_columns = fe.column_type_detection(data, self.id_columns, self.target_column)

        # Step1: Wash data with repetitive and abnormal data.
        data = dc.wash_data(data, self.target_column, self.id_columns)

        # Step2: Under-sample the data to balanced ratio.
        data = dc.sample_data(data, self.target_column, self.id_columns, method='under-sampling')

        # Step3: Category column feature engineering.
        cate_fe = fe.CategoryFeatureEngineer(cate_columns)
        data = cate_fe.fit_transform(data)
        # Save converter.
        self.cate_fe = cate_fe

        # Step4: Outlier detection.
        outlier_detector = dc.OutlierDetector(data, self.target_column, self.id_columns)
        outlier_index = outlier_detector.isolation_forest()
        print('The list of outlier index is: ' + str(outlier_index))

        # Feature engineering.
        # --------------------
        # Step1: Numeric column feature engineering.
        num_fe = fe.NumericFeatureEngineer(num_columns)
        data = num_fe.fit_transform(data)
        # Save converter.
        self.num_fe = num_fe

        # Step2: Over-sample the data to balanced ratio.
        data = dc.sample_data(data, self.target_column, self.id_columns, method='over-sampling')

        # Step3: Convert target variable to numerical variable.
        target_converter = fe.TargetConverter(self.target_column)
        data = target_converter.fit_transform(data)
        # Save converter.
        self.target_converter = target_converter

        # Feature selection.
        # ------------------
        # Step1: Feature filter.
        feature_filter = ff.FeatureFilter(self.target_column)
        data = feature_filter.fit_transform(data)
        # Save converter.
        self.feature_filter = feature_filter

        # Step2: Feature embedded.
        feature_embedded = ff.FeatureEmbedded(self.target_column)
        data = feature_embedded.fit_transform(data)
        # Save converter.
        self.feature_embedded = feature_embedded

        # Step3: Feature decomposition.
        feature_decomposition = ff.FeatureDecomposition(self.target_column)
        data = feature_decomposition.fit_transform(data)
        # Save converter.
        self.feature_decomposition = feature_decomposition

        return data

    def transform(self, dataset):
        """Transform test data, with automatic feature engineering methods.

        Automatically clean train data and perform feature engineering methods, return
        the clean data which is ready to be fed into model. The converters are from the
        fit_transform() function, which means to use the same feature engineering methods.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe which is suitable as model input.
        """
        # Data clean process.
        # -------------------
        # Detect column types.
        data = dataset.copy()
        num_columns, cate_columns = fe.column_type_detection(data, self.id_columns, self.target_column)

        # Step1: Category column feature engineering.
        cate_fe = self.cate_fe
        data = cate_fe.transform(data)

        # Feature engineering.
        # --------------------
        # Step1: Numeric column feature engineering.
        num_fe = self.num_fe
        data = num_fe.transform(data)

        # Step2: Convert target variable to numerical variable.
        target_converter = self.target_converter
        data = target_converter.transform(data)

        # Feature selection.
        # ------------------
        # Step1: Feature filter.
        feature_filter = self.feature_filter
        data = feature_filter.transform(data)

        # Step2: Feature embedded.
        feature_embedded = self.feature_embedded
        data = feature_embedded.transform(data)

        # Step3: Feature decomposition.
        feature_decomposition = self.feature_decomposition
        data = feature_decomposition.transform(data)

        return data
