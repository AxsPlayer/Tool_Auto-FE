# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed to store some kind of feature filtering methods.
"""
# Import necessary libraries.
import pandas as pd
from sklearn import feature_selection as fe
from sklearn import decomposition as dc


class FeatureFilter(object):
    """The class for feature filtering.

    """
    def __init__(self, target_column):
        """Initialization with given parameters.

        :param target_column: String. The target column name.
        """
        # Assign parameters.
        self.target_column = target_column
        # Create dictionary or variable to store converters.
        self.variance_threshold = None
        self.univar_select = None

    def fit_transform(self, data):
        """Fit and transform using feature filtering.

        Fit and transform using several kind of feature filtering methods to
        select features in data.

        :param data: Dataframe. The Pandas dataframe, to be converted.

        :return: Dataframe. The converted dataframe after feature filtering.
        """
        # Removing features with low variance.
        threshold = 0.0
        var_thre = fe.VarianceThreshold(threshold=threshold)
        result = var_thre.fit_transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)
        # Store converter.
        self.variance_threshold = var_thre

        # Univariate feature selection, using univariate statistical tests.
        univar_select = fe.GenericUnivariateSelect(score_func=fe.mutual_info_classif,
                                                   mode='fwe', param=0.05)
        result = univar_select.fit_transform(data[data.columns.difference([self.target_column])],
                                             data[self.target_column])
        data = pd.concat([result, data[self.target_column]], axis=1)
        # Store converter.
        self.univar_select = univar_select

        return data

    def transform(self, data):
        """Transform using same feature filtering method.

        Transform target data using same feature filtering methods in fit_transform() function.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe.
        """
        # Removing features with low variance.
        result = var_thre.transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)

        # Univariate feature selection, using univariate statistical tests.
        result = univar_select.transform(data[data.columns.difference([self.target_column])],
                                         data[self.target_column])
        data = pd.concat([result, data[self.target_column]], axis=1)

        return data


class FeatureEmbedded(object):
    """The class to perform feature embedded.

    """
    def __init__(self, target_column):
        """Initialization with given parameters.

        :param target_column: String. The target column name.
        """
        # Assign parameters.
        self.target_column = target_column
        # Create dictionary or variable to store converters.
        self.feature_embedded = None

    def fit_transform(self, data):
        """Fit and transform using feature embedded.

        Fit and transform using several kind of feature embedded methods to
        select features in data.

        :param data: Dataframe. The Pandas dataframe, to be converted.

        :return: Dataframe. The converted dataframe after feature filtering.
        """
        # Feature selection using SelectFromModel, first version just use L1-based feature selection.
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data[data.columns.difference([self.target_column])],
                                                               data[self.target_column])
        model = SelectFromModel(lsvc, prefit=True)
        result = model.transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)
        # Store converter.
        self.feature_embedded = model

        return data

    def transform(self, data):
        """Transform using same feature embedded method.

        Transform target data using same feature embedded methods in fit_transform() function.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe.
        """
        # Feature selection using SelectFromModel, first version just use L1-based feature selection.
        model = self.feature_embedded
        result = model.transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)

        return data


class FeatureDecomposition(object):
    """The class for feature decomposition.

    """
    def __init__(self, target_column, ratio=0.3):
        """Initialization with given parameters.

        :param target_column: String. The target column name.
        :param ratio: Number. [Default: 0.3]. The ratio of the amount
                    of variance that needs to be explained
        """
        # Assign parameters.
        self.target_column = target_column
        self.ratio = ratio
        # Create dictionary or variable to store converters.
        self.pca = None

    def fit_transform(self, data):
        """Fit and transform using feature decomposition.

        Fit and transform using several kind of feature decomposition methods to
        select features in data.

        :param data: Dataframe. The Pandas dataframe, to be converted.

        :return: Dataframe. The converted dataframe after feature filtering.
        """
        # Apply PCA to original data and fetch result dataframe.
        pca = dc.PCA(n_components='mle', svd_solver='full', random_state=1021)
        result = pca.fit_transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)
        # Store converter.
        self.pca = pca

        return data

    def transform(self, data):
        """Transform using same feature decomposition method.

        Transform target data using same feature decomposition methods in fit_transform() function.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The converted dataframe.
        """
        # Apply PCA to original data and fetch result dataframe.
        pca = self.pca
        result = pca.transform(data[data.columns.difference([self.target_column])])
        data = pd.concat([result, data[self.target_column]], axis=1)

        return data
