# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed for evaluation methods to choose feature engineering methods in step-by-step.
"""
# Import necessary libraries.
import numpy as np
from numpy import linalg as la
from sklearn import linear_model as lm

import data_clean as dc


class NumericalConditionEvaluation(object):
    """
    The class to calculate numerical condition of dataframe for evaluation.
    Numerical condition is one kind of evaluation of efficiency to neural network.
    """
    def __init__(self, id_columns, target_column):
        """Initialization with given parameters.

        :param id_columns: List. The list of ID column names.
        :param target_column: String. The target column name.
        """
        # Assign parameters.
        self.target_column = target_column
        self.id_columns = id_columns
        # Create initial score of condition number.
        self.condition_num = None

    def preprocess(self, data):
        """Preprocess input dataframe.

        Preprocess input dataframe to suitable formate to be calculated condition number.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The processed dataframe.
        """
        # Fill None with 'mean' for numerical columns, and 'missing' for category columns.
        # And delete target column, as well.
        na_filler = dc.NaFiller()
        data = na_filler.fit_transform(data, data.columns.difference([self.target_column]))

        # Remove ID columns.
        data = data.columns.difference(self.id_columns)

        return data

    def calculate_condition_num(self, data):
        """Calculate condition number for feature engineering evaluation.

        Theory:
        -------
        According to theory of Numerical condition(ftp://ftp.sas.com/pub/neural/illcond/illcond.html),
        condition number is useful to evaluate the efficiency of data matrix. In this method, the simple
        neural network with one input, no hidden units, one linear output with bias and a least-squares
        error function, in another word, simple linear regression is considered to compute the condition
        number.

        Calculation:
        ------------
        If the inputs are arranged in an n (cases) by p (variables) matrix X, the Hessian is proportional
        to X'X, where the apostrophe indicates transposition. If an output bias is used, X should contain
        a column of ones representing the bias unit. The eigenvalues of X'X are the squares of the singular
        values of X. And condition number is the ratio of the largest and smallest eigenvalues of the
        Hessian matrix.

        :param data: Dataframe. The Pandas Dataframe to be calculated for condition number.

        :return: Number. The condition number for dataframe.
        """
        # Pre-process dataframe to deal with 'NA' problem as well as deleting target column.
        data = self.preprocess(data)

        # Calculate condition number according to Hessian matrix.
        hessian_matrix = np.matmul(data.T, data)
        eigvals = la.eigvals(hessian_matrix).sort()
        condition_num = rountd(eigvals[-1] / float(eigvals[0]), 3)

        return condition_num

    def compare_result(self, data):
        """Compare results between previous condition number and current one.

        Save previous condition number and compare with current condition number
        to prevent repetitive calculation.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Boolean. Whether it's better than previous condition number or not.
        """
        # Calculate current condition number.
        condition_num = self.calculate_condition_num(data)

        # Compare results and save current condition number.
        condition_num_previous = self.condition_num
        if not condition_num_previous:
            self.condition_num = condition_num
            return True
        else:
            if condition_num_previous > condition_num:
                self.condition_num = condition_num
                return True
            else:
                return False


class ModelBasedEvaluation(object):
    """
    The class is designed to build simple models to evaluate methods of feature engineering.
    """
    def __init__(self, id_columns, target_column, model_used='logistic'):
        """Initialization with given parameters.

        :param id_columns: List. The list of ID column names.
        :param target_column: String. The target column name.
        :param model_used: String. [Default: 'logistic']. The simple model used for evaluation.
        """
        # Assign parameters.
        self.target_column = target_column
        self.id_columns = id_columns
        self.model_used = model_used
        # Create saver for model accuracy.
        self.model_accuracy = None

    def preprocess(self, data):
        """Preprocess input dataframe.

        Preprocess input dataframe to suitable formate to be fed into model.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Dataframe. The processed dataframe.
        """
        # Fill None with 'mean' for numerical columns, and 'missing' for category columns.
        # And delete target column, as well.
        na_filler = dc.NaFiller()
        data = na_filler.fit_transform(data, data.columns.difference([self.target_column]))
        data = pd.concat([result, data[self.target_column]], axis=1)

        # Remove ID columns.
        data = data.columns.difference(self.id_columns)

        return data

    def calculate_model_accuracy(self, data):
        """Calculate model accuracy to evaluate feature engineering methods.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Number. The accuracy of model after feature engineering.
        """
        # Pre-process data before feeding data into model.
        data = self.preprocess(data)

        # Build machine learning model, train based on data and calculate the accuracy.
        mean_accuracy = None
        if self.model_used == 'logistic':
            lr = lm.LogisticRegressionCV(cv=2,random_state=1021)
            lr.fit(data.columns.difference([self.target_column]), data[self.target_column])
            mean_accuracy = lr.score(data.columns.difference([self.target_column]), data[self.target_column])

        return mean_accuracy

    def compare_result(self, data):
        """Compare results between previous model accuracy and current one.

        Save previous model accuracy and compare with current model accuracy
        to prevent repetitive calculation.

        :param data: Dataframe. The Pandas dataframe to be processed.

        :return: Boolean. Whether it's better than previous model accuracy or not.
        """
        # Calculate current model accuracy.
        model_accuracy = self.calculate_model_accuracy(data)

        # Compare results and save current model accuracy.
        model_accuracy_previous = self.model_accuracy
        if not model_accuracy_previous:
            self.model_accuracy = model_accuracy
            return True
        else:
            if model_accuracy_previous < model_accuracy:
                self.model_accuracy = model_accuracy
                return True
            else:
                return False
