# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed for evaluation methods to choose feature engineering methods in step-by-step.
"""
# Import necessary libraries.


class NumericalCondition(object):
    """The class to calculate numerical condition for dataframe.

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
