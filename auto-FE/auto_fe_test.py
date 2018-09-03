# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
This script is designed for unit test for all the functions in automatic feature engineering.
"""
# Import necessary packages.
import logging
import os
import unittest
import pandas as pd

import data_clean
import feature_engineering
import feature_selection
import step_by_step
import evaluation


def fetch_data(fold_path):
    """Fetch data saving in fold path.

    Convert data into suitable format, using csv files in fold path.

    :param fold_path: String. The fold in which data files are saved.

    :return: Dataframe. Combined dataframe to be tested on.
    """



class TestFeatureEngineering(unittest.TestCase):
    """Test functions in all the scripts corresponding to feature engineering.

    Test each feature engineering function and the total automatic feature engineering
    script.
    """
    def __init__(self, *args, **kwargs):
        """Init the test class with all the attributes."""
        super(TestFeatureEngineering, self).__init__(*args, **kwargs)
        self.file_path = 'data/'
        self.

    def test_data_clean(self):

    def test_feature_engineering(self):

    def test_feature_selection(self):

    def test_step_by_step(self):

    def test_evaluation(self):


if __name__ == "__main__":
    # Use 'python -m unittest auto_fe_test' in console for unit test.
    unittest.main()