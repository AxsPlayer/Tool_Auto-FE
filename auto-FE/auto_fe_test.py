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


def convert_data(combats, features, win_column=None):
    """Convert data into suitable format.

    Combine features and combats data, and convert into data in suitable
    format for training.

    :param combats: Dataframe. The Pandas dataframe which contains combats with
                two players with their id, as well as the winner id.
    :param features: Dataframe. The Pandas dataframe which contains every player's characters.
    :param win_column: String or None. [Default: None]. Whether combats dataframe contains winner
                    column or not. If yes, the column name of winner column.

    :return: Dataframe. The converted dataframe for training or testing.
    """
    # Create new dataframe to store values, and loop through.
    column_name = list(features.columns + '_1') + list(features.columns + '_2') + ['win']
    results = pd.DataFrame(columns=column_name)

    for
        results.loc[0]



def fetch_data(fold_path):
    """Fetch data saving in fold path.

    Convert data into suitable format, using csv files in fold path.

    :param fold_path: String. The fold in which data files are saved.

    :return: Dataframe. Combined dataframe to be tested on.
    """
    # Read all the data from target fold path.
    pokemon = pd.read_csv(fold_path+'/pokemon.csv')
    combats = pd.read_csv(fold_path+'/combats.csv')
    test_data = pd.read_csv(fold_path+'/tests.csv')

    # Convert data into suitable format for training and testing.
    training_data = convert_data(combats, pokemon, win_column='Winner')
    testing_data = convert_data(test_data, pokemon)


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