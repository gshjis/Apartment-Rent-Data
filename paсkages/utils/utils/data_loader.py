import pandas as pd
from config import Config


def load_train_data():
    """
    Load the training data for the apartment rent prediction model.

    Returns:
        pandas.DataFrame: The training data.
    """
    train_file = "apfr_classified_100K.csv"
    train_path = Config.get_data_path(train_file)

    # Load the training data from the file
    train_data = pd.read_csv(train_path, encoding="windows-1252", sep=";")

    return train_data


def load_test_data():
    """
    Load the test data for the apartment rent prediction model.

    Returns:
        tuple: A tuple containing the test features and target variables.
    """
    test_file = "apfr_classified_10K.csv"
    test_path = Config.get_data_path(test_file)

    # Load the test data from the file
    test_data = pd.read_csv(test_path, encoding="windows-1252", sep=";")

    return test_data
