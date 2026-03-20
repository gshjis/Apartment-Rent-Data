"""
Configuration settings for the apartment rent data project.
"""

import os


class Config:
    """
    Configuration settings for the apartment rent data project.

    Attributes:
        PROJECT_ROOT (str): The absolute path to the project root directory.
        DATA_DIR (str): The absolute path to the data directory.
    """

    PROJECT_ROOT = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")

    @classmethod
    def get_data_path(cls, filename):
        """
        Get the full path to a data file.

        Args:
            filename (str): The name of the data file.

        Returns:
            str: The full path to the data file.
        """
        return os.path.join(cls.DATA_DIR, filename)
