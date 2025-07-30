import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import urllib
from dotenv import load_dotenv

from appClass import EnvLoader, FileManager, DataCleaner

import random

class TestOperations(unittest.TestCase):

    # Tests if environment has variables. 
    # Does not test if there is a connection and valid credentials
    def test_EnvLoader_hasVariables(self):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = EnvLoader(script_dir)

        for key, val in env.get_db_credentials().items():
            self.assertTrue(val, f"Environment variable '{key}' is missing or blank")


    def test_FileManager_filesExist(self):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fileManager = FileManager(script_dir)

            books_df, customers_df = fileManager.read_csvs()  # call the method!

            for df, name in [(books_df, 'books_df'), (customers_df, 'customers_df')]:
                self.assertFalse(df.empty, f"{name} is empty or missing data")


if __name__ == '__main__':
    unittest.main()