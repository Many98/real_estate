import pandas as pd
import numpy as np


class Synchronizer(object):
    """
    Class handling unification/synchronization between sreality and bezrealitky data sources.
    It should also merge textual columns into one and
     ensure correct dtypes (string, floats, bool) within columns # TODO define exact dtypes for columns

    """
    def __init__(self, sreality_csv_path: str, breality_csv_path: str):
        try:
            self.sreality_df = pd.read_csv(sreality_csv_path)
            self.breality_df = pd.read_csv(breality_csv_path)  # dataframes to be synchronized
            self.final_df = None
        except Exception as e:
            print(e)

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        self.check_dtypes()  # checks dtypes on both dataframes
        self.unify()
        self.merge_text()

        return self.final_df

    def unify(self):
        """
        method to unify all levels within both dataframes and merge to one `final_df` dataframe
        Returns
        -------
        """

        self.final_df = pd.DataFrame()

    def merge_text(self):
        """
        method for merging all columns which should be processed by NLP
        * description (textual attribute => NLP)
        * note (from sreality data) => NLP
        * tags (from bezrealitky data) => NLP
        * place (from bezrealitky data) => NLP
        Returns
        -------

        """
        pass

    def check_dtypes(self):
        """
        method to check and ensure correct dtypes on relevant columns on both dataframes
        Returns
        -------

        """
        pass
