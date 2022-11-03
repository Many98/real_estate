import pandas as pd
import numpy as np
import os


class Synchronizer(object):
    """
    Class handling unification/synchronization between sreality and bezrealitky data sources.
    It should also merge textual columns into one and
     ensure correct dtypes (string, floats, bool) within columns # TODO define exact dtypes for columns

    """
    def __init__(self, from_row: tuple[int, int]):
        """

        Parameters
        ----------
        from_row : tuple[int, int]
            Indices of row from which start. from_row[0] -> sreality
                                             from_row[1] -> breality TODO make this more robust
        """
        self.from_row = from_row
        self.final_df = None

    def __call__(self, sreality_csv_path: str, breality_csv_path: str, *args, **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
        sreality_csv_path : str
            Path of sreality csv
        breality_csv_path : str
            Path of breality csv

        Returns
        -------

        """
        try:
            self.sreality_df = pd.read_csv(sreality_csv_path).iloc[self.from_row[0]:, :]
            self.breality_df = pd.read_csv(breality_csv_path).iloc[self.from_row[1]:, :]  # dataframes to be synchronized
        except Exception as e:
            print(e)

        self.check_dtypes()  # checks dtypes on both dataframes
        self.unify()
        self.merge_text()

        self.final_df.to_csv('../data/tmp_synchronized.csv', mode='w', index=False)

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
