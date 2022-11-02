import pandas as pd
import numpy as np


class Synchronizer(object):
    """
    Class handling unification/synchronization between sreality and bezrealitky data sources

    TODO all these textual columns will be somewhere merged into one text / probably here in synchronize phase

    """
    def __init__(self, sreality_csv_path: str, breality_csv_path: str):
        try:
            self.sreality_df = pd.read_csv(sreality_csv_path)
            self.breality_df = pd.read_csv(breality_csv_path)  # dataframes to be synchronized
        except Exception as e:
            print(e)

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        pass
