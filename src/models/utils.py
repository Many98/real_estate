# from torch.utils.data import Dataset
from datasets import Dataset
import torch
import pandas as pd


def prepare_text_dataset(csv_path):
    df = pd.read_csv(csv_path, dtype={'description': 'str'})

    df.drop_duplicates(subset=['hash'], ignore_index=True, inplace=True)
    df.dropna(how='any', subset=['price', 'description'], inplace=True)

    return Dataset.from_pandas(df[['description', 'price']]).train_test_split(test_size=0.15)


# TODO implement pytorch lightning datamodule if needed
