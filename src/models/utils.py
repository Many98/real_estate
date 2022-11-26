# from torch.utils.data import Dataset
from datasets import Dataset, DatasetDict
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_text_dataset(csv_path, split_on=None, use_price_m2=False, use_gp_residual=False) -> DatasetDict:
    df = pd.read_csv(csv_path, dtype={'description': 'str'})

    df['price_m2'] = df['price'] / df['usable_area']

    df = df[(df['price_m2'] > 40000) & (df['price_m2'] < 300000)]
    df = df[(df['floor'] > -2) & (df['floor'] <= 30)]

    df.drop_duplicates(subset=['hash'], ignore_index=True, inplace=True)
    df.dropna(how='any', subset=['price', 'description'], inplace=True)

    if split_on is not None and split_on:
        df['description'] = df['description'].apply(lambda x: x.split(split_on)[0])

    if use_gp_residual and use_price_m2:
        df['label'] = df['price_m2'] - df['gp_mean_price']
    elif use_gp_residual and not use_price_m2:
        df['gp'] = df['gp_mean_price'] * df['usable_area']
        df['label'] = df['price'] - df['gp']
    elif not use_gp_residual and use_price_m2:
        df['label'] = df['price_m2']
    else:
        df['label'] = df['price']

    return Dataset.from_pandas(df[['description', 'label']]).train_test_split(test_size=0.15)


def compute_metrics(eval_pred) -> dict:
    predictions, labels = eval_pred
    return {'rmse': mean_squared_error(labels, predictions, squared=False),
            'mae': mean_absolute_error(labels, predictions),
            'r2': r2_score(labels, predictions)}


# TODO implement pytorch lightning datamodule if needed
