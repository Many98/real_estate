import pandas as pd
import numpy as np
import re


def prepare_atlas_cen_data(csv_path: str, dtypes: dict = {'when': str, 'href': str, 'area': str,
                                                         'price': str, 'long': float, 'lat': float}) -> pd.DataFrame:
    """
    function to prepare atlas cen data from scraped csv. mainly extract numbers from strings
    Parameters
    ----------
    csv_path :
    dtypes :

    Returns
    -------

    """
    try:
        df = pd.read_csv(csv_path, dtype=dtypes)
    except Exception as e:
        print(e)

    df['when'] = (df['when']
                  .apply(lambda x: x.replace(' ', ''))
                  .apply(lambda x: re.search("[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}", x).group())
                  # TODO if needed then will convert to datetime
                 )
    df['area'] = (df['area']
                  .apply(lambda x: x.replace(' ', '').replace('m2', ''))
                  .apply(lambda x: x.split(',')[1])
                  .astype(float)
                  )
    df['price'] = (df['price']
                  .apply(lambda x: x.replace(' ', '').replace('m2', '').replace('Kč', '').replace('/', ''))
                  .apply(lambda x: x.split()[-1])  # default splits on white spaces
                  .astype(float)
                  )
    df.rename(columns={"price": "price/m2"}, inplace=True)

    return df


def data_filtering(csv_path: str) -> pd.DataFrame:
    """
    function to filter reality data from outliers caused by wrong data inputs
    Parameters
    ----------
    csv_path

    Returns
    -------

    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(e)

    # Not sure if this is done before
    if 'price_m2' not in df.columns:
        df['price_m2'] = df["price"] / df["usable_area"]

    # Maybe better to put those "empirical" values into the input of the function
    threshold_low = 47000
    threshold_high = 400000

    df = df[df['price_m2'] > threshold_low]
    df = df[df['price_m2'] < threshold_high]

    return df
