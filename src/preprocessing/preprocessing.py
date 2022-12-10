import pandas as pd
import numpy as np


class Preprocessor(object):
    """
    Class handling missing values, categorization, categorical encodings & standardization.
    Input dataframe is required to have "synchronized" values (two data sources needs to be unified/synchronized)
    https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f

    TODO consider using decorators as it would be probably more elegant solution
    """

    def __init__(self, df: pd.DataFrame, inference: bool):

        self.df = df  # dataframe to be preprocessed
        self.inference = inference
        self.status = 'OK'

    def __call__(self, *args, **kwargs) -> dict:
        if not self.df.empty:

            try:
                self.expand()

                self.impute()  # impute in pandas-like way too lazy to overwrite
                if self.status not in ['OK', 'OOPP']:
                    return {'data': self.df, 'quality_data': None, 'distance_data': None, 'criminality_data': None,
                            'status': self.status}

                self.categorize()
                self.remove()
                self.adjust()
            except Exception as e:
                if not self.inference:
                    print(e)
                else:
                    return {'data': None,
                            'quality_data': None,
                            'distance_data': None,
                            'criminality_data': None,
                            'status': 'INTERNAL ERROR (PREPROCESS)'}

        if self.df.empty:
            self.status = 'EMPTY'

        return {'data': self.df[self.df.columns.difference(['air_quality', 'built_density', 'sun_glare',
                                                            'daily_noise'])],
                'quality_data': self.df[['air_quality', 'built_density', 'sun_glare',
                                         'daily_noise']],
                'distance_data': self.df[[i for i in self.df.columns if 'dist' in i and '_te' not in i and 'city' not in i]],
                'criminality_data': self.df[[i for i in self.df.columns if '_crime' in i]],
                'status': self.status}

    def expand(self):
        """
        method to expand some critical features to copies which will be numerical/ordinal because by default
         almost all features will be categorical/categorized and one-hot encoded
        Returns
        -------

        """
        cols = [i for i in self.df.columns if 'has' in i and 'hash' not in i] + \
               [i for i in self.df.columns if 'dist' in i and 'city' not in i]
        for d in cols:
            self.df[d + '_te'] = self.df[d]

        # adding price_m2 column

        self.df['price_m2'] = self.df['price'] / self.df['usable_area']

        self.df['gp'] = self.df['gp_mean_price'] * self.df['usable_area']

    def impute(self):
        """
        # interesting article about missing values
          https://towardsdatascience.com/all-about-missing-data-handling-b94b8b5d2184
        method to handle missing values imputation
        Parameters
        ----------

        Returns
        -------

        """
        if not self.inference:
            self.df.dropna(how='any', subset=['price', 'usable_area', 'long', 'lat', 'disposition', 'city_district'],
                           inplace=True)
        else:
            if self.df[['usable_area', 'long', 'lat']].isna().any().any():
                self.status = 'RANP'  # RequiredAttributesNotPresent
                return
            if self.df[['city_district']].isna().any().any():
                self.status = 'OOPP'  # OutOfPraguePrediction

        # fill unknown/undefined
        self.df[['air_quality', 'built_density', 'sun_glare']] = \
            self.df[['air_quality', 'built_density', 'sun_glare']].astype(str)
        self.df = self.df.replace("nan", np.nan)

        self.df.fillna(value={'floor': -99,  # -> arbitrary imputation  (floor handled only as numeric)
                              'year_reconstruction': -99,  # 2038,
                              'energy_effeciency': 'unknown', 'ownership': 'unknown', 'description': '',
                              'gas': 'unknown', 'waste': 'unknown', 'equipment': 'unknown', 'state': 'unknown',
                              'construction_type': 'unknown', 'electricity': 'unknown',
                              'heating': 'unknown', 'transport': 'unknown',
                              'telecomunication': 'unknown',
                              # 'age': 'undefined',
                              'air_quality': 'unknown', 'built_density': 'unknown', 'sun_glare': 'unknown'
                              },
                       inplace=True)
        # Note that `city_district`, `disposition` is not imputed because Target encoder will
        # replace nan with mean price from training distribution

        # fill <>_dist_te features
        dist_cols = [i for i in self.df.columns if 'dist_te' in i]
        self.df.fillna(value={i: 10000 for i in dist_cols}, inplace=True)

        # fill has_<> & no_barriers attributes
        has_cols = [i for i in self.df.columns if 'has' in i and 'hash' not in i]
        has_cols.append('no_barriers')  # TODO probably `no_barriers will be removed as we do not have it`
        self.df[has_cols] = self.df[has_cols].astype(bool)
        self.df.fillna(value={i: False for i in has_cols}, inplace=True)

    def categorize(self):
        """
        method to categorize some numeric features
        Returns
        -------

        """
        # categorize <>_dist_te features
        dist_cols = [i for i in self.df.columns if 'dist_te' in i]
        dists = list(range(0, 1600, 100)) + [np.infty]
        for col in dist_cols:
            self.df[col] = pd.cut(self.df[col], bins=dists,
                                  include_lowest=True,
                                  labels=[f'{dists[i]}-{dists[i + 1] - 1}m' for i in range(len(dists[:-2]))] + [
                                      '>=1500m'])

    def remove(self):
        """
        method to remove unnecessary columns/features
        Returns
        -------

        """
        if not self.inference:
            self.df.drop_duplicates(subset=['hash'], ignore_index=True, inplace=True)

            # removing data based on "empirical" values
            threshold_low = 40000
            threshold_high = 380000

            self.df = self.df[(self.df['price_m2'] > threshold_low) & (self.df['price_m2'] < threshold_high)]
            self.df = self.df[(self.df['floor'] > -2) & (self.df['floor'] < 30)]

        ok_cols = self.df.columns.difference(["note", "description", "hash", "name", "desc_hash",
                                              "floor_area", "geometry", "place", "tags",
                                              "additional_disposition", "transport", "header",
                                              'year_reconstruction', 'no_barriers',
                                              # 'air_quality', 'built_density', 'sun_glare',
                                              # 'daily_noise',
                                              'nightly_noise',
                                              'gas', 'waste', 'telecomunication', 'electricity', 'heating',
                                              "date"
                                              ] + [i for i in self.df.columns if '_txt' in i]
                                             )

        self.df = self.df[ok_cols]

    def adjust(self):
        self.df.columns = self.df.columns.str.replace(r'.', '_')
        self.df.columns = self.df.columns.str.replace(r'-', '_')
        self.df.columns = self.df.columns.str.replace(r'>=', '_more_equal_')
        self.df.columns = self.df.columns.str.replace(r'+', '_plus_')
        self.df.columns = self.df.columns.str.replace(r' ', '_')
        self.df.columns = self.df.columns.str.replace(r'/', '_')
        self.df.columns = self.df.columns.str.replace(r'[', '_')
        self.df.columns = self.df.columns.str.replace(r']', '_')
        self.df.columns = self.df.columns.str.replace(r'<', 'less')
