import os

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder


class Preprocessor(object):
    """
    Class handling missing values, categorization, categorical encodings & standardization.
    Input dataframe is required to have "synchronized" values (two data sources needs to be unified/synchronized)
    https://towardsdatascience.com/pipeline-columntransformer-and-featureunion-explained-f5491f815f

    TODO consider using decorators as it would be probably more elegant solution
    """

    def __init__(self, df: pd.DataFrame, inference: bool, base: bool):
        self.df = df  # dataframe to be preprocessed
        self.inference = inference
        self.base = base  # whether to only perform imputation

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        if not self.df.empty:
            if not self.base:
                self.expand()

            self.impute()  # impute in pandas-like way too lazy to overwrite

            if not self.base:
                self.categorize()
                self.encode_ordinal()  # ordinal encoding in pandas-like way too lazy to overwrite

                if not self.inference:
                    # this breaks consistency a bit but is necessary because we need to hold state of imputer and
                    # one-hot encoder from train dataset and use it on test dataset
                    dist_cols = [i for i in self.df.columns if
                                 'dist' in i and 'ord' not in i and 'num' not in i]
                    categorical_to_ohe = ['energy_effeciency', 'ownership', 'equipment',
                                          'state',
                                          'disposition',
                                          'construction_type', 'year_reconstruction',
                                          'air_quality', 'built_density', 'sun_glare',
                                          'gas', 'waste', 'telecomunication', 'electricity', 'heating',
                                          ] + dist_cols

                    noise_cols = ['daily_noise', 'nightly_noise']

                    noise_imputer = SimpleImputer(missing_values=0.0, strategy='median')
                    ohe = OneHotEncoder(categories=[['unknown', 'G', 'E', 'B', 'D', 'C', 'A', 'F'],
                                                    ['Osobní', 'Státní/obecní', 'Družstevní', 'unknown'],
                                                    ['unknown', 'ne', 'Částečně', 'ano'],
                                                    ['unknown', 'V rekonstrukci', 'Před rekonstrukcí',
                                                     'Po rekonstrukci',
                                                     'Novostavba', 'Velmi dobrý', 'Dobrý', 'Ve výstavbě', 'Projekt',
                                                     'Špatný', ],
                                                    ['unknown', '1+kk', '1+1', '3+1', '3+kk', '2+kk', '4+1', '2+1',
                                                     '5+kk',
                                                     '4+kk', 'atypické', '6 pokojů a více', '5+1', '6+kk'],
                                                    ['unknown', 'Cihlová', 'Smíšená', 'Panelová', 'Skeletová',
                                                     'Kamenná',
                                                     'Montovaná', 'Nízkoenergetická', 'Drevostavba'],
                                                    ['<1950', '1951-1980', '1981-2000', '2001-2010', '2011-2015',
                                                     '2016-2020',
                                                     '2021-2025',
                                                     'undefined'],
                                                    ['unknown', '1.0', '2.0', '3.0', '4.0', '5.0'],
                                                    ['unknown', '1.0', '2.0', '3.0', '4.0', '5.0'],
                                                    ['unknown', '1.0', '2.0', '3.0', '4.0', '5.0'],
                                                    ['unknown', True, False],
                                                    ['unknown', True, False],
                                                    ['unknown', True, False],
                                                    ['unknown', True, False],
                                                    ['unknown', True, False],
                                                    ] + [['>=1500m', '0-99m', '100-199m', '200-299m', '300-399m',
                                                          '400-499m',
                                                          '500-599m', '600-699m', '700-799m', '800-899m', '900-999m',
                                                          '1000-1099m', '1100-1199m', '1200-1299m', '1300-1399m',
                                                          '1400-1499m'
                                                          ] for _ in range(len(dist_cols))],
                                        handle_unknown='error',  # TODO for now raise error
                                        sparse=False)
                    # TODO Target encoders using TargetEncoder()
                    # some theory https://www.kaggle.com/code/ryanholbrook/target-encoding/tutorial
                    categorical_to_tenc = ['has_balcony', 'has_cellar', 'has_garage',
                                           'has_garden', 'has_lift', 'has_loggia',
                                           'has_parking']
                    tenc = TargetEncoder(handle_unknown='error', sparse=False)

                    self.subprocessor = ColumnTransformer([
                        ('impute', noise_imputer, noise_cols),
                        ('ohe', ohe, categorical_to_ohe),
                        ('tenc', tenc, categorical_to_tenc)
                    ], remainder='passthrough', n_jobs=1)

                    # adding price_m2 column
                    self.df['price_m2'] = self.df['price']/self.df['usable_area']

                    # possible problem with noise_imputer in fit_transform, added "self.df['price_m2']" because of
                    # TargetEncoding
                    transformed = self.subprocessor.fit_transform(self.df, self.df['price_m2'])
                    col_names = ['_'.join(
                        i.replace('_', ' ').replace('impute', '').replace('remainder', '').replace('ohe', '').split())
                        for
                        i in self.subprocessor.get_feature_names_out()]

                    self._set_state()  # save state of subprocessor

                    self.df = pd.DataFrame(transformed, columns=col_names)

                else:
                    self._get_state()  # obtain latest state of subprocessor (apply to test data / inference)

                    transformed = self.subprocessor.transform(self.df)
                    col_names = ['_'.join(
                        i.replace('_', ' ').replace('impute', '').replace('remainder', '').replace('ohe', '').split())
                        for
                        i in self.subprocessor.get_feature_names_out()]

                    self.df = pd.DataFrame(transformed, columns=col_names)

            self.scale()
            self.remove()

        return self.df

    def _get_state(self):
        """
        auxiliary method to obtain state of `self.subprocessor` in `inference` phase
        Returns
        -------

        """
        if os.path.isfile('preprocessing/subprocessor_state.pickle'):
            with open('preprocessing/subprocessor_state.pickle', 'rb') as handle:
                self.subprocessor = pickle.load(handle)
        else:
            raise Exception('Subprocessor state dict not found. Consider running `ETL` in train phase first'
                            'to get properly fitted subprocessor. \n '
                            'subprocessor ~ (SimpleImputer + OneHotEncoder)')

    def _set_state(self):
        """
        auxiliary method to save state of `self.subprocessor` in `train` phase
        Returns
        -------

        """
        with open('preprocessing/subprocessor_state.pickle', 'wb') as handle:
            pickle.dump(self.subprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def expand(self):
        """
        method to expand some critical features to copies which will be numerical/ordinal because by default
         almost all features will be categorical/categorized and one-hot encoded
        Returns
        -------

        """
        num_ord_cols = [i for i in self.df.columns if 'dist' in i] + ['year_reconstruction']
        ord_cols = ['energy_effeciency', 'air_quality', 'built_density', 'sun_glare']
        for d in num_ord_cols:
            self.df[d + '_num'] = self.df[d]  # create feature copy handled as numeric/continuous
            self.df[d + '_ord'] = self.df[d]  # create feature copy handled as ordinal
        for d in ord_cols:
            self.df[d + '_ord'] = self.df[d]  # create feature copy handled as ordinal

    def impute(self):
        """
        # TODO interesting article about missing values
            https://towardsdatascience.com/all-about-missing-data-handling-b94b8b5d2184
        method to handle missing values imputation
        * required columns (nulls/Nones/nans/missing values are not tolerated => row removed):
            price (response); usable_area/floor area; disposition (in sreality extracted from header);
            lng, lat both are secondary features (probably not used in final model)

        * not required numeric (float) features:
            floor -> we will use typical imputing technique /mean/median/arbitrary

        * not required binary/bool (categorical) features:
            has_<> (all `has` features) => missing = False
            is_new (probably only from bezreality data) => missing = False
            no_barriers => missing = False or unknown ????
            civic amenities (most of them has True) => so use of corresponding `<>_dist` features will be
                probably enough
                also we will need to use OSM to impute all missing data i.e. find whether MHD is metro/bus/tram
                    in bezrealitky data
        * not required categorical features:
            * ordinal
                <>_dist (all `dist` features) (typically distance to nearest amenity) => this values are numeric but
                    probably will be better to use it as ordinal categorical variable with possible
                     `unknown` level e.g. levels will be:
                        `<50m`, `50-99m`, `100-149m` ... `>1500m`, `unknown`
                    also we will need to use OSM to impute all missing data i.e. find whether MHD is metro/bus/tram
                    in bezrealitky data
            * nominal
                ownership; equipment; waste; gas; construction_type; electricity; heating;
                telecomunication; disposition
            * nominal or ordinal
                energy_effeciency => unknown level (possible nominal)
                age, condition, year_reconstruction

        * textual:
            description (textual attribute => NLP) (kind of required)
            note (from sreality data) => NLP
            tags (from bezrealitky data) => NLP
            place (from bezrealitky data) => NLP
            ----
        * others
            some will be removed
            some columns will be added by `enricher`

        Parameters
        ----------

        Returns
        -------

        """

        self.df.dropna(how='any', subset=['price', 'usable_area', 'header', 'long', 'lat', 'disposition'], inplace=True)

        # fill unknown/undefined
        self.df[['air_quality', 'built_density', 'sun_glare']] = \
            self.df[['air_quality', 'built_density', 'sun_glare']].astype(str)
        self.df = self.df.replace("nan", np.nan)

        self.df.fillna(value={'floor': -99,  # -> arbitrary imputation  (floor handled only as numeric)
                              'year_reconstruction': 2038,
                              'energy_effeciency': 'unknown', 'ownership': 'unknown', 'description': '',
                              'gas': 'unknown', 'waste': 'unknown', 'equipment': 'unknown', 'state': 'unknown',
                              'construction_type': 'unknown', 'electricity': 'unknown',
                              'heating': 'unknown', 'transport': 'unknown',
                              'telecomunication': 'unknown',
                              # 'age': 'undefined',
                              'air_quality': 'unknown', 'built_density': 'unknown', 'sun_glare': 'unknown'
                              },
                       inplace=True)

        if not self.base:
            self.df.fillna(value={
                'year_reconstruction_num': 0,  # -> arbitrary
                'year_reconstruction_ord': 2038,  # -> arbitrary
                'energy_effeciency_ord': 'X',  # -> arbitrary
                'air_quality_ord': 0, 'built_density_ord': 0, 'sun_glare_ord': 0  # -> arbitrary
            },
                inplace=True)
        # fill <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i and 'ord' not in i and 'num' not in i and 'city' not in i]
        self.df.fillna(value={i: 10000 for i in dist_cols}, inplace=True)

        if not self.base:
            self.df.fillna(value={i + '_ord': 10000 for i in dist_cols},
                           inplace=True)  # -> after categorization mapped to arbitrary highest ordinal value
            self.df.fillna(value={i + '_num': -999 for i in dist_cols},
                           inplace=True)  # -> arbitrary to handle >1500m in
        # numeric feature we will use indicator from one-hot to indicate whether >1500m

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
        # categorize <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i and 'num' not in i]
        dists = list(range(0, 1600, 100)) + [np.infty]
        for col in dist_cols:
            self.df[col] = pd.cut(self.df[col], bins=dists,
                                  include_lowest=True,
                                  labels=[f'{dists[i]}-{dists[i + 1] - 1}m' for i in range(len(dists[:-2]))] + [
                                      '>=1500m'])

        # categorize year_reconstruction  TODO define maybe better categories
        self.df['year_reconstruction'] = pd.cut(self.df['year_reconstruction'],
                                                bins=[0, 1950, 1980, 2000, 2010, 2015, 2020, 2025, np.infty],
                                                include_lowest=True,
                                                labels=['<1950', '1951-1980', '1981-2000', '2001-2010', '2011-2015',
                                                        '2016-2020',
                                                        '2021-2025',
                                                        'undefined'])
        self.df['year_reconstruction_ord'] = pd.cut(self.df['year_reconstruction_ord'],
                                                    bins=[0, 1950, 1980, 2000, 2010, 2015, 2020, 2025, np.infty],
                                                    include_lowest=True,
                                                    labels=['<1950', '1951-1980', '1981-2000', '2001-2010',
                                                            '2011-2015',
                                                            '2016-2020',
                                                            '2021-2025',
                                                            'arbitrary'])

    def encode_ordinal(self):
        """
        method to handle one-hot and nominal encoding of categorical features
        Returns
        -------

        """
        # ordinal encoding
        self.df['energy_effeciency_ord'] = self.df['energy_effeciency_ord'].replace({'X': 0, 'A': 1, 'B': 2, 'C': 3,
                                                                                     'D': 4, 'E': 5, 'F': 6, 'G': 7})
        self.df['year_reconstruction_ord'] = self.df['year_reconstruction_ord'].replace({'<1950': 1, '1951-1980': 2,
                                                                                         '1981-2000': 3,
                                                                                         '2001-2010': 4,
                                                                                         '2011-2015': 5,
                                                                                         '2016-2020': 6,
                                                                                         '2021-2025': 7,
                                                                                         'arbitrary': 0})

        dist_cols = [i for i in self.df.columns if 'dist_ord' in i]
        dists = list(range(0, 1600, 100)) + [np.infty]
        l = [f'{dists[i]}-{dists[i + 1] - 1}m' for i in range(len(dists[:-2]))] + ['>=1500m']
        for col in dist_cols:
            self.df[col] = self.df[col].replace(
                {k: v for k, v in zip(l, list(range(1, len(l) + 1)))})

    def scale(self):
        """
        method to handle scaling/standardization of numeric features
        Returns
        -------

        """
        # TODO which features will be numerical, scaling probably will no be necessary
        #  probably only response, floor and usable_area will be numeric (maybe dists), `noise`
        pass

    def remove(self):
        """
        method to remove unnecessary columns/features
        Returns
        -------

        """
        self.df.drop_duplicates(subset=['hash'], ignore_index=True, inplace=True)
