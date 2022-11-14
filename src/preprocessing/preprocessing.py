import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class Preprocessor(object):
    """
    Class handling missing values, categorization, categorical encodings & standardization.
    Input dataframe is required to have "synchronized" values (two data sources needs to be unified/synchronized)

    TODO consider using decorators as it would be probably more elegant solution
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be preprocessed

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        self.expand()
        self.impute()
        self.categorize()
        self.encode()
        self.scale()
        self.remove()

        self.df.to_csv('../data/tmp_preprocessed.csv', mode='w', index=False)

        return self.df

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

        self.df.dropna(how='any', subset=['price', 'usable_area', 'header', 'long', 'lat'], inplace=True)

        # fill unknown/undefined
        self.df[['air_quality', 'built_density', 'sun_glare']] = \
            self.df[['air_quality', 'built_density', 'sun_glare']].astype(str)
        self.df.fillna(value={'floor': -99,  # -> arbitrary imputation  (floor handled only as numeric)
                              'year_reconstruction': 2038,
                              'year_reconstruction_num': 0,  # -> arbitrary
                              'year_reconstruction_ord': 2038,  # -> arbitrary
                              'energy_effeciency_ord': 'X',  # -> arbitrary
                              'energy_effeciency': 'unknown', 'ownership': 'unknown', 'description': '',
                              'gas': 'unknown', 'waste': 'unknown', 'equipment': 'unknown', 'state': 'unknown',
                              'construction_type': 'unknown', 'electricity': 'unknown',
                              'heating': 'unknown', 'transport': 'unknown',
                              'telecomunication': 'unknown',
                              # 'age': 'undefined',
                              'air_quality': 'unknown', 'built_density': 'unknown', 'sun_glare': 'unknown',
                              'air_quality_ord': 0, 'built_density_ord': 0, 'sun_glare_ord': 0  # -> arbitrary
                              },
                       inplace=True)

        # fill <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i and 'ord' not in i and 'num' not in i]
        self.df.fillna(value={i: 10000 for i in dist_cols}, inplace=True)
        self.df.fillna(value={i + '_ord': 10000 for i in dist_cols},
                       inplace=True)  # -> after categorization mapped to arbitrary highest ordinal value
        self.df.fillna(value={i + '_num': -999 for i in dist_cols}, inplace=True)  # -> arbitrary to handle >1500m in
        # numeric feature we will use indicator from one-hot to indicate whether >1500m

        # fill has_<> & no_barriers attributes
        has_cols = [i for i in self.df.columns if 'has' in i]
        has_cols.append('no_barriers')  # TODO probably `no_barriers will be removed as we do not have it`
        self.df[has_cols] = self.df[has_cols].astype(bool)
        self.df.fillna(value={i: False for i in dist_cols}, inplace=True)

        # fill daily/nightly noise with simple mean imputation
        self.df.fillna(values={'daily_noise': float(self.df['daily_noise'].mean(skipna=True))}, inplace=True)
        self.df.fillna(values={'nightly_noise': float(self.df['nightly_noise'].mean(skipna=True))}, inplace=True)

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

    def encode(self):
        """
        method to handle one-hot and nominal encoding of categorical features
        Returns
        -------

        """
        # one-hot encoding
        self.df = pd.get_dummies(self.df, columns=['energy_effeciency', 'ownership', 'equipment',
                                                   'state',
                                                   'disposition'  # TODO needs known exact levels
                                                   'construction_type', 'year_reconstruction', 'heating',
                                                   'air_quality', 'built_density', 'sun_glare',
                                                   # 'gas', 'waste', 'telecomunication', 'electricity',
                                                   ] +
                                                  [i for i in self.df.columns if
                                                   'dist' in i and 'ord' not in i and 'num' not in i],
                                 drop_first=True)

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
        pass


if __name__ == '__main__':
    data = pd.read_csv('/home/emanuel/Music/prodej_breality_scraped.csv')
    pr = Preprocessor(data)
    pr()

    # TODO changes:
    #  TODO  some preprocessing will need to be always on whole dataset e.g. robust scaling ???
    #  TODO maybe preprocessor step should be done as last after all data are appended to final csv so all scalings etc
    #   will return relevant values
    #  TODO what about new columns indicating missingness
