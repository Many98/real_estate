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
        self.impute()
        self.categorize()
        self.encode()
        self.scale()

        self.df.to_csv('../data/tmp_preprocessed.csv', mode='w', index=False)

        return self.df

    def impute(self):
        """
        method to handle missing values imputation
        * required columns (nulls/Nones/nans/missing values are not tolerated => row removed):
            price (response); usable_area/floor area; disposition (in sreality extracted from header);
            lng, lat both are secondary features (probably not used in final model)

        * not required numeric (float) features:
            floor ???? how to define unknown, => maybe ordinal encoding with `unknown` level

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
                # TODO <>_dist features from bezreality will be replaced with euclidean distance/great circle dist
                    as in sreality ... this type of distance will be still relevant because can serve as proxy for
                    real network distance to place; this will be also computationally more efficient
            * nominal
                ownership; equipment; waste; gas; construction_type; electricity; heating; year_reconstruction;
                telecomunication; disposition
            * nominal or ordinal
                energy_effeciency => unknown level (possible nominal)
                age, condition

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
        self.df.fillna(value={'floor': -99, 'energy_effeciency': 'unknown', 'ownership': 'unknown', 'description': '',
                              'gas': 'unknown', 'waste': 'unknown', 'equipment': 'unknown', 'state': 'unknown',
                              'construction_type': 'unknown', 'place': 'unknown', 'electricity': 'unknown',
                              'heating': 'unknown', 'transport': 'unknown', 'year_reconstruction': 2038,
                              'telecomunication': 'unknown', 'age': 'undefined',
                              'air_quality': 'unknown', 'built_density': 'unknown', 'sun_glare': 'unknown'},
                       inplace=True)

        # fill <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i]
        self.df[dist_cols].fillna(value=10000, inplace=True)

        # fill has_<> & no_barriers attributes
        has_cols = [i for i in self.df.columns if 'has' in i]
        has_cols.append('no_barriers')
        self.df[has_cols].fillna(value=False, inplace=True)

        # fill daily/nightly noise with simple mean imputation
        self.df['daily_noise'].fillna(values=self.df['daily_noise'].mean(skipna=True), inplace=True)
        self.df['nightly_noise'].fillna(values=self.df['nightly_noise'].mean(skipna=True), inplace=True)

    def categorize(self):
        """
        method to categorize some numeric features
        Returns
        -------

        """
        # categorize <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i]
        dists = list(range(0, 1700, 100))
        for col in dist_cols:
            self.df[col] = pd.cut(self.df[col], bins=dists,
                   include_lowest=True,
                   labels=[f'{dists[i]}-{dists[i+1]-1}m' for i in range(len(dists[:-2]))] + ['>=1500m'])

        # categorize year_reconstruction  TODO define maybe better categories
        self.df['year_reconstruction'] = pd.cut(self.df['year_reconstruction'], bins=[0, 1950, 1980, 2000, 2010, 2015, 2020, 2025, 2040],
               include_lowest=True, labels=['<1950', '1951-1980', '1981-2000', '2001-2010', '2011-2015', '2016-2020',
                                            '2021-2025',
                                            'undefined'])

        # TODO do we want categorize `floor` features

    def encode(self):
        """
        method to handle on-hot and nominal encoding of categorical features
        Returns
        -------

        """
        pass

    def scale(self):
        """
        method to handle scaling/standardization of numeric features
        Returns
        -------

        """
        pass

if __name__ == '__main__':
    data = pd.read_csv('/home/emanuel/Music/prodej_breality_scraped.csv')
    pr = Preprocessor(data)
    pr.impute()
    pr.encode()