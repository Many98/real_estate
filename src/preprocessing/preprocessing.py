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
        num_ord_cols = [i for i in self.df.columns if 'dist' in i] + ['floor', 'year_reconstruction']
        ord_cols = ['energy_effeciency', 'air_quality', 'built_density', 'sun_glare']
        for d in num_ord_cols:
            self.df[d+'_num'] = self.df[d]  # create feature copy handled as numeric/continuous
            self.df[d+'_ord'] = self.df[d]  # create feature copy handled as ordinal
        for d in ord_cols:
            self.df[d+'_ord'] = self.df[d]  # create feature copy handled as ordinal

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
        self.df.fillna(value={'floor': -99, 'energy_effeciency': 'unknown', 'ownership': 'unknown', 'description': '',
                              'gas': 'unknown', 'waste': 'unknown', 'equipment': 'unknown', 'state': 'unknown',
                              'construction_type': 'unknown', 'place': 'unknown', 'electricity': 'unknown',
                              'heating': 'unknown', 'transport': 'unknown', 'year_reconstruction': 2038,
                              'telecomunication': 'unknown',
                              #'age': 'undefined',
                              'air_quality': 'unknown', 'built_density': 'unknown', 'sun_glare': 'unknown'},
                       inplace=True)

        # fill <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i]
        self.df.fillna(value={i: 10000 for i in dist_cols}, inplace=True)

        # fill has_<> & no_barriers attributes
        has_cols = [i for i in self.df.columns if 'has' in i]
        has_cols.append('no_barriers')
        self.df[has_cols] = self.df[has_cols].astype(bool)
        self.df.fillna(value={i: False for i in dist_cols}, inplace=True)

        # fill daily/nightly noise with simple mean imputation
        self.df.fillna(values={'daily_noise': self.df['daily_noise'].mean(skipna=True)}, inplace=True)
        self.df.fillna(values={'nightly_noise': self.df['nightly_noise'].mean(skipna=True)}, inplace=True)

    def categorize(self):
        """
        method to categorize some numeric features
        Returns
        -------

        """
        # TODO do we want categorize dist features
        # categorize <>_dist features
        dist_cols = [i for i in self.df.columns if 'dist' in i]
        dists = list(range(0, 1600, 100)) + [np.infty]
        for col in dist_cols:
            self.df[col] = pd.cut(self.df[col], bins=dists,
                   include_lowest=True,
                   labels=[f'{dists[i]}-{dists[i+1]-1}m' for i in range(len(dists[:-2]))] + ['>=1500m'])

        # categorize year_reconstruction  TODO define maybe better categories
        self.df['year_reconstruction'] = pd.cut(self.df['year_reconstruction'], bins=[0, 1950, 1980, 2000, 2010, 2015, 2020, 2025, np.infty],
               include_lowest=True, labels=['<1950', '1951-1980', '1981-2000', '2001-2010', '2011-2015', '2016-2020',
                                            '2021-2025',
                                            'undefined'])

        # TODO do we want categorize `floor` features

    def encode(self):
        """
        method to handle one-hot and nominal encoding of categorical features
        Returns
        -------

        """
        # one-hot
        self.df = pd.get_dummies(self.df, columns=['energy_effeciency', 'ownership', 'equipment',
                                                   'state',
                                                   'disposition'  # TODO needs known exact levels
                                                   'construction_type', 'year_reconstruction', 'heating'
                                                   #'gas', 'waste', 'telecomunication', 'electricity', 
                                                     ''], drop_first=True)

        # TODO ordinal encoding which featuers ??

    def scale(self):
        """
        method to handle scaling/standardization of numeric features
        Returns
        -------

        """
        # TODO which features will be numerical

    def remove(self):
        """
        method to remove unnecessary columns/features
        Returns
        -------

        """

if __name__ == '__main__':
    data = pd.read_csv('/home/emanuel/Music/prodej_breality_scraped.csv')
    pr = Preprocessor(data)
    pr.impute()
    pr.encode()

    # TODO changes:
    #  gas ?? --> binary hasgas <<
    #  waste
    #  electricity
    #  place, transport -> move to description
    #  no barriers not present -> remove ??
    #  telecomunication  >> to description ???
    #  what to do with dist features ???
    #  what to do with floor ??? ordinal categorize ?? how to handle missing
    #  which features will be numerical
    #  TODO  some preprocessing will need to be always on whole dataset e.g. robust scaling ???
    #  TODO maybe here create also categorized features e.g. `dist` and also numerical and test it on model
    #    same for similar features like floor, energy_effeciency(ordinal, vs onehot)
    #  TODO maybe we will need test more imputation techniques
    #  TODO maybe preprocessor step should be done as last after all data are appended to final csv so all scalings etc
    #   will return relevant values