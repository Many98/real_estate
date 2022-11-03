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
        self.encode()
        self.scale()
        return self.df

    def impute(self):
        """
        method to handle missing values imputation
        * required columns (nulls/Nones/nans/missing values are not tolerated => row removed):
            price (response); usable_area/floor area; disposition (in sreality extracted from header);
            lng, lat both are secondary features (probably not used in final model)

        * not required numeric (float) features:
            floor ???? how to define unknown, => maybe ordinal encoding with `unknown` level
            <>_area (all `area` features) it seems they will be very sparse => use of corresponding
                binary features will be probably enough
        # TODO probably <>_area features should be removed from scraping

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
            TODO all these textual columns will be somewhere merged into one text / probably in synchronize phase
        * others
            some will be removed
            some columns will be added by `enricher`

        Parameters
        ----------

        Returns
        -------

        """

        pass

    def categorize(self):
        """
        method to categorize some numeric features
        Returns
        -------

        """
        pass

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