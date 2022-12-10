import pandas as pd
import numpy as np
import os
import warnings

import geopandas as gpd
import xarray as xr
import datetime

import requests
from compress_fasttext.feature_extraction import FastTextTransformer
from compress_fasttext.models import CompressedFastTextKeyedVectors

from tqdm import tqdm

from models.gaussian_process import gp_inference
from preprocessing.utils import prepare_rasters, osmnx_nearest, osmnx_call


class Enricher(object):
    """
    Class handling overall enrichment of scrapped dataset by noise data, geospatial data, embeddings
    TODO consider using decorators as it would be probably more elegant solution
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be enriched

    def __call__(self, inference, *args, **kwargs) -> pd.DataFrame:
        if not self.df.empty:
            try:
                self.add_gp('models/fitted_gp_low')
                self.add_quality_data('../data/geodata/')
                self.add_location('../data/geodata/TMMESTSKECASTI_P.json')
                self.add_osm_data()
                self.add_criminality_data('../data/criminality.csv')
            except Exception as e:
                if not inference:
                    print(e)
                else:
                    return pd.DataFrame()

        return self.df

    def add_gp(self, path: str):
        """
        Function to call already fitted gaussian process over dataframe (with longitudes and latitudes)
        to add new feature (predicted mean price and variance/ confidence intervals bounds)
        Parameters
        ----------
        path : str
            Path to pickled model

        Returns
        -------

        """

        X = self.df[['long', 'lat']].to_numpy()

        mean, std, ci_low, ci_high = gp_inference(X, path)

        # TODO price is predicted per m2 so probably multiply by usable area or something
        self.df['gp_mean_price'] = mean
        self.df['gp_std_price'] = std
        # kind of 95% CI
        self.df['gp_ci_low_price'] = ci_low
        self.df['gp_ci_high_price'] = ci_high

    def add_quality_data(self, path: str):
        """
        Function to add features about mean noise, criminality etc to dataframe
        TODO maybe weighted mean noise (using convolution) would be nice but for now "nearest" value is used
        Parameters
        ----------
        path : str
            Path to directory with geodata (rasters/csv/.7z)
        Returns
        -------

        """
        air, built, sun, noise_day, noise_night = prepare_rasters(path)

        # query points
        x_coords = xr.DataArray(self.df['long'].to_list(), dims="points")
        y_coords = xr.DataArray(self.df['lat'].to_list(), dims="points")

        air_query = air.sel(x=x_coords, y=y_coords, method='nearest').values.flatten()
        built_query = built.sel(x=x_coords, y=y_coords, method='nearest').values.flatten()
        sun_query = sun.sel(x=x_coords, y=y_coords, method='nearest').values.flatten()
        noise_day_query = noise_day.sel(x=x_coords, y=y_coords, method='nearest').values.flatten()
        noise_night_query = noise_night.sel(x=x_coords, y=y_coords, method='nearest').values.flatten()

        self.df['air_quality'] = np.where(air_query == 0, np.nan, air_query)
        self.df['built_density'] = np.where(built_query == 0, np.nan, air_query)
        self.df['sun_glare'] = np.where(sun_query == 0, np.nan, air_query)
        self.df['daily_noise'] = noise_day_query
        self.df['nightly_noise'] = noise_night_query

    def add_criminality_data(self, path: str, dist=150):
        """
        method to add information about criminality in Prague
        Parameters
        ----------
        path : str
            Path to directory with criminality_data (data/criminality.csv)
        Returns
        -------

        """
        self.df['theft_crime'] = 0.
        self.df['burglary_crime'] = 0.
        self.df['violence_crime'] = 0.
        self.df['accident_crime'] = 0.
        self.df['murder_crime'] = 0.
        self.df['hijack_crime'] = 0.

        crime_df = pd.read_csv(path, sep=',', delimiter=None, encoding="utf8")

        crime_df['date'] = pd.to_datetime(crime_df['date'], utc=True)
        crime_df['date'] = pd.to_datetime(crime_df['date']).dt.date

        crime_df = crime_df[crime_df['date'] > datetime.date(2016, 1, 1)]

        # list of all crimes ['krádeže na osobách' 'krádeže součástek aut' 'vloupání do prodejny'
        #  'krádeže motorových vozidel (dvoustopových)' 'krádeže jízdních kol'
        #  'loupež' 'vloupání do bytu' 'vydírání' 'úmyslné ublížení na zdraví'
        #  'vloupání do rodinných domů' 'vloupání do ubytovacích objektů'
        #  'dopravní nehody' 'výtržnictví' 'vloupání do restaurace'
        #  'nebezpečné vyhrožování' 'omezování osobní svobody' 'obecné ohrožení'
        #  'krádeže motorových vozidel (jednostopových)' 'nedovolené ozbrojování'
        #  'útok proti výkonu pravomoci stát. orgánu' 'vražda' 'chladná zbraň'
        #  'násilí proti skupině/jednotlivci' 'střelná zbraň' 'rvačka'
        #  'obchod s lidmi' 'únos']

        # We omit 'obchod s lidmi' and group types of crimes into several categories
        nasili = ['obecné ohrožení', 'omezování osobní svobody', 'nebezpečné vyhrožování',
                  'nedovolené ozbrojování', 'vydírání', 'rvačka', 'úmyslné ublížení na zdraví',
                  'výtržnictví', 'útok proti výkonu pravomoci stát. orgánu', 'chladná zbraň',
                  'střelná zbraň', 'násilí proti skupině/jednotlivci']
        kradez = ['krádeže na osobách', 'loupež', 'krádeže součástek aut', 'krádeže motorových vozidel (dvoustopových)',
                  'krádeže jízdních kol', 'krádeže motorových vozidel (jednostopových)']
        vloupani = ['vloupání do restaurace', 'vloupání do rodinných domů', 'vloupání do bytu',
                    'vloupání do prodejny', 'vloupání do ubytovacích objektů']

        crime_df = crime_df[crime_df.types != 'obchod s lidmi']
        crime_df[['types']] = crime_df[['types']].replace(dict.fromkeys(nasili, 'násilí'))
        crime_df[['types']] = crime_df[['types']].replace(dict.fromkeys(kradez, 'krádež'))
        crime_df[['types']] = crime_df[['types']].replace(dict.fromkeys(vloupani, 'vloupání'))

        # All crimes we have in our data
        crime_list = ['krádež', 'vloupání', 'násilí', 'dopravní nehody', 'vražda', 'únos']

        # If we got any other type of crime, it is omitted here
        crime_df = crime_df[crime_df['types'].isin(crime_list)]

        # defines weights for crimes used later for sort (not normalized)
        crime_w = [0.05, 0.057, 0.057, 0.045, 0.2, 0.07]
        crime_df['crime_idx'] = 0

        for x in range(len(crime_list)):
            crime_df.loc[crime_df['types'] == crime_list[x], 'crime_idx'] = crime_w[x]

        max_date = max(crime_df['date'])
        crime_df['years_past'] = (max_date - crime_df['date']) / np.timedelta64(1, 'Y')
        discount_factor = 0.9
        crime_df['disc_crime'] = discount_factor ** (np.floor(crime_df['years_past'])) * crime_df['crime_idx']

        # final filter now using weight based on seriousness of crime `crime_idx` and time passed
        crime_df = crime_df[crime_df.disc_crime >= 0.045]

        # Now we have our data almost prepared
        gdf_from_crime = gpd.GeoDataFrame(crime_df,
                                          geometry=gpd.points_from_xy(
                                              crime_df.x,
                                              crime_df.y,
                                          ),
                                          crs='epsg:4326',
                                          )

        gdf_from_df = gpd.GeoDataFrame(self.df,
                                       geometry=gpd.points_from_xy(
                                           self.df.long,
                                           self.df.lat,
                                       ),
                                       crs='epsg:4326',
                                       )

        gdf_from_df = gdf_from_df.to_crs('epsg:32633')
        # create buffer of `dist` m
        buffer = gdf_from_df.buffer(dist)
        buffer = buffer.to_crs('epsg:4326')
        buf = gpd.GeoDataFrame(geometry=buffer)

        # perform spatial join
        joined = gpd.sjoin(gdf_from_crime, buf)

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc='Processing criminality data'):

            relevant = joined.loc[joined.index_right == _, :]

            if not relevant.empty:
                df_group_crime = relevant.groupby('types', as_index=False).count()

                self.df.at[_, 'theft_crime'] = df_group_crime[df_group_crime['types'] == 'krádež']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'krádež']['id'].empty else 0
                self.df.at[_, 'burglary_crime'] = df_group_crime[df_group_crime['types'] == 'vloupání']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'vloupání']['id'].empty else 0
                self.df.at[_, 'violence_crime'] = df_group_crime[df_group_crime['types'] == 'násilí']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'násilí']['id'].empty else 0
                self.df.at[_, 'accident_crime'] = df_group_crime[df_group_crime['types'] == 'dopravní nehody']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'opravní nehody']['id'].empty else 0
                self.df.at[_, 'murder_crime'] = df_group_crime[df_group_crime['types'] == 'vražda']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'vražda']['id'].empty else 0
                self.df.at[_, 'hijack_crime'] = df_group_crime[df_group_crime['types'] == 'únos']['id'].item() if not \
                    df_group_crime[df_group_crime['types'] == 'únos']['id'].empty else 0

        self.df[[i for i in self.df.columns if '_crime' in i]] = \
            self.df[[i for i in self.df.columns if '_crime' in i]].fillna(0.)

    def add_location(self, geojson: str):
        """
        method to add information about location (part of prague)
        Parameters
        ----------
        geojson :

        Returns
        -------

        """
        gdf = gpd.read_file(geojson)
        gdf = gdf.to_crs('epsg:4326')

        gdf_from_df = gpd.GeoDataFrame(self.df,
                                       geometry=gpd.points_from_xy(
                                           self.df.long,
                                           self.df.lat,
                                       ),
                                       crs=gdf.crs,
                                       )
        pointInPoly = gpd.sjoin(gdf_from_df, gdf[['NAZEV_MC', 'geometry']], how='left', predicate='within')

        self.df = pd.DataFrame(pointInPoly[pointInPoly.columns.difference(['index_right'])])
        self.df.rename(columns={"NAZEV_MC": "city_district"}, inplace=True)

    def add_osm_data(self, dist: int = 1500):
        """
        adds geospatial data retrieved from OSM
        Parameters
        ----------
        dist: int
            Max distance to process amenities
        Returns
        -------

        """

        tags = {'leisure': ['park', 'dog_park',  # park
                            'playground',  # hriste
                            'fitness_centre', 'stadium', 'swimming_pool', 'sports_centre', 'pitch'],  # sportoviste
                'amenity': ['school',  # skola
                            'kindergarten',  # skolka
                            'cafe', 'pub', 'restaurant',  # restaurace/krcma
                            'atm',  # bankomat
                            'post_office',  # posta
                            'clinic', 'hospital',  # doktory
                            'veterinary',  # veterinar
                            'pharmacy',  # lekarna
                            'cinema', 'theatre',  # kino/divadlo
                            ],
                'building': ['train_station'],
                'shop': ['supermarket', 'mall', 'general'],  # obchod
                'highway': ['bus_stop'],  # bus
                'railway': ['tram_stop', 'station'],  # tram / train station
                'station': ['subway']}  # metro

        # TODO make this more robust
        gdf = osmnx_call(14.43809, 50.06851, 18000, tags)  # prepare big geodataframe for whole prague

        gdf_from_df = gpd.GeoDataFrame(self.df,
                                       geometry=gpd.points_from_xy(
                                           self.df.long,
                                           self.df.lat,
                                       ),
                                       crs=gdf.crs,
                                       )
        # reproject to projected coordinate system UTM 33N (to get "proper" metres)
        gdf_from_df = gdf_from_df.to_crs('epsg:32633')
        # create buffer of 1500m
        buffer = gdf_from_df.buffer(dist)
        buffer = buffer.to_crs(gdf.crs)
        buf = gpd.GeoDataFrame(geometry=buffer)

        # perform spatial join
        joined = gpd.sjoin(gdf, buf)

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc='Processing OSM data'):  #

            relevant = joined.loc[joined.index_right == _, :]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if row['name'] == 'breality' or row[
                    [i for i in self.df.columns if 'dist' in i and 'park' not in i]].isnull().values.any():
                    nearest = osmnx_nearest(gdf=relevant, long=row["long"], lat=row['lat'], dist=dist,
                                            dist_type='great_circle')

                    self.df.at[_, 'bus_station_dist'] = float(
                        nearest[nearest.what.str.contains('bus_stop')]['dist'].min())
                    self.df.at[_, 'train_station_dist'] = float(
                        nearest[nearest.what.str.contains('train')]['dist'].min())
                    self.df.at[_, 'subway_station_dist'] = float(
                        nearest[nearest.what.str.contains('subway')]['dist'].min())
                    self.df.at[_, 'tram_station_dist'] = float(nearest[nearest.what.str.contains('tram')]['dist'].min())
                    self.df.at[_, 'post_office_dist'] = float(
                        nearest[nearest.what.str.contains('post_off')]['dist'].min())
                    self.df.at[_, 'atm_dist'] = float(nearest[nearest.what.str.contains('atm')]['dist'].min())
                    self.df.at[_, 'doctor_dist'] = float(
                        nearest[nearest.what.str.contains('hospital|clinic')]['dist'].min())
                    self.df.at[_, 'vet_dist'] = float(nearest[nearest.what.str.contains('veterinary')]['dist'].min())
                    self.df.at[_, 'primary_school_dist'] = float(
                        nearest[nearest.what.str.contains('school')]['dist'].min())
                    self.df.at[_, 'kindergarten_dist'] = float(
                        nearest[nearest.what.str.contains('kinder')]['dist'].min())
                    self.df.at[_, 'supermarket_grocery_dist'] = float(
                        nearest[nearest.what.str.contains('supermarket|general|mall')]['dist'].min())
                    self.df.at[_, 'restaurant_pub_dist'] = float(
                        nearest[nearest.what.str.contains('restaurant|pub|cafe')]['dist'].min())
                    self.df.at[_, 'playground_dist'] = float(
                        nearest[nearest.what.str.contains('playground')]['dist'].min())
                    self.df.at[_, 'sports_field_dist'] = float(
                        nearest[nearest.what.str.contains('stadium|sports|fitness|swim')]['dist'].min())
                    self.df.at[_, 'theatre_cinema_dist'] = float(
                        nearest[nearest.what.str.contains('theatre|cinema')]['dist'].min())
                    self.df.at[_, 'pharmacy_dist'] = float(nearest[nearest.what.str.contains('pharmacy')]['dist'].min())
                    self.df.at[_, 'park_dist'] = float(nearest[nearest.what.str.contains('park')]['dist'].min())
                else:
                    relevant = relevant[relevant.what.str.contains('park')]  # TODO handle cases when df is empty
                    nearest = osmnx_nearest(gdf=relevant, long=row["long"], lat=row['lat'], dist=dist,
                                            dist_type='great_circle')
                    self.df.at[_, 'park_dist'] = float(nearest[nearest.what.str.contains('park')]['dist'].min())


class Generator(object):
    """
        Class handling generation of new aggregated features from existing features
        Some research is required prior to defining relevant aggregated features
        TODO consider using decorators as it would be probably more elegant solution
        """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be enriched

    def __call__(self, *args, **kwargs) -> pd.DataFrame:

        if not self.df.empty:
            self.add_fasttext_embeddings()  # fastext embeddings did not added anything to model performance
            self.add_electra_embeddings()
            self.add_roberta_embeddings()

        return self.df

    def add_fasttext_embeddings(self):
        """
        adds fasttext embeddings
        Note that fasttext binaries are very huge, for czech it is about 7GB therefore
        compressed fasttext model is used instead. for details see
        https://vasnetsov93.medium.com/shrinking-fasttext-embeddings-so-that-it-fits-google-colab-cd59ab75959e
        https://github.com/avidale/compress-fasttext
        Returns
        -------

        """
        # fasttext.util.download_model('cs', if_exists='strict') # download model for czech -> too big
        # https://github.com/avidale/compress-fasttext
        if not os.path.isfile('models/fasttext-cs-mini'):
            with requests.get('https://zenodo.org/record/4905385/files/fasttext-cs-mini?download=1', stream=True) as r:
                with open('models/fasttext-cs-mini', 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        small_model = CompressedFastTextKeyedVectors.load('models/fasttext-cs-mini')

        # `FastTextTransformer` has sklearn-like API
        ft = FastTextTransformer(model=small_model)

        embeddings = ft.transform(self.df.description)  # represents a text as the average of the embedding of its words

        dd = pd.DataFrame(data=embeddings, columns=[f'ft_emb_{i}' for i in range(1, embeddings.shape[1] + 1)])
        self.df = pd.concat([self.df, dd], axis=1)

    def add_electra_embeddings(self):
        pass

    def add_roberta_embeddings(self):
        pass
