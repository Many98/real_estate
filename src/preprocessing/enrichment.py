import pandas as pd
import numpy as np
import os
import warnings

import geopandas as gpd
import xarray as xr

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

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        if not self.df.empty:
            self.add_gp('models/fitted_gp_low')
            self.add_quality_data('../data/geodata/')
            self.add_criminality_data()
            self.add_location('../data/geodata/TMMESTSKECASTI_P.json')
            self.add_osm_data()

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

    def add_criminality_data(self):
        """

        Returns
        -------

        """
        # TODO firstly needs write scraper to get all geojsons from https://kriminalita.policie.cz/download
        #  just needs easy loop using requests library on https://kriminalita.policie.cz/api/v2/downloads/201406.geojson
        #   where will be used always another year and month ... there are data from 2012
        pass

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

                    self.df.at[_, 'bus_station_dist'] = float(nearest[nearest.what.str.contains('bus_stop')]['dist'].min())
                    self.df.at[_, 'train_station_dist'] = float(nearest[nearest.what.str.contains('train')]['dist'].min())
                    self.df.at[_, 'subway_station_dist'] = float(nearest[nearest.what.str.contains('subway')]['dist'].min())
                    self.df.at[_, 'tram_station_dist'] = float(nearest[nearest.what.str.contains('tram')]['dist'].min())
                    self.df.at[_, 'post_office_dist'] = float(nearest[nearest.what.str.contains('post_off')]['dist'].min())
                    self.df.at[_, 'atm_dist'] = float(nearest[nearest.what.str.contains('atm')]['dist'].min())
                    self.df.at[_, 'doctor_dist'] = float(nearest[nearest.what.str.contains('hospital|clinic')]['dist'].min())
                    self.df.at[_, 'vet_dist'] = float(nearest[nearest.what.str.contains('veterinary')]['dist'].min())
                    self.df.at[_, 'primary_school_dist'] = float(nearest[nearest.what.str.contains('school')]['dist'].min())
                    self.df.at[_, 'kindergarten_dist'] = float(nearest[nearest.what.str.contains('kinder')]['dist'].min())
                    self.df.at[_, 'supermarket_grocery_dist'] = float(nearest[nearest.what.str.contains('supermarket|general|mall')]['dist'].min())
                    self.df.at[_, 'restaurant_pub_dist'] = float(nearest[nearest.what.str.contains('restaurant|pub|cafe')]['dist'].min())
                    self.df.at[_, 'playground_dist'] = float(nearest[nearest.what.str.contains('playground')]['dist'].min())
                    self.df.at[_, 'sports_field_dist'] = float(nearest[nearest.what.str.contains('stadium|sports|fitness|swim')]['dist'].min())
                    self.df.at[_, 'theatre_cinema_dist'] = float(nearest[nearest.what.str.contains('theatre|cinema')]['dist'].min())
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

    def __init__(self, df: pd.DataFrame, base: bool):
        self.df = df  # dataframe to be enriched
        self.base = base  # whether to perform only some transformations

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        if not self.df.empty:
            if not self.base:
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
