import pandas as pd
import numpy as np
import os

import geopandas as gpd
import rioxarray
import xarray as xr

from tqdm import tqdm

from src.models.gaussian_process import gp_inference
from src.preprocessing.utils import prepare_rasters, osmnx_nearest, osmnx_call


class Enricher(object):
    """
    Class handling overall enrichment of scrapped dataset by noise data, geospatial data, embeddings
    TODO consider using decorators as it would be probably more elegant solution
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be enriched

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        self.add_gp('../models/fitted_gp_low')
        self.add_quality_data('../data/geodata/')
        self.add_criminality_data()
        self.add_osm_data()
        self.add_embeddings()

        self.df.to_csv('../data/tmp_enriched.csv', mode='w', index=False)

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
        # TODO prepare gdf here
        # TODO add new column to sreality and breality scraper

        tags = {'leisure': ['park', 'dog_park',  # park
                            'playground',  # hriste
                            'fitness_centre', 'stadium', 'swimming_pool', 'sports_centre', 'pitch'],  # sportoviste
                'amenity': ['school',  # skola
                            'kindergarten',  # skolka
                            'cafe', 'pub', 'restaurant',  # restaurace/krcma
                            'atm',  # bankomat
                            'post_office'  # posta
                            'clinic', 'hospital',  # doktory
                            'veterinary',  # veterinar
                            'pharmacy',  # lekarna
                            'cinema', 'theatre',  # kino/divadlo
                            ],
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

            relevant = joined[joined.index_right == _]

            # presence of attribute disposition => breality record
            if row['disposition'] is not np.nan or row[
                [i for i in self.df.columns if 'dist' in i and 'park' not in i]].isnull().values.any():
                nearest = osmnx_nearest(gdf=relevant, long=row["long"], lat=row['lat'], dist=dist,
                                        dist_type='great_circle')
            else:
                relevant = relevant[relevant.what.str.contains('park')]  # TODO handle cases when df is empty
                nearest = osmnx_nearest(gdf=relevant, long=row["long"], lat=row['lat'], dist=dist,
                                        dist_type='great_circle')

            # TODO update self.df with particular dists
            # TODO Estimated speed was about 1000 records/ 5 mins

    def add_embeddings(self):
        pass


class Generator(object):
    """
        Class handling generation of new aggregated features from existing features
        Some research is required prior to defining relevant aggregated features
        TODO consider using decorators as it would be probably more elegant solution
        """

    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be enriched

    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        self.df.to_csv('../data/tmp_final.csv', mode='w', index=False)

        return self.df


if __name__ == '__main__':
    data = pd.read_csv('/home/emanuel/Music/prodej_breality_scraped.csv')
    en = Enricher(data)
    #en.add_osm_data(dist=1500)
    #en.add_gp('/home/emanuel/Documents/real_estate/src/models/fitted_gp_low')
    en.add_quality_data(path='/home/emanuel/Documents/real_estate/data/geodata')
