import pandas as pd
import numpy as np


import geopandas as gpd
import rioxarray
import xarray as xr


from src.models.gaussian_process import gp_inference
from src.preprocessing.preprocessing import prepare_rasters


class Enricher(object):
    """
    Class handling overall enrichment of scrapped dataset by noise data, geospatial data, embeddings and probably
    aggregated features
    TODO consider using decorators as it would be probably more elegant solution
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df  # dataframe to be enriched

    def __call__(self, *args, **kwargs):
        pass

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
        x_coords = xr.DataArray(self.df['x'].to_list(), dims="points")
        y_coords = xr.DataArray(self.df['x'].to_list(), dims="points")

        air_query = air.sel(x_coords, y_coords, method='nearest')
        built_query = built.sel(x_coords, y_coords, method='nearest')
        sun_query = sun.sel(x_coords, y_coords, method='nearest')
        noise_day_query = noise_day.sel(x_coords, y_coords, method='nearest')
        noise_night_query = noise_night.sel(x_coords, y_coords, method='nearest')

        # TODO update `self.df` and replace 0 (nodata) with nan probably

        """
        rds = rioxarray.open_rasterio("input.tif")
        rds.name = "data"
        df = rds.squeeze().to_dataframe().reset_index()
        geometry = gpd.points_from_xy(df.x, df.y)
        gdf = gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)
        """

    def add_criminality_data(self):
        """

        Returns
        -------

        """
        # TODO firstly needs write scraper to get all geojsons from https://kriminalita.policie.cz/download
        #  just needs easy loop using requests library on https://kriminalita.policie.cz/api/v2/downloads/201406.geojson
        #   where will be used always another year and month ... there are data from 2012

    def add_osm_data(self):
        """
        adds geospatial data retrieved from OSM
        Parameters
        ----------

        Returns
        -------

        """
        # TODO probably we will need loop through self.df and call `osmnx_nearest` func which will not be
        #  very effecient on large df

    def add_embeddings(self):
        pass


if __name__ == '__main__':
    en = Enricher(pd.DataFrame())
    en.add_quality_data(path='/home/emanuel/Documents/real_estate/data/geodata')
