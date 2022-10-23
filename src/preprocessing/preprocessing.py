import pandas as pd
import numpy as np

import re
import itertools
import py7zr
import os

import geopandas as gpd
import osmnx as ox
import rioxarray
from shapely.geometry import Point, Polygon, MultiPolygon


# ## miscellaneous preprocessing
###############################


def prepare_atlas_cen_data(csv_path: str, dtypes: dict = {'when': str, 'href': str, 'area': str,
                                                          'price': str, 'long': float, 'lat': float}) -> pd.DataFrame:
    """
    function to prepare atlas cen data from scraped csv. mainly extract numbers from strings
    Parameters
    ----------
    csv_path :
    dtypes :

    Returns
    -------

    """
    try:
        df = pd.read_csv(csv_path, dtype=dtypes)
    except Exception as e:
        print(e)

    df['when'] = (df['when']
                  .apply(lambda x: x.replace(' ', ''))
                  .apply(lambda x: re.search("[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}", x).group())
                  # TODO if needed then will convert to datetime
                  )
    df['area'] = (df['area']
                  .apply(lambda x: x.replace(' ', '').replace('m2', ''))
                  .apply(lambda x: x.split(',')[1])
                  .astype(float)
                  )
    df['price'] = (df['price']
                   .apply(lambda x: x.replace(' ', '').replace('m2', '').replace('Kč', '').replace('/', ''))
                   .apply(lambda x: x.split()[-1])  # default splits on white spaces
                   .astype(float)
                   )
    df.rename(columns={"price": "price/m2"}, inplace=True)

    return df


# ### geo-enrichment via OpenStreetMap & raster data / geo-spatial preprocessing
################################################


def osmnx_call(long: float, lat: float, dist: int, tags: dict) -> gpd.GeoDataFrame:
    """
    Helper function to create call on OSM via OSMnx module to get
    attributes like nearest metro, parks etc.

    Parameters
    ----------
    long : float
        Longitude of point of interest
    lat : float
        Latitude of point of interest
    dist : int
        Maximal (bbox) distance from point of interest in metres
    tags: dict
        Schema defining geometries to be retrieved from OSM
    Returns
    -------

    """
    gdf = ox.geometries_from_point((lat, long), tags, dist)

    # only POLYGONS, MULTIPOLYGONS and POINTS are allowed
    gdf = gdf[(gdf['geometry'].apply(lambda geo: type(geo) is Polygon)) |
              (gdf['geometry'].apply(lambda geo: type(geo) is Point)) |
              (gdf['geometry'].apply(lambda geo: type(geo) is MultiPolygon))]

    # remove not needed tags
    not_drop = list(tags.keys())
    not_drop.extend(['geometry', 'unique_id', 'name'])
    gdf.drop(gdf.columns.difference(not_drop), 1, inplace=True)

    values = list(itertools.chain.from_iterable(list(tags.values())))

    # remove nans
    gdf['geometry'].dropna(axis=0, inplace=True)

    gdf.replace(np.nan, '', inplace=True)

    # merge names to one attribute
    gdf['what'] = gdf[list(tags.keys())].agg("".join, axis=1)

    # TODO remove other attributes

    return gdf


def osmnx_nearest(long: float, lat: float, dist: int) -> gpd.GeoDataFrame:
    """
    Finds nearest features (bus station, metro, park etc) from point of interest
    Parameters
    ----------
    long :
    lat :
    dist :
        Maximal (bbox) distance from point of interest in metres
    Returns
    -------

    """
    # TODO prepare tags

    tags = {'leisure': ['park'], 'amenity': ['university', 'college', 'kindergarten'],
            'building': ['supermarket', 'hospital']}
    tags2 = {'highway': ['bus_stop'], 'railway': ['tram_stop']}

    gdf = osmnx_call(long, lat, dist, tags)

    # process street network
    graph = ox.graph_from_point((lat, long), dist)
    #ox.plot_graph(graph)
    orig_nn = ox.nearest_nodes(graph, long, lat)  # nearest node in graph network to ours point of interest
    dest_nns = ox.nearest_nodes(graph, gdf['geometry'].x.values,
                                gdf['geometry'].y.values)  # nearest nodes to all relevant geometries
    orig_nns = [orig_nn] * len(dest_nns)
    edges = ox.shortest_path(graph, orig_nns, dest_nns)
    lengths = [ox.utils_graph.get_route_edge_attributes(graph, edge, 'length') for edge in edges]
    distances = [sum(x) for x in lengths]

    # TODO filter obtained dataset to left only nearest amenities/geometries

    return gdf


def prepare_rasters(path: str) -> tuple:
    """
    prepares raster data
    Parameters
    ----------
    path : str
        Path to directory with geodata (rasters/csv/.7z)
    Returns
    -------

    """
    if not os.path.isdir(os.path.join(path, 'geodata')):
        if os.path.isfile(os.path.join(path, 'geodata.7z')):
            with py7zr.SevenZipFile(os.path.join(path, 'geodata.7z'), mode='r') as z:
                z.extractall(path=path)
        else:
            raise Exception('Geodata not found !')

    # prague air quality
    #   5 levels 1 best 5 worse / 0 nodata
    if os.path.isfile(os.path.join(path, 'geodata', 'prague_air_quality.tif')):
        air = rioxarray.open_rasterio(os.path.join(path, 'geodata', 'prague_air_quality.tif'))
    else:
        raise Exception('Air quality data not found')

    # prague built up density
    #   5 levels 1 best 5 worse / 0 nodata
    if os.path.isfile(os.path.join(path, 'geodata', 'prague_built_up.tif')):
        built = rioxarray.open_rasterio(os.path.join(path, 'geodata', 'prague_built_up.tif'))
    else:
        raise Exception('Built up data not found')

    # prague day noise data (2016)
    if os.path.isfile(os.path.join(path, 'geodata', 'prague_noise_map_day_2016.tif')):
        noise_day = rioxarray.open_rasterio(os.path.join(path, 'geodata', 'prague_noise_map_day_2016.tif'))
    else:
        raise Exception('Daily noise level data not found')

    # prague night noise data (2016)
    if os.path.isfile(os.path.join(path, 'geodata', 'prague_noise_map_night_2016.tif')):
        noise_night = rioxarray.open_rasterio(os.path.join(path, 'geodata', 'prague_noise_map_night_2016.tif'))
    else:
        raise Exception('Nightly noise level data not found')

    # prague sun glare data
    #   5 levels 1 best 5 worse / 0 nodata
    if os.path.isfile(os.path.join(path, 'geodata', 'prague_sun_glare.tif')):
        sun = rioxarray.open_rasterio(os.path.join(path, 'geodata', 'prague_sun_glare.tif'))
    else:
        raise Exception('Nightly noise level data not found')

    return air, built, sun, noise_day, noise_night


# ###### NLP preprocessing
#####################

def generate_embeddings(text: str) -> np.array:
    pass


if __name__ == '__main__':

    osmnx_nearest(long=14.43809, lat=50.06851, dist=500)
