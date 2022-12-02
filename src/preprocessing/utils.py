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
    ox.config(use_cache=True, log_console=True, cache_folder='preprocessing/cache')
    gdf = ox.geometries_from_point((lat, long), tags, dist)

    # only POLYGONS, MULTIPOLYGONS and POINTS are allowed
    gdf = gdf[(gdf['geometry'].apply(lambda geo: type(geo) is Polygon)) |
              (gdf['geometry'].apply(lambda geo: type(geo) is Point)) |
              (gdf['geometry'].apply(lambda geo: type(geo) is MultiPolygon))]

    # remove not needed tags
    not_drop = list(tags.keys())
    not_drop.extend(['geometry', 'unique_id', 'name'])
    gdf.drop(gdf.columns.difference(not_drop), axis=1, inplace=True)

    values = list(itertools.chain.from_iterable(list(tags.values())))

    # remove nans
    gdf['geometry'].dropna(axis=0, inplace=True)

    # get rid of unwanted values
    gdf.replace(np.nan, '', inplace=True)
    for col in list(tags.keys()):
        gdf.loc[~gdf[col].isin(values), col] = ''

    # merge names to one attribute
    gdf['what'] = gdf[list(tags.keys())].agg("&".join, axis=1)
    #gdf['what'] = gdf['what'].apply(lambda x: '&')  not needed anymore

    gdf.drop(list(tags.keys()), axis=1, inplace=True)

    # replace collection of geometries like Multipolygon with multiple single geometries
    gdf = gdf.explode(ignore_index=True)

    # replace Polygons with centroids (needed for network processing) |
    #  another point representation e.g. nearest outer node of polygon will be better but centroid is "cheaper"
    gdf['geometry'] = gdf['geometry'].apply(lambda geo: geo.centroid if type(geo) is Polygon else geo)

    return gdf


def osmnx_nearest(gdf: gpd.GeoDataFrame, long: float, lat: float, dist: int = 1500,
                  dist_type: str = 'great_circle') -> pd.DataFrame:
    """
    Finds nearest features (bus station, metro, park etc) from point of interest
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        GeoDataframe which will be queried
    long : float
        longitude of point of interest (e.g. middle of prague)
    lat : float
        longitude of point of interest (e.g. middle of prague)
    dist : int
        Maximal distance from point of interest in metres
    dist_type: str:
        Type of distance to be processed:
            'great_circle' = great circle distance
            'network' = process street/road network and find nearest nodes in graph
    Returns
    -------

    """

    # TODO fix network for df if needed
    # TODO prepare version for bezrealitky (all tags) and sreality (only parks)
    # TODO https://geopandas.org/en/stable/docs/user_guide/geometric_manipulations.html
    #  https://gis.stackexchange.com/questions/349637/given-list-of-points-lat-long-how-to-find-all-points-within-radius-of-a-give
    if dist_type == 'network':
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
    elif dist_type == 'great_circle':
        distances = ox.distance.great_circle_vec(np.array([lat] * gdf.shape[0]), np.array([long] * gdf.shape[0]),
                                                 gdf['geometry'].y.values, gdf['geometry'].x.values)
    else:
        raise Exception(f'`dist_type` {dist_type} not supported')

    gdf['dist'] = distances

    gdfc = gdf.groupby('what', as_index=False).agg({'dist': 'min', 'geometry': 'first', 'name': 'first'})

    return pd.DataFrame(gdfc.drop(columns='geometry'))


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
                z.extractall(path=os.path.join(os.path.split(os.getcwd())[0], path.split('../')[-1]))
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

