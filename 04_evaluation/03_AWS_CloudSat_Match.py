#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Identify CloudSat profile points that are within 15 km of PROMICE weather stations.

"""

# Import modules
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import os
from Functions import geopandas_utils

# Define coordinate system
df_crs = 'EPSG:4326'

# Define PROMICE coordinates
promice = pd.read_csv('/home/johnny/Documents/Clouds/Data/PROMICE_Coordinates.csv')

# Define CloudSat data
files = sorted(glob.glob('/home/johnny/Documents/Clouds/Data/Merged_Data/*'))

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(promice['lon'], promice['lat'])]
promice_gdf = gpd.GeoDataFrame(promice, crs=df_crs, geometry=geometry)

near_lat = []
near_lon = []
near_lw_cs = []
near_lw_as = []
near_sw_cs = []
near_sw_as = []
near_datetime = []
near_station = []
near_filename = []

for i in files:
    
    # Get path and filename separately
    infilepath1, infilename1 = os.path.split(i)
    # Get file name without extension
    infileshortname1, extension1 = os.path.splitext(infilename1)
    
    # Read CloudSat data
    df = pd.read_csv(i)
    
    # Convert CloudSat to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
    gdf = gpd.GeoDataFrame(df, crs=df_crs, geometry=geometry)
    
    # Call function
    closest_point, idx = geopandas_utils.nearest_neighbor(gdf.reset_index(drop=True), 
                                                     promice_gdf.reset_index(drop=True), 
                                                     return_dist=True)  
    
    # Add station and distance to DataFrame
    gdf['station'] = closest_point['station']
    gdf['distance'] = closest_point['distance']
    
    # Filter only close distances
    near = gdf[gdf['distance'] < 15000]
    
    if near.shape[0] == 0:
        pass
    else:
        # Append to list
        near_lat.append(near['lat'].values)
        near_lon.append(near['lon'].values)
        near_lw_cs.append(near['lw_down_cs'].values)
        near_sw_cs.append(near['sw_down_cs'].values)
        near_lw_as.append(near['lw_down_as'].values)
        near_sw_as.append(near['sw_down_as'].values)
        near_datetime.append(near['datetime'].values)
        near_station.append(near['station'].values)
        near_filename.append(np.repeat(infileshortname1, near.shape[0]))
        
near_lat_flat = [item for sublist in near_lat for item in sublist]
near_lon_flat = [item for sublist in near_lon for item in sublist]
near_lw_cs_flat = [item for sublist in near_lw_cs for item in sublist]
near_lw_as_flat = [item for sublist in near_lw_as for item in sublist]
near_sw_cs_flat = [item for sublist in near_sw_cs for item in sublist]
near_sw_as_flat = [item for sublist in near_sw_as for item in sublist]
near_datetime_flat = [item for sublist in near_datetime for item in sublist]
near_station_flat = [item for sublist in near_station for item in sublist]
near_filename_flat = [item for sublist in near_filename for item in sublist]

# Put into DataFrame
df = pd.DataFrame(list(zip(near_lat_flat,near_lon_flat,near_lw_cs_flat,near_lw_as_flat,
                           near_sw_cs_flat, near_sw_as_flat,near_datetime_flat, 
                           near_station_flat, near_filename_flat)))

df.columns = ['lat', 'lon', 'lw_cs', 'lw_as', 'sw_cs', 'sw_as',
              'datetime', 'station', 'filename']
        
# Save to csv
df.to_csv('/home/johnny/Documents/Clouds/Data/Validation/CloudSat_Matched.csv')
        
        
        
        
        
        
        
        
        
        
        


