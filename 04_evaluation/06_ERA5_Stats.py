#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Regrid Greenland basins, elevation and mask to ERA5 grid

"""

# Import modules
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import os
#import pyresample
import netCDF4
#from pyproj import Transformer
from Functions import geopandas_utils
from datetime import timedelta, datetime
from shapely.geometry import Point

# Define coordinate system
df_crs = 'EPSG:4326'

# Define CloudSat data
cs_files = sorted(glob.glob('/home/johnny/Documents/Clouds/Data/Merged_Data/*'))

# =============================================================================
# # Define ice sheet mask and elevations
# ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
# ismip_lon = ismip.variables['lon'][:]
# ismip_lat = ismip.variables['lat'][:]
# ismip_mask = ismip.variables['ICE'][:]
# ismip_elev = ismip.variables['SRF'][:]
# 
# # Define region mask
# regions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/Ultimate_Mask.nc')
# regions_y = regions.variables['y'][:]
# regions_x = regions.variables['x'][:]
# regions_1 = regions.variables['North'][:]
# regions_2 = regions.variables['NorthEast'][:]
# regions_3 = regions.variables['East'][:]
# regions_4 = regions.variables['SouthEast'][:]
# regions_5 = regions.variables['South'][:]
# regions_6 = regions.variables['SouthWest'][:]
# regions_7 = regions.variables['West'][:]
# regions_8 = regions.variables['NorthWest'][:]
# 
# regions_2[regions_2 > 0] = 2
# regions_3[regions_3 > 0] = 3
# regions_4[regions_4 > 0] = 4
# regions_5[regions_5 > 0] = 5
# regions_6[regions_6 > 0] = 6
# regions_7[regions_7 > 0] = 7
# regions_8[regions_8 > 0] = 8
# 
# regions_mask = regions_1 + regions_2 + regions_3 + regions_4 + regions_5 + regions_6 +\
#     regions_7 + regions_8
# =============================================================================

# Define ERA5 data
era5 = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/adaptor.mars.internal-1606183801.7756338-19985-26-b03535d9-4837-4050-a18e-1555bb168a01.nc')
era5_lon = era5.variables['longitude'][:]
era5_lat = era5.variables['latitude'][:]
era5_xx, era5_yy = np.meshgrid(era5_lon, era5_lat)

# Get time
base = datetime(1900,1,1)
era5_time = pd.DataFrame(era5.variables['time'][:], columns=['hours'])
era5_time['datetime'] = era5_time['hours'].apply(lambda x: base + timedelta(hours=x))

# Convert to GeoDataFrame
era5_df = pd.DataFrame(list(zip((np.ravel(era5_xx)), np.ravel(era5_yy))), columns = ['lon', 'lat'])
geometry = [Point(xy) for xy in zip(era5_df['lon'], era5_df['lat'])]
era_gdf = gpd.GeoDataFrame(era5_df, crs=df_crs, geometry=geometry)

# =============================================================================
# # Convert from region mask stereographic to WGS84
# transformer = Transformer.from_crs(3413, 4326)
# lat, lon = transformer.transform(regions_x, regions_y)
# 
# # Define grid using pyresample       
# orig_def_regions = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
# targ_def = pyresample.geometry.GridDefinition(lons=era5_xx, lats=era5_yy)
# orig_def_elev = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
# 
# # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
# region_resample = pyresample.kd_tree.resample_nearest(source_geo_def=orig_def_regions, 
#                                                  target_geo_def=targ_def, 
#                                                  data=regions_mask, 
#                                                  radius_of_influence=50000)
# 
# elev_resample = pyresample.kd_tree.resample_nearest(source_geo_def=orig_def_elev, 
#                                                  target_geo_def=targ_def, 
#                                                  data=ismip_elev, 
#                                                  radius_of_influence=50000)
# 
# mask_resample = pyresample.kd_tree.resample_nearest(source_geo_def=orig_def_elev, 
#                                                  target_geo_def=targ_def, 
#                                                  data=ismip_mask, 
#                                                  radius_of_influence=50000)
# =============================================================================

# Loop over every CloudSat profile
region = []
elevation = []

cs_sw = []
cs_lw = []

era_sw = []
era_lw = []

cs_sw_cs = []
cs_lw_cs = []
era_sw_cs = []
era_lw_cs = []

cs_type = []
era_hcc = []
era_mcc = []
era_lcc = []
filename = []

count = []

diff_sw = []
diff_lw = []

for i in range(len(cs_files)):
    
    print('Matching %.0f of %.0f' %(i+1, len(cs_files)))
    
    # Get path and filename separately
    infilepath1, infilename1 = os.path.split(cs_files[i])
    # Get file name without extension
    infileshortname1, extension1 = os.path.splitext(infilename1)
    
    # Read CloudSat data
    cs = pd.read_csv(cs_files[i], parse_dates=['datetime'])
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(cs['lon'], cs['lat'])]
    gdf = gpd.GeoDataFrame(cs, crs=df_crs, geometry=geometry)

    # Get a list of ERA5 lat/lons to extract data
    closest_era, idx = geopandas_utils.nearest_neighbor(gdf.reset_index(drop=True), 
                                                   era_gdf.reset_index(drop=True), 
                                                   return_dist=True)
    
    # Add index to DataFrame to groupby
    gdf['idx'] = idx
      
    # Get ERA5 corresponding to nearest datetime
    date_match = np.abs(era5_time['datetime'] - cs['datetime'].iloc[0]).argmin()
    era5_hcc = era5.variables['hcc'][date_match, :, :]
    era5_mcc = era5.variables['mcc'][date_match, :, :]
    era5_lcc = era5.variables['mcc'][date_match, :, :]
    era5_strd = era5.variables['strd'][date_match, :, :]
    era5_ssrd = era5.variables['ssrd'][date_match, :, :]
    era5_strdc = era5.variables['strdc'][date_match, :, :]
    era5_ssrdc = era5.variables['ssrdc'][date_match, :, :]

    # Add to DataFrame
    gdf['ssrd'] = np.ravel(era5_ssrd)[idx]
    gdf['strd'] = np.ravel(era5_strd)[idx]
    gdf['strdc'] = np.ravel(era5_strdc)[idx]
    gdf['ssrdc'] = np.ravel(era5_ssrdc)[idx]
    gdf['hcc'] = np.ravel(era5_hcc)[idx]
    gdf['mcc'] = np.ravel(era5_mcc)[idx]
    gdf['lcc'] = np.ravel(era5_lcc)[idx]
    
    # Filter 
    gdf = gdf[gdf['lw_down_as'] < 400]
    gdf = gdf[gdf['lw_down_as'] > 150]
    gdf = gdf[gdf['lw_down_cs'] != 0]
    
    gdf = gdf[gdf['sw_down_as'] != 0]
    gdf = gdf[gdf['sw_down_cs'] != 0]

# =============================================================================
#     # Get difference
#     test_lw = np.abs((gdf['strd'] / 3600) - gdf['lw_down_as'])
#     test_sw = np.abs((gdf['ssrd'] / 3600) - gdf['sw_down_as'])
#     diff_lw.append(gdf['index'][test_lw > 50].values)
#     diff_sw.append(gdf['index'][test_sw > 50].values)
# =============================================================================
    # Add a column for counting
    gdf['number'] = 1    

    # Groupby index
    new_df = gdf[['sw_down_as', 'lw_down_as', 'sw_down_cs', 'lw_down_cs', 
                  'ssrd', 'strd', 'ssrdc', 'strdc',
                  'elev', 'hcc', 'mcc', 'lcc', 'idx']].groupby('idx').mean()
    new_df_groups = gdf[['region', 'cloud_type', 'idx']].groupby('idx').median()
    new_df_count = gdf[['number', 'idx']].groupby('idx').sum()
    
    # Append to list
    region.append(new_df_groups['region'].values)
    elevation.append(new_df['elev'].values)
    cs_sw.append(new_df['sw_down_as'].values)
    cs_lw.append(new_df['lw_down_as'].values)
    cs_sw_cs.append(new_df['sw_down_cs'].values)
    cs_lw_cs.append(new_df['lw_down_cs'].values)
    era_sw.append(new_df['ssrd'].values / 3600)
    era_lw.append(new_df['strd'].values / 3600)    
    era_sw_cs.append(new_df['ssrdc'].values / 3600)
    era_lw_cs.append(new_df['strdc'].values / 3600)
    era_hcc.append(new_df['hcc'].values)
    era_mcc.append(new_df['mcc'].values)
    era_lcc.append(new_df['lcc'].values)
    cs_type.append(new_df_groups['cloud_type'].values)
    filename.append(np.repeat(infileshortname1, new_df.shape[0]))
    count.append(new_df_count['number'].values)

# Flatten
cloud_type_flat = [item for sublist in cs_type for item in sublist]
region_flat = [item for sublist in region for item in sublist]
elev_flat = [item for sublist in elevation for item in sublist]
cs_sw_flat = [item for sublist in cs_sw for item in sublist]
cs_lw_flat = [item for sublist in cs_lw for item in sublist]
cs_sw_cs_flat = [item for sublist in cs_sw_cs for item in sublist]
cs_lw_cs_flat = [item for sublist in cs_lw_cs for item in sublist]
era_sw_flat = [item for sublist in era_sw for item in sublist]
era_lw_flat = [item for sublist in era_lw for item in sublist]
era_sw_cs_flat = [item for sublist in era_sw_cs for item in sublist]
era_lw_cs_flat = [item for sublist in era_lw_cs for item in sublist]
filename_flat = [item for sublist in filename for item in sublist]
hcc_flat = [item for sublist in era_hcc for item in sublist]
mcc_flat = [item for sublist in era_mcc for item in sublist]
lcc_flat = [item for sublist in era_lcc for item in sublist]
count_flat = [item for sublist in count for item in sublist]

# Put back into DataFrame
df = pd.DataFrame(list(zip(cloud_type_flat,region_flat,
                           elev_flat, cs_sw_flat,cs_lw_flat, 
                           cs_sw_cs_flat, cs_lw_cs_flat, 
                           era_sw_flat,era_lw_flat,
                           era_sw_cs_flat, era_lw_cs_flat,
                           hcc_flat, mcc_flat,lcc_flat, count_flat, filename_flat)))

df.columns = ['cs_type', 'region', 'elev', 'cs_sw','cs_lw', 'cs_sw_cs', 'cs_lw_cs',
              'era_sw', 'era_lw','era_sw_cs', 'era_lw_cs', 'hcc','mcc', 'lcc',
              'count', 'filename']

# Drop rows with NaNs
df = df.dropna()

# Save as csv
df.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/CS_vs_ERA5.csv')

# =============================================================================
# # Write list of potentially bad CloudSat data points to csv
# diff_lw_flat = [item for sublist in diff_lw for item in sublist]
# diff_sw_flat = [item for sublist in diff_sw for item in sublist]
# diff_df = pd.DataFrame(list(zip(diff_lw_flat, diff_sw_flat)))
# diff_df.columns = ['diff_lw', 'diff_sw']
# diff_df.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Bad_CS_Values.csv')
# =============================================================================




