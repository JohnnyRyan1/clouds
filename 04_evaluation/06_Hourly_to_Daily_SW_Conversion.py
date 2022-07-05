#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Compute hourly to daily correction for SW from ERA5.

"""

# Import modules
import numpy as np
import pandas as pd
import netCDF4
from datetime import timedelta, datetime
import pyresample

# Define destination
dest = '/home/johnny/Documents/Clouds/Data/ERA5/'

# Import ERA5 data
era = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_ssrd_2006_2016.nc')
era_lon = era.variables['longitude'][:]
era_lat = era.variables['latitude'][:]
era_xx, era_yy = np.meshgrid(era_lon, era_lat)

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]

# Get time
base = datetime(1900,1,1)
era_time = pd.DataFrame(era.variables['time'][:], columns=['hours'])
era_time['datetime'] = era_time['hours'].apply(lambda x: base + timedelta(hours=x))
era_time['hour'] = era_time['datetime'].dt.hour

# Get daily average
clrsky_daily_mean = np.nanmean(era.variables['ssrdc'][:], axis=0) / 3600
allsky_daily_mean = np.nanmean(era.variables['ssrd'][:], axis=0) / 3600

# Get hourly average
unique_hours = np.unique(era_time['hour'].values)

allsky_hourly = np.zeros(era_xx.shape)
clrsky_hourly = np.zeros(era_xx.shape)
for i in unique_hours:
    
    # Get index of hour
    idx = era_time.loc[era_time['hour'] == i].index
    
    # Sample ERA5 data
    ssrdc = np.nanmean(era.variables['ssrdc'][idx, :, :], axis=0) / 3600
    ssrd = np.nanmean(era.variables['ssrd'][idx, :, :], axis=0) / 3600
      
    # Stack
    allsky_hourly = np.dstack((allsky_hourly, ssrd))
    clrsky_hourly = np.dstack((clrsky_hourly, ssrdc))

# Remove first layer
allsky_hourly = allsky_hourly[:, :, 1:]
clrsky_hourly = clrsky_hourly[:, :, 1:]

# Regrid to ISMIP grid
swath_def = pyresample.geometry.SwathDefinition(lons=era_xx, lats=era_yy)
swath_con = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)

# Determine nearest (w.r.t. great circle distance) neighbour in the grid.
allsky_hourly_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                 target_geo_def=swath_con, 
                                                 data=allsky_hourly, 
                                                 radius_of_influence=50000)
allsky_daily_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                 target_geo_def=swath_con, 
                                                 data=allsky_daily_mean, 
                                                 radius_of_influence=50000)
clrsky_hourly_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                 target_geo_def=swath_con, 
                                                 data=clrsky_hourly, 
                                                 radius_of_influence=50000)
clrsky_daily_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                 target_geo_def=swath_con, 
                                                 data=clrsky_daily_mean, 
                                                 radius_of_influence=50000)
# Set some values to NaN
allsky_daily_data[allsky_daily_data == 0] = np.nan
allsky_hourly_data[allsky_hourly_data == 0] = np.nan
clrsky_daily_data[clrsky_daily_data == 0] = np.nan
clrsky_hourly_data[clrsky_hourly_data == 0] = np.nan

# Make final calculation
allsky_hourly_multiply = allsky_daily_data[..., np.newaxis] /  allsky_hourly_data 
clrsky_hourly_multiply = clrsky_daily_data[..., np.newaxis] / clrsky_hourly_data

allsky_hourly_add = allsky_daily_data[..., np.newaxis] -  allsky_hourly_data 
clrsky_hourly_add = clrsky_daily_data[..., np.newaxis] - clrsky_hourly_data

###############################################################################
# Save 1 km dataset to NetCDF
###############################################################################
dataset = netCDF4.Dataset(dest + 'ERA5_SW_Conversions.nc', 
                          'w', format='NETCDF4_CLASSIC')
print('Creating... %s' % dest + 'ERA5_SW_Conversions.nc')
dataset.Title = "Conversions from hourly to daily from ERA5"
import time
dataset.History = "Created " + time.ctime(time.time())
dataset.Projection = "WGS 84"
dataset.Reference = "Ryan, J. C., Smith, L. C., et al. (unpublished)"
dataset.Contact = "jonathan_ryan@brown.edu"
    
# Create new dimensions
lat_dim = dataset.createDimension('y', ismip_lat.shape[0])
lon_dim = dataset.createDimension('x', ismip_lat.shape[1])
data_dim = dataset.createDimension('z', 24)
    
# Define variable types
Y = dataset.createVariable('latitude', np.float32, ('y','x'))
X = dataset.createVariable('longitude', np.float32, ('y','x'))

y = dataset.createVariable('y', np.float32, ('y'))
x = dataset.createVariable('x', np.float32, ('x'))

# Define units
Y.units = "degrees"
X.units = "degrees"
   
# Create the actual 3D variable
correction_allsky_multiply_nc = dataset.createVariable('correction_allsky_multiply', np.float32, ('y','x','z'))
correction_clrsky_multiply_nc = dataset.createVariable('correction_clrsky_multiply', np.float32, ('y','x','z'))
correction_allsky_add_nc = dataset.createVariable('correction_allsky_add', np.float32, ('y','x','z'))
correction_clrsky_add_nc = dataset.createVariable('correction_clrsky_add', np.float32, ('y','x','z'))

# Write data to layers
Y[:] = ismip_lat
X[:] = ismip_lon
correction_allsky_multiply_nc[:] = allsky_hourly_multiply
correction_clrsky_multiply_nc[:] = clrsky_hourly_multiply

correction_allsky_add_nc[:] = allsky_hourly_add
correction_clrsky_add_nc[:] = clrsky_hourly_add

print('Writing data to %s' % dest + 'ERA5_SW_Conversions.nc')
    
# Close dataset
dataset.close()













