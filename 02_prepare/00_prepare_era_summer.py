#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Prepare grids for downscaling.

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
import netCDF4
from datetime import timedelta, datetime

# Define filepath
era_filepath = '/Volumes/Seagate Backup Plus Drive/Clouds/Data/ERA5/era_monthly_2003_2020.nc'

# Open dataset
era = netCDF4.Dataset(era_filepath)

# Read data
era_lon = era.variables['longitude'][:]
era_lat = era.variables['latitude'][:]
era_xx, era_yy = np.meshgrid(era_lon, era_lat)

# Get time
base = datetime(1900,1,1)
era_time = pd.DataFrame(era5.variables['time'][:], columns=['hours'])
era_time['datetime'] = era5_time['hours'].apply(lambda x: base + timedelta(hours=x))
era_time['year'] = era_time['datetime'].dt.year

# Define some empty arrays
t2m_values = np.zeros(era_xx.shape)

# Compute summer means
years = np.arange(2003, 2021, 1)

for i in years:
    # Get years
    idx = era_time.index[era_time['year'] == i].values
    
    # Get data
    t2m = era.variables['t2m'][idx, :, :]
    
    # Append mean
    t2m_values = np.dstack((t2m_values, np.nanmean(t2m, axis=0)))
    
# Remove first layer
t2m_values = t2m_values[:, :, 1:]

    
# Define destination to save
dest = '/Users/jryan4/Dropbox (University of Oregon)/Clouds/Data/For_Downscaling/'
    
# Save as NetCDF4
dataset = netCDF4.Dataset(dest + 'era5_summer_climatologies.nc', 'w', format='NETCDF4_CLASSIC')

print('Creating... %s' % dest + 'era5_summer_climatologies.nc')
dataset.Title = "Mean summer 2 m air temperature from ERA5"

import time
dataset.History = "Created " + time.ctime(time.time())
dataset.Projection = "WGS 84"
dataset.Reference = "Ryan, J. C., Smith. L. C., Cooley, S. W., and Pearson, B. (in review), Emerging importance of clouds for Greenland Ice Sheet energy balance and meltwater production."
dataset.Contact = "jryan4@uoregon.edu"

# Create new dimensions
lat_dim = dataset.createDimension('y', era_xx.shape[0])
lon_dim = dataset.createDimension('x', era_xx.shape[1])
data_dim = dataset.createDimension('z', t2m_values.shape[2])

# Define variable types
Y = dataset.createVariable('latitude', np.float32, ('y','x'))
X = dataset.createVariable('longitude', np.float32, ('y','x'))

# Define units
Y.units = "degrees"
X.units = "degrees"

# Create the actual 3D variable
t2m_nc = dataset.createVariable('t2m', np.float32, ('y','x','z'))

# Write data to layers
Y[:] = era_yy
X[:] = era_xx
t2m_nc[:] = t2m_values

print('Writing data to %s' % dest + 'era5_summer_climatologies.nc')

# Close dataset
dataset.close()


# Read to test
data = netCDF4.Dataset(dest + 'era5_summer_climatologies.nc')
