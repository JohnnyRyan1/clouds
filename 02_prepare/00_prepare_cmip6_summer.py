#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Get mean summer stats from CMIP6 data.

"""

# Import modules
import numpy as np
import glob
import netCDF4
import matplotlib.pyplot as plt
import pathlib

# Define path name
path = '/Users/jryan4/Dropbox (University of Oregon)/research/clouds/data/'

# Get a list of all models
models = sorted(glob.glob(path + 'cmip6-monthly/*/*.nc'))

for model in models:
    
    # Read data
    f = netCDF4.Dataset(model)

    # Define years
    years = np.arange(0, 1033, 12)

    summer_t = np.zeros((f.variables['lat'][:].shape[0], f.variables['lon'][:].shape[0]))
    
    # Convert from monthly to summer
    for i in range(len(years) - 1):

        # Find year
        temp_m = f.variables['tas'][years[i]:years[i+1],:,:]

        # Add summer to grid 
        summer_t = np.dstack((summer_t, np.mean(temp_m[5:8], axis=0)))

    # Remove first axis
    summer_t = summer_t[:,:,1:]

    # Get some file names
    sp = '/Users/jryan4/Dropbox (University of Oregon)/research/clouds/data/cmip6-summer/'
    p = pathlib.Path(model)
    folder = p.parent.name
    savepath = sp + folder + '/' + p.name

    ###############################################################################
    # Save 1 km dataset to NetCDF
    ###############################################################################
    dataset = netCDF4.Dataset(savepath, 'w', format='NETCDF4_CLASSIC')
    print('Creating... %s' % savepath)
    dataset.Title = "Mean summer 2 m air temperature"
    import time
    dataset.History = "Created " + time.ctime(time.time())
    dataset.Projection = "WGS 84"
    dataset.Reference = "Ryan, J. C., Smith. L. C., Cooley, S. W., and Pearson, B. (in review), Emerging importance of clouds for Greenland Ice Sheet energy balance and meltwater production."
    dataset.Contact = "jryan4@uoregon.edu"

    # Create new dimensions
    lat_dim = dataset.createDimension('y', summer_t.shape[0])
    lon_dim = dataset.createDimension('x', summer_t.shape[1])
    data_dim = dataset.createDimension('z', summer_t.shape[2])

    # Define variable types
    Y = dataset.createVariable('latitude', np.float32, ('y'))
    X = dataset.createVariable('longitude', np.float32, ('x'))

    # Define units
    Y.units = "degrees"
    X.units = "degrees"

    # Create the actual 3D variable
    t2m_nc = dataset.createVariable('t2m', np.float32, ('y','x','z'))

    # Write data to layers
    Y[:] = f.variables['lat'][:]
    X[:] = (f.variables['lon'][:] + 180) % 360 - 180
    t2m_nc[:] = summer_t

    print('Writing data to %s' % savepath)

    # Close dataset
    dataset.close()