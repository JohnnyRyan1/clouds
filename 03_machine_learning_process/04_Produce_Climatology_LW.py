#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Produce summer LW radiative flux climatology

"""

# Import modules
import numpy as np
import glob
import netCDF4

# Define year to process
years = np.arange(2003, 2021, 1)
for year in years:
    
    # Define files
    files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_LW/*' + str(year) + '*.nc'))
            
    # Define destination
    dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/'
    
    # Define ice sheet grid
    ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
    ismip_lon = ismip.variables['lon'][:]
    ismip_lat = ismip.variables['lat'][:]
    
    # Define some empty arrays
    clrsky_lw = np.zeros(ismip_lat.shape)
    allsky_lw = np.zeros(ismip_lat.shape)
    
    for i in range(len(files)):
        print('Processing... %.0f out of %.0f' %(i, len(files)))
    
        # Read netCDF
        f = netCDF4.Dataset(files[i])
            
        # Stack
        clrsky_lw = np.dstack((clrsky_lw, f.variables['clrsky_lw'][:]))
        allsky_lw = np.dstack((allsky_lw, f.variables['allsky_lw'][:]))
    
    # Remove first layer
    clrsky_lw = clrsky_lw[:, :, 1:]
    allsky_lw = allsky_lw[:, :, 1:]
    
    # Average
    clrsky_lw_mean = np.nanmean(clrsky_lw, axis=2)
    allsky_lw_mean = np.nanmean(allsky_lw, axis=2)
    
    ###############################################################################
    # Save 1 km dataset to NetCDF
    ###############################################################################
    dataset = netCDF4.Dataset(dest + 'MYD06_LW_Fluxes_' + str(year) + '.nc', 
                              'w', format='NETCDF4_CLASSIC')
    print('Creating... %s' % dest + 'MYD06_LW_Fluxes_' + str(year) + '.nc')
    dataset.Title = "All-sky fluxes and clear-sky fluxes from MYD06_L2 product"
    import time
    dataset.History = "Created " + time.ctime(time.time())
    dataset.Projection = "WGS 84"
    dataset.Reference = "Ryan, J. C., Smith, L. C., et al. (unpublished)"
    dataset.Contact = "jonathan_ryan@brown.edu"
        
    # Create new dimensions
    lat_dim = dataset.createDimension('y', ismip_lat.shape[0])
    lon_dim = dataset.createDimension('x', ismip_lat.shape[1])
        
    # Define variable types
    Y = dataset.createVariable('latitude', np.float32, ('y','x'))
    X = dataset.createVariable('longitude', np.float32, ('y','x'))
        
    # Define units
    Y.units = "degrees"
    X.units = "degrees"
       
    # Create the actual 3D variable
    clrsky_lw_nc = dataset.createVariable('clrsky_lw', np.float32, ('y','x'))
    allsky_lw_nc = dataset.createVariable('allsky_lw', np.float32, ('y','x'))
               
    # Write data to layers
    Y[:] = ismip_lat
    X[:] = ismip_lon
    clrsky_lw_nc[:] = clrsky_lw_mean
    allsky_lw_nc[:] = allsky_lw_mean
    
    print('Writing data to %s' % dest + 'MYD06_LW_Fluxes_' + str(year) + '.nc')
        
    # Close dataset
    dataset.close()






































