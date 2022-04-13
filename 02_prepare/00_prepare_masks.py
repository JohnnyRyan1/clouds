#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Convert masks to correct grid size.

"""

# Import modules
import numpy as np
import netCDF4
import pyresample
from pyproj import Transformer

# Define destination to save
dest = '/media/johnny/Cooley_Data/Johnny/Clouds/Data/Masks/'

# Define ice sheet grid
ismip = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Define region mask
regions = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds/Data/Masks/Ultimate_Mask.nc')
regions_lon = regions.variables['x'][:]
regions_lat = regions.variables['y'][:]
regions_1 = regions.variables['North'][:]
regions_2 = regions.variables['NorthEast'][:]
regions_3 = regions.variables['East'][:]
regions_4 = regions.variables['SouthEast'][:]
regions_5 = regions.variables['South'][:]
regions_6 = regions.variables['SouthWest'][:]
regions_7 = regions.variables['West'][:]
regions_8 = regions.variables['NorthWest'][:]

regions_2[regions_2 > 0] = 2
regions_3[regions_3 > 0] = 3
regions_4[regions_4 > 0] = 4
regions_5[regions_5 > 0] = 5
regions_6[regions_6 > 0] = 6
regions_7[regions_7 > 0] = 7
regions_8[regions_8 > 0] = 8

regions_mask = regions_1 + regions_2 + regions_3 + regions_4 + regions_5 + regions_6 +\
    regions_7 + regions_8
    
# Convert from stereographic to WGS84
transformer = Transformer.from_crs(3413, 4326)
lat, lon = transformer.transform(regions_lon, regions_lat)

# Define grid using pyresample       
grid = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
swath = pyresample.geometry.GridDefinition(lons=lon, lats=lat)

# Determine nearest (w.r.t. great circle distance) neighbour in the grid.
region_swath = pyresample.kd_tree.resample_nearest(source_geo_def=swath, 
                                             target_geo_def=grid, 
                                             data=regions_mask, 
                                             radius_of_influence=50000)


###############################################################################
# Save 1 km dataset to NetCDF
###############################################################################
dataset = netCDF4.Dataset(dest + 'Region_Mask.nc', 
                          'w', format='NETCDF4_CLASSIC')
print('Creating... %s' % dest + 'Region_Mask.nc')
dataset.Title = "Mask denoting regions of ice sheet."
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
   
# Create the actual 2D variable
mask_nc = dataset.createVariable('mask', np.int8, ('y','x'))

# Write data to layers
Y[:] = ismip_lat
X[:] = ismip_lon
mask_nc[:] = region_swath

print('Writing data to %s' % dest + 'Region_Mask.nc')
    
# Close dataset
dataset.close()










