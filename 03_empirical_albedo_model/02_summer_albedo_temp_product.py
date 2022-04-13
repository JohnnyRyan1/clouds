#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Get mean summer stats for regions.

"""

# Import modules
import numpy as np
import glob
import pandas as pd
import pyresample
import netCDF4
import xarray as xr
from pyproj import Transformer
from datetime import timedelta, datetime

# Define path name
path = '/Users/jryan4/Dropbox (University of Oregon)/projects/clouds/data/'

# Define coordinate system
df_crs = 'EPSG:4326'

# Define MODIS files
modis_files = sorted(glob.glob(path + 'mod10a1_monthly_means/*.nc'))

# Define downscaled ERA5 data
era_files = sorted(glob.glob(path + 'era_downscaled/*.asc'))

# Define ice sheet mask and elevations
ismip_1km = netCDF4.Dataset(path + 'masks/1km-ISMIP6.nc')
ismip_lon = ismip_1km.variables['lon'][:]
ismip_lat = ismip_1km.variables['lat'][:]
ismip_mask = ismip_1km.variables['ICE'][:]
ismip_elev = ismip_1km.variables['SRF'][:]

# Define region mask
regions = netCDF4.Dataset(path + 'masks/Ultimate_Mask.nc')
regions_y = regions.variables['y'][:]
regions_x = regions.variables['x'][:]
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

# Convert from region mask stereographic to WGS84
transformer = Transformer.from_crs(3413, 4326)
region_lat, region_lon = transformer.transform(regions_x, regions_y)

# Compute summer means
years = np.arange(2003, 2021, 1)

def read_asc_file(file_path, verbose=True):
    """
    Read in a file in ESRI ASCII raster format, 
    which consists of a header describing the grid followed by 
    values on the grid.
    For more information see:
        http://resources.esri.com/help/9.3/arcgisengine/java/GP_ToolRef/spatial_analyst_tools/esri_ascii_raster_format.htm
    """

    import numpy as np
    asc_file = open(file_path, 'r')
    
    tokens = asc_file.readline().split()
    ncols = int(tokens[1])
    
    tokens = asc_file.readline().split()
    nrows = int(tokens[1])
    
    tokens = asc_file.readline().split()
    xllcorner = float(tokens[1])
    
    tokens = asc_file.readline().split()
    yllcorner = float(tokens[1])
    
    tokens = asc_file.readline().split()
    cellsize = float(tokens[1])
    
    tokens = asc_file.readline().split()
    nodata_value = float(tokens[1])
    
    if verbose:
        print("ncols = %i" % ncols)
        print("nrows = %i" % nrows)
        print("xllcorner = %g" % xllcorner)
        print("yllcorner = %g" % yllcorner)
        print("cellsize = %g" % cellsize)
        print("nodata_value = %g" % nodata_value)
        
    # read in all the data, assumed to be on ncols lines, 
    # each containing nrows values
    
    asc_file.close()  # close file so we can load array
    asc_data = np.loadtxt(file_path, skiprows=6)  # skip header
    
    # reshape
    values = asc_data.reshape((nrows,ncols))
    
    # flip in y because of data order
    values = np.flipud(values)    
    
    x = xllcorner + cellsize * np.arange(0,ncols)
    y = yllcorner + cellsize * np.arange(0,nrows)
    
    X,Y = np.meshgrid(x,y)
        
    asc_data_dict = {'ncols': ncols, 'nrows': nrows, 'xllcorner':xllcorner, \
                     'yllcorner':yllcorner, 'cellsize':cellsize, \
                     'nodata_value':nodata_value, \
                     'X': X, 'Y':Y, 'values': values}
    return asc_data_dict

# Define some empty arrays
t2m_values = np.zeros(ismip_lon.shape)

for i in range(len(era_files)):
    
    # Open data
    t2m_data = read_asc_file(era_files[i])
    
    # Append
    t2m_values = np.dstack((t2m_values, t2m_data['values']))
    
# Remove first layer
t2m_values = t2m_values[:, :, 1:]

# Regrid to ISMIP and add elevation and regions attributes
orig_def_regions = pyresample.geometry.GridDefinition(lons=region_lon, lats=region_lat)
targ_def = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)

# Determine nearest (w.r.t. great circle distance) neighbour in the grid.
region_resample = pyresample.kd_tree.resample_nearest(source_geo_def=orig_def_regions, 
                                                 target_geo_def=targ_def, 
                                                 data=regions_mask, 
                                                 radius_of_influence=50000)

###############################################################################
# Summer albedo values
###############################################################################

print('Number of files in folder = %.0f' %len(modis_files))

# Define years
years = np.arange(2003, 2021, 1)
albedo_values = np.zeros(ismip_lat.shape)

for i in years:
    modis_list = []   
    # Get MODIS tiles
    for f in modis_files:
        if  f[-7:-3]  == str(i):
            modis_list.append(f)
    
    # Define new master grid
    master_grid_albedo = np.zeros((7200, 7200), dtype='int16')
    master_grid_lat = np.zeros((7200, 7200), dtype='float')
    master_grid_lon = np.zeros((7200, 7200), dtype='float')

    # Add tile to master grid
    for j in modis_list:
        if j[-13:-8] == '15v00':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[0:2400,0:2400] = modis.variables['albedo'][:]
            master_grid_lat[0:2400,0:2400] = modis.variables['latitude'][:]
            master_grid_lon[0:2400,0:2400] = modis.variables['longitude'][:]
        if j[-13:-8] == '16v00':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[0:2400,2400:4800] = modis.variables['albedo'][:]
            master_grid_lat[0:2400,2400:4800] = modis.variables['latitude'][:]
            master_grid_lon[0:2400,2400:4800] = modis.variables['longitude'][:]
        if j[-13:-8] == '17v00':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[0:2400,4800:7200] = modis.variables['albedo'][:]
            master_grid_lat[0:2400,4800:7200] = modis.variables['latitude'][:]
            master_grid_lon[0:2400,4800:7200] = modis.variables['longitude'][:]
        if j[-13:-8] == '15v01':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[2400:4800,0:2400] = modis.variables['albedo'][:]
            master_grid_lat[2400:4800,0:2400] = modis.variables['latitude'][:]
            master_grid_lon[2400:4800,0:2400] = modis.variables['longitude'][:]
        if j[-13:-8] == '16v01':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[2400:4800,2400:4800] = modis.variables['albedo'][:]
            master_grid_lat[2400:4800,2400:4800] = modis.variables['latitude'][:]
            master_grid_lon[2400:4800,2400:4800] = modis.variables['longitude'][:]
        if j[-13:-8] == '17v01':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[2400:4800,4800:7200] = modis.variables['albedo'][:]
            master_grid_lat[2400:4800,4800:7200] = modis.variables['latitude'][:]
            master_grid_lon[2400:4800,4800:7200] = modis.variables['longitude'][:]
        if j[-13:-8] == '15v02':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[4800:7200,0:2400] = modis.variables['albedo'][:]
            master_grid_lat[4800:7200,0:2400] = modis.variables['latitude'][:]
            master_grid_lon[4800:7200,0:2400] = modis.variables['longitude'][:]
        if j[-13:-8] == '16v02':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[4800:7200,2400:4800] = modis.variables['albedo'][:]
            master_grid_lat[4800:7200,2400:4800] = modis.variables['latitude'][:]
            master_grid_lon[4800:7200,2400:4800] = modis.variables['longitude'][:]
        if j[-13:-8] == '17v02':
            modis = netCDF4.Dataset(j, 'r')
            master_grid_albedo[4800:7200,4800:7200] = modis.variables['albedo'][:]
            master_grid_lat[4800:7200,4800:7200] = modis.variables['latitude'][:]
            master_grid_lon[4800:7200,4800:7200] = modis.variables['longitude'][:]
        else:
            pass

    # Get ISMIP6 lat lons
    lon_1km = ismip_1km.variables['lon'][:]
    lat_1km = ismip_1km.variables['lat'][:]
    
    # Convert 0s to NaNs so they do not interfere with resampling
    master_grid_albedo = master_grid_albedo.astype('float')
    
    # Define regridding conversion
    swath_def = pyresample.geometry.SwathDefinition(lons=lon_1km, lats=lat_1km)
    swath_con = pyresample.geometry.SwathDefinition(lons=master_grid_lon, lats=master_grid_lat)
    albedo_con = pyresample.image.ImageContainer(master_grid_albedo, swath_con)
    row_indices, col_indices = pyresample.utils.generate_nearest_neighbour_linesample_arrays(swath_con, swath_def, 1000)
    
    # Perform regridding
    albedo_result = albedo_con.get_array_from_linesample(row_indices, col_indices)
    
    # Convert zeros to NaNs
    albedo_result[albedo_result == 0] = np.nan
    
    # Stack
    albedo_values = np.dstack((albedo_values, albedo_result))

# Remove first layer
albedo_values = albedo_values[:, :, 1:]


###############################################################################
# Save 1 km dataset to NetCDF
###############################################################################
dataset = netCDF4.Dataset(path + 'temp_albedo_summer_climatologies.nc', 
                          'w', format='NETCDF4_CLASSIC')
print('Creating... %s' % path + 'temp_albedo_summer_climatologies.nc')
dataset.Title = "Mean summer 2 m air temperature and albedo"
import time
dataset.History = "Created " + time.ctime(time.time())
dataset.Projection = "WGS 84"
dataset.Reference = "Ryan, J. C., Smith. L. C., Cooley, S. W., and Pearson, B. (in review), Emerging importance of clouds for Greenland Ice Sheet energy balance and meltwater production."
dataset.Contact = "jryan4@uoregon.edu"
    
# Create new dimensions
lat_dim = dataset.createDimension('y', ismip_lat.shape[0])
lon_dim = dataset.createDimension('x', ismip_lat.shape[1])
data_dim = dataset.createDimension('z', t2m_values.shape[2])

# Define variable types
Y = dataset.createVariable('latitude', np.float32, ('y','x'))
X = dataset.createVariable('longitude', np.float32, ('y','x'))
    
# Define units
Y.units = "degrees"
X.units = "degrees"
   
# Create the actual 3D variable
t2m_nc = dataset.createVariable('t2m', np.float32, ('y','x','z'))
regions_nc = dataset.createVariable('regions', np.float32, ('y','x'))
mask_nc = dataset.createVariable('mask', np.float32, ('y','x'))
albedo_nc = dataset.createVariable('albedo', np.float32, ('y','x','z'))
           
# Write data to layers
Y[:] = ismip_lat
X[:] = ismip_lon
t2m_nc[:] = t2m_values
regions_nc[:] = region_resample
mask_nc[:] = ismip_mask
albedo_nc[:] = albedo_values

print('Writing data to %s' % path + 'temp_albedo_summer_climatologies.nc')
    
# Close dataset
dataset.close()
































