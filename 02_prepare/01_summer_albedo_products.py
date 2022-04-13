#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Produce monthly albedo products from MOD10A1 data.

"""

# Import modules
import numpy as np
import netCDF4
from pyhdf.SD import SD, SDC
import os
import glob
import itertools
from pyproj import Proj

# =============================================================================
# # Edit the text file
# file_list = pd.read_csv('/home/johnny/Documents/Clouds/Scripts/Download/mod10a1_2018-2020b.txt',
#                         header=None)
# new_file_list = []
# for i in range(file_list.shape[0]):
#     if (int(file_list.iloc[i].values[0][76:79]) > 151) & (int(file_list.iloc[i].values[0][76:79]) < 244):
#         new_file_list.append(file_list.iloc[i].values[0])
# 
# new_new_file_list = []
# for j in range(len(new_file_list)):
#     if new_file_list[j][81:86] == '15v01':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '15v02':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '16v00':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '16v00':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '16v01':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '16v02':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '17v00':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '17v01':
#         new_new_file_list.append(new_file_list[j])
#     if new_file_list[j][81:86] == '17v02':
#         new_new_file_list.append(new_file_list[j])
#     else:
#         pass
# 
# files = pd.DataFrame(new_new_file_list)
# files.to_csv('/home/johnny/Documents/Clouds/Scripts/Download/mod10a1_2018-2020_Filtered.csv')
# =============================================================================

# Define destination to save
dest = '/media/johnny/Cooley_Data/Johnny/Clouds/MOD10A1_Monthly_Means/'

# Define location of MODIS data
modis_files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds/MOD10A1/*.hdf'))
print('Number of files in folder = %.0f' %len(modis_files))

# Get number of unique tiles
modis_tiles = []
for k, v in itertools.groupby(modis_files, key=lambda x: x[66:71]):
    modis_tiles.append(k)

#tiles = np.unique(np.array(modis_tiles))
tiles = np.array(['15v01', '15v02', '16v00', '16v01', '16v02', '17v00', '17v01', '17v02'])
years = np.arange(2003, 2021, 1)

def save2netcdf(lats, lons, data1, tile, year):
    dataset = netCDF4.Dataset(dest + 'MOD10A1_Albedo_' + tile + '_' + year + '.nc', 
                              'w', format='NETCDF4_CLASSIC')
    print('Creating %s' %dest + 'MOD10A1_Albedo_' + tile + '_' + year + '.nc')
    dataset.Title = "Snow albedo for tile %s for %s from MOD10A1 product" %(tile, year)
    import time
    dataset.History = "Created " + time.ctime(time.time())
    dataset.Projection = "WGS 84"
    dataset.Reference = "Ryan, J. C., Smith, L. C., et al. (unpublished)"
    dataset.Contact = "jonathan_ryan@brown.edu"
        
    # Create new dimensions
    lat_dim = dataset.createDimension('y', lon.shape[0])
    lon_dim = dataset.createDimension('x', lat.shape[1])
    
    # Define variable types
    Y = dataset.createVariable('latitude', np.float64, ('y','x'))
    X = dataset.createVariable('longitude', np.float64, ('y','x'))
        
    # Define units
    Y.units = "degrees"
    X.units = "degrees"
       
    # Create the actual 3D variable
    albedo = dataset.createVariable('albedo', np.int8, ('y','x'))
    
    albedo.units = "unitless"
    
    # Write data to layers
    Y[:] = lats
    X[:] = lons
    albedo[:] = data1

    print('Writing data to %s' %dest + 'MOD10A1_Albedo_' + tile + '_' + year + '.nc')
        
    # Close dataset
    dataset.close()
    
# Loop through tiles
for i in tiles:
    # Get MODIS files
    modis_list = []
    for j in range(len(modis_files)):
        if  modis_files[j][66:71] == i:
            modis_list.append(modis_files[j])
    
    print('Processing tile... %s' %i)

    # Define lat and lons of tile
    grid_cell = 463.31271653
    upper_left_grid = (-20015109.354, 10007554.677)
    lower_right_grid = (20015109.354, -10007554.677)
    
    upper_left_corner = (upper_left_grid[0] + (2400*int(i[0:2])*grid_cell),
                         upper_left_grid[1] - (2400*int(i[3:])*grid_cell))
    lower_right_corner = (upper_left_corner[0] + (2400*grid_cell),
                         (upper_left_corner[1] - (2400*grid_cell)))
    
    x = np.linspace(upper_left_corner[0], lower_right_corner[0], 2400)
    y = np.linspace(upper_left_corner[1], lower_right_corner[1], 2400)
    
    # Produce grid
    xv, yv = np.meshgrid(x, y)
    
    # Define MODIS grid in pyproj
    modis_grid = Proj('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext')
    
    # Convert to lat, lons
    lon, lat = modis_grid(xv, yv, inverse=True)
    
    for p in years:
        # Get MODIS files
        modis_list_by_years = []
        for j in range(len(modis_list)):
            if  modis_list[j][57:61] == str(p):
                modis_list_by_years.append(modis_list[j])
                
        new_file_list = []
        for j in range(len(modis_list_by_years)):
            if (int(modis_list_by_years[j][61:64]) > 151) & (int(modis_list_by_years[j][61:64]) < 244):
                new_file_list.append(modis_list_by_years[j])
        
        #print('Processing year... %s' %str(p))
        print('Number of files in %s = %.0f' %(str(p), len(new_file_list)))
        if os.path.exists(dest + 'MOD10A1_Albedo_' + i + '_' + str(p) + '.nc'):
            print('Skipping... %s' %(dest + 'MOD10A1_Albedo_' + i + '_' + str(p) + '.nc'))
        else:
            # Define empty arrays for data
            snow_albedo_grid = []
                 
            for h in range(len(new_file_list)):
                # Get path and filename seperately 
                infilepath, infilename = os.path.split(new_file_list[h]) 
                # Get file name without extension            
                infileshortname, extension = os.path.splitext(infilename)
            
                # Read MODIS file
                f = SD(new_file_list[h], SDC.READ)
                
                # Get datasets
                sds_snow_albedo = f.select('Snow_Albedo_Daily_Tile')
                snow_albedo = sds_snow_albedo.get()
                
                # Stack to empty array
                snow_albedo_grid.append(snow_albedo)
            
            # Count good values for each pixel for land mask
            data = np.dstack(snow_albedo_grid).astype(float)
            data[data > 100] = np.nan
            data[data > 83] = 84
            data[data < 0] = np.nan
            data[data < 30] = 30
            data_mean = np.nanmean(data, axis=2)
    
            # Concatenate array and save to netcdf
            save2netcdf(lat, lon, np.nanmean(data, axis=2), i, str(p))

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        