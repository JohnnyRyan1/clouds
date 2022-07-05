#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Aquire MODIS cloud properties.

"""

# Import modules
import numpy as np
import pandas as pd
import glob
import os
import netCDF4
import pyresample
from functions import hdf_read

# Define files
modis_list = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/MYD06_L2/*.hdf'))

# Define destination for predicted data
dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Cloud_Props_NC/'

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]

# Define years
years = np.arange(2003, 2021, 1)

# Define months
months = [152, 182, 213, 244]

# Define good hours
good_hours = ['06', '07', '08', '09', '10', '11', '12', '13', '14']

good_files = []
for file in modis_list:
    # Get path and filename seperately 
    infilepath, infilename = os.path.split(file)
    # Get file name without extension            
    infilehortname, extension = os.path.splitext(infilename)
    
    # Append hour
    hour = infilehortname[18:20]
    if hour in good_hours:
        good_files.append(file)

for year in years:
    for month in range(len(months)-1):
        
        # Get MODIS files
        modis_list_by_years = []
        for j in range(len(good_files)):
            
            # Get path and filename seperately 
            infilepath, infilename = os.path.split(good_files[j]) 
            # Get file name without extension            
            infileshortname, extension = os.path.splitext(infilename)
            
            if (infileshortname[10:14] == str(year)) & (int(infileshortname[14:17]) >= months[month]) & (int(infileshortname[14:17]) <= months[month+1]):
                modis_list_by_years.append(good_files[j])
    
        # Chunk into groups of 75
        bounds = np.arange(0, len(modis_list_by_years), 75)
    
        print('Processing year... %s and month... %.0f' %(str(year), month+1))
        for bound in bounds: 
            
            # Get slice of list
            files_sliced = modis_list_by_years[bound:bound+75]
            
            if os.path.exists(dest + 'MYD06_Cloud_' + str(year) + '_' + str(month+1) + '_' + str(bound) + '_' + str(bound+75)  + '.nc'):
                pass
            else:
                
                data_stacked = np.zeros((2881, 1681))
                for i in range(len(files_sliced)):
                    print('Processing... %.0f out of %.0f' %(i+1, len(files_sliced)))
                    
                    # Read HDF
                    data = hdf_read.MYD06_L2_Read(files_sliced[i], 'Cloud_Optical_Thickness')
                    
                    # Set zeros to NaNs
                    data[data == 0] = np.nan
                    
                    # 2 = CTH, 3 = CTT, 4 = CTP, 5 = PHASE, 6 = COT, 7 = CER, 8 = CWP
                    if np.nansum(np.isfinite(data[:,:,2])) > 200000:
                        # Resample radiative fluxes to ISMIP grid
                        swath_def = pyresample.geometry.SwathDefinition(lons=data[:,:,1], lats=data[:,:,0])
                        swath_con = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
                        
    
                        # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
                        data_resampled = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                                         target_geo_def=swath_con, 
                                                                         data=data[:,:,2],
                                                                         radius_of_influence=5000)
                    
                        # Set zeros to NaNs
                        data_resampled[data_resampled == 0] = np.nan
                        
                        # Stack
                        data_stacked = np.dstack((data_stacked, data_resampled))
                
                # Remove first layer
                data_stacked = data_stacked[:, :, 1:]
                
                # Average
                data_mean = np.nanmean(data_stacked, axis=2)

