#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Regrid and stack LW radiative flux data to ISMIP grid.

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
import netCDF4
import pyresample

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]

dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_LW_Blocking/'

# Define blocking events
block_list = pd.read_csv('/home/johnny/Documents/Clouds/Data/Blocking/Ward_Blocking_Events.csv', parse_dates=['Date'])
block_list['year'] = block_list['Date'].dt.year
block_list['julian_day'] = block_list['Date'].dt.dayofyear
block_list['date_str'] = block_list['year'].astype(str) + block_list['julian_day'].astype(str)

# Define region
regions = ['SW', 'SE', 'NW', 'NE']

# Define number of files to process at a time
number = 50

for region in regions:
    
    # Define good hours
    good_hours = ['06', '07', '08', '09', '10', '11', '12', '13', '14']
    
    blocks = list(block_list[block_list['Quadrant'] == region]['date_str'])
    
    # Define files
    files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_LW/*.csv'))
    
    # Filter only blocking events
    block_files = []
    for j in files:
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(j)
        # Get file name without extension            
        infilehortname, extension = os.path.splitext(infilename)
        
        # Append hour
        date_str = infilehortname[10:17]
        if date_str in blocks:
            block_files.append(j)
            
    good_files = []
    for j in block_files:
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(j)
        # Get file name without extension            
        infilehortname, extension = os.path.splitext(infilename)
        
        # Append hour
        hour = infilehortname[18:20]
        if hour in good_hours:
            good_files.append(j)
         
    # Chunk into groups
    bounds = np.arange(0, len(good_files), number)
    
    for bound in bounds: 
        
        # Get slice of list
        files_sliced = good_files[bound:bound+number]
        
        if os.path.exists(dest + 'MYD06_Fluxes_' + str(bound) + '_' + str(bound+number)  + '_' + region + '_Block.nc'):
            pass
        else:
        
            clrsky_lw = np.zeros(ismip_lat.shape)
            allsky_lw = np.zeros(ismip_lat.shape)
        
            for i in range(len(files_sliced)):
                print('Processing... %.0f out of %.0f' %(i, len(files_sliced)))
                                   
                # Read csv
                df = pd.read_csv(files_sliced[i], error_bad_lines=False)
                
                if df.shape[0] > 200000:
                
                    # Resample radiative fluxes to ISMIP grid
                    swath_def = pyresample.geometry.SwathDefinition(lons=df['lon'].values, lats=df['lat'].values)
                    swath_con = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
                    
                
                    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
                    clearsky_lw_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                                     target_geo_def=swath_con, 
                                                                     data=df['clearsky_lw'].values, 
                                                                     radius_of_influence=5000)
                    allsky_lw_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                                     target_geo_def=swath_con, 
                                                                     data=df['allsky_lw'].values, 
                                                                     radius_of_influence=5000)
                    
                    # Set zeros to NaNs
                    allsky_lw_data[allsky_lw_data == 0] = np.nan
                    clearsky_lw_data[clearsky_lw_data == 0] = np.nan
                    
                    # Stack
                    clrsky_lw = np.dstack((clrsky_lw, clearsky_lw_data))
                    allsky_lw = np.dstack((allsky_lw, allsky_lw_data))
                    
                else:
                    pass
                
            # Remove first layer
            clrsky_lw = clrsky_lw[:, :, 1:]
            allsky_lw = allsky_lw[:, :, 1:]
            
            # Average
            clrsky_lw_mean = np.nanmean(clrsky_lw, axis=2)
            allsky_lw_mean = np.nanmean(allsky_lw, axis=2)
            
            ###############################################################################
            # Save 1 km dataset to NetCDF
            ###############################################################################
            dataset = netCDF4.Dataset(dest + 'MYD06_Fluxes_' + str(bound) + '_' + str(bound+number)  + '_' + region + '_Block.nc', 
                                      'w', format='NETCDF4_CLASSIC')
            print('Creating... %s' % dest + 'MYD06_Fluxes_' + str(bound) + '_' + str(bound+number)  + '_' + region + '_Block.nc')
            dataset.Title = "All-sky fluxes and clear-sky fluxes from MYD06_L2 product for blocking events."
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
            
            y = dataset.createVariable('y', np.float32, ('y'))
            x = dataset.createVariable('x', np.float32, ('x'))
        
                
            # Define units
            Y.units = "degrees"
            X.units = "degrees"
               
            # Create the actual 3D variable
            clrsky_lw_nc = dataset.createVariable('clrsky_lw', np.float32, ('y','x'))
            allsky_lw_nc = dataset.createVariable('allsky_lw', np.float32, ('y','x'))
                       
            # Write data to layers
            Y[:] = ismip_lat
            X[:] = ismip_lon
            x[:] = ismip_lon[0,:]
            y[:] = ismip_lat[:,0]
            clrsky_lw_nc[:] = clrsky_lw_mean
            allsky_lw_nc[:] = allsky_lw_mean
            
            print('Writing data to %s' % dest + 'MYD06_Fluxes_' + str(bound) + '_' + str(bound+number)  + '_' + region + '_Block.nc')
                
            # Close dataset
            dataset.close()
    
