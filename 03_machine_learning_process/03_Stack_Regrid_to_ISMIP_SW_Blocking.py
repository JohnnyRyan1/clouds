#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Regrid and stack radiative flux data *during blocking events* to ISMIP grid.

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
import netCDF4
import pyresample

# Define blocking events
block_list = pd.read_csv('/home/johnny/Documents/Clouds/Data/Blocking/Ward_Blocking_Events.csv', parse_dates=['Date'])
block_list['year'] = block_list['Date'].dt.year
block_list['julian_day'] = block_list['Date'].dt.dayofyear
block_list['date_str'] = block_list['year'].astype(str) + block_list['julian_day'].astype(str)

# Define region
regions = ['SW', 'SE', 'NW', 'NE']

for region in regions:
    
    blocks = list(block_list[block_list['Quadrant'] == region]['date_str'])
    
    # Define files
    files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_SW/*.csv'))
    
    # Define good hours
    good_hours = ['06', '07', '08', '09', '10', '11', '12', '13', '14']
        
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
    
    # Define ice sheet grid
    ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
    ismip_lon = ismip.variables['lon'][:]
    ismip_lat = ismip.variables['lat'][:]
    
    # Define destination
    dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_SW_Blocking/'
    
    for p in good_hours:
        
        if os.path.exists(dest + 'MYD06_Fluxes_' + p + '_' + region + '_Block.nc'):
            pass
        else:
            
            # Get a list of similar hours
            files_hours = []
            
            for j in block_files:
                # Get path and filename seperately 
                infilepath, infilename = os.path.split(j)
                # Get file name without extension            
                infilehortname, extension = os.path.splitext(infilename)
                
                if infilehortname[18:20] == p:
                    files_hours.append(j)
                    
            clrsky_sw = np.zeros(ismip_lat.shape)
            allsky_sw = np.zeros(ismip_lat.shape)
             
            for i in range(len(files_hours)):
                print('Processing... %.0f out of %.0f' %(i, len(files_hours)))
                
                # Get path and filename seperately
                infilepath, infilename = os.path.split(files_hours[i])
                # Get file name without extension            
                infileshortname, extension = os.path.splitext(infilename)
                    
                # Read csv
                df = pd.read_csv(files_hours[i])
                
                if df.shape[0] > 200000:
                
                    # Resample radiative fluxes to ISMIP grid
                    swath_def = pyresample.geometry.SwathDefinition(lons=df['lon'].values, lats=df['lat'].values)
                    swath_con = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
                    
                    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
                    clearsky_sw_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                                     target_geo_def=swath_con, 
                                                                     data=df['clearsky_sw'].values, 
                                                                     radius_of_influence=5000)
                    allsky_sw_data = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                                     target_geo_def=swath_con, 
                                                                     data=df['allsky_sw'].values, 
                                                                     radius_of_influence=5000)
                    
                    # Set zeros to NaNs
                    allsky_sw_data[allsky_sw_data == 0] = np.nan
                    clearsky_sw_data[clearsky_sw_data == 0] = np.nan
                    
                    # Stack
                    clrsky_sw = np.dstack((clrsky_sw, clearsky_sw_data))
                    allsky_sw = np.dstack((allsky_sw, allsky_sw_data))
                
                else:
                    pass
                
            # Remove first layer
            clrsky_sw = clrsky_sw[:, :, 1:]
            allsky_sw = allsky_sw[:, :, 1:]
            
            # Remove layer if only observed a few times
            allsky_sw[np.nansum(np.isfinite(allsky_sw), axis=2) < 4] = np.nan
            clrsky_sw[np.nansum(np.isfinite(allsky_sw), axis=2) < 4] = np.nan
            
            ###############################################################################
            # Save 1 km dataset to NetCDF
            ###############################################################################
            dataset = netCDF4.Dataset(dest + 'MYD06_Fluxes_' + p + '_' + region + '_Block.nc', 
                                      'w', format='NETCDF4_CLASSIC')
            print('Creating... %s' % dest + 'MYD06_Fluxes_' + p + '_' + region + '_Block.nc')
            dataset.Title = "All-sky fluxes and clear-sky fluxes from MYD06_L2 product during blocking events."
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
            clrsky_sw_nc = dataset.createVariable('clrsky_sw', np.float32, ('y','x'))
            allsky_sw_nc = dataset.createVariable('allsky_sw', np.float32, ('y','x'))
                       
            # Write data to layers
            Y[:] = ismip_lat
            X[:] = ismip_lon
            clrsky_sw_nc[:] = np.nanmean(clrsky_sw, axis=2)
            allsky_sw_nc[:] = np.nanmean(allsky_sw, axis=2)
            
            print('Writing data to %s' % dest + 'MYD06_Fluxes_' + p + '_' + region + '_Block.nc')
                
            # Close dataset
            dataset.close()
    
