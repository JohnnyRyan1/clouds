#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Regrid and stack radiative flux data to ISMIP grid.

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
import netCDF4
import pyresample

# Define years
years = np.arange(2003, 2021, 1)
for year in years:
    
    # Define files
    files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_SW/*' + str(year) + '*.csv'))
    
    # Define good hours
    good_hours = ['06', '07', '08', '09', '10', '11', '12', '13', '14']
    
    good_files = []
    for j in files:
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(j)
        # Get file name without extension            
        infilehortname, extension = os.path.splitext(infilename)
        
        # Append hour
        hour = infilehortname[18:20]
        if hour in good_hours:
            good_files.append(j)
            
    # Define ice sheet grid
    ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
    ismip_lon = ismip.variables['lon'][:]
    ismip_lat = ismip.variables['lat'][:]
    
    # Define destination
    dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_SW/'
    
    for p in good_hours:
        
        if os.path.exists(dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_a.nc'):
            pass
        else:
            
            # Get a list of similar hours
            files_hours = []
            
            for j in files:
                # Get path and filename seperately 
                infilepath, infilename = os.path.split(j)
                # Get file name without extension            
                infilehortname, extension = os.path.splitext(infilename)
                
                if infilehortname[18:20] == p:
                    files_hours.append(j)
            
            # Split into two again
            files_hours_a = files_hours[:len(files_hours)//2]
            files_hours_b = files_hours[len(files_hours)//2:]
            
            clrsky_sw_a = np.zeros(ismip_lat.shape)
            allsky_sw_a = np.zeros(ismip_lat.shape)
        
            for i in range(len(files_hours_a)):
                print('Processing... %.0f out of %.0f' %(i, len(files_hours_a)))
                
                # Get path and filename seperately
                infilepath, infilename = os.path.split(files_hours_a[i])
                # Get file name without extension            
                infileshortname, extension = os.path.splitext(infilename)
                    
                # Read csv
                df = pd.read_csv(files_hours_a[i])
                #print(i, df.shape[0])
                
                # Check if there are enough points, could be a weird strip if too few
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
                    clrsky_sw_a = np.dstack((clrsky_sw_a, clearsky_sw_data))
                    allsky_sw_a = np.dstack((allsky_sw_a, allsky_sw_data))
                
                else:
                    pass
                
            # Remove first layer
            clrsky_sw_a = clrsky_sw_a[:, :, 1:]
            allsky_sw_a = allsky_sw_a[:, :, 1:]
            
            # Remove layer if only observed a few times
            allsky_sw_a[np.nansum(np.isfinite(allsky_sw_a), axis=2) < 4] = np.nan
            clrsky_sw_a[np.nansum(np.isfinite(clrsky_sw_a), axis=2) < 4] = np.nan
            
            clrsky_sw_b = np.zeros(ismip_lat.shape)
            allsky_sw_b = np.zeros(ismip_lat.shape)
        
            for i in range(len(files_hours_b)):
                print('Processing... %.0f out of %.0f' %(i, len(files_hours_b)))
                
                # Get path and filename seperately 
                infilepath, infilename = os.path.split(files_hours_b[i])
                # Get file name without extension            
                infileshortname, extension = os.path.splitext(infilename)
                    
                # Read csv
                df = pd.read_csv(files_hours_b[i])
                
                # Check if there are enough points, could be a weird strip if too few
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
                    clrsky_sw_b = np.dstack((clrsky_sw_b, clearsky_sw_data))
                    allsky_sw_b = np.dstack((allsky_sw_b, allsky_sw_data))
                
                else:
                    pass
                
            # Remove first layer
            clrsky_sw_b = clrsky_sw_b[:, :, 1:]
            allsky_sw_b = allsky_sw_b[:, :, 1:]
            
            # Remove layer if only observed a few times
            allsky_sw_b[np.nansum(np.isfinite(allsky_sw_b), axis=2) < 4] = np.nan
            clrsky_sw_b[np.nansum(np.isfinite(clrsky_sw_b), axis=2) < 4] = np.nan
                
            ###############################################################################
            # Save 1 km dataset to NetCDF
            ###############################################################################
            dataset = netCDF4.Dataset(dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_a.nc', 
                                      'w', format='NETCDF4_CLASSIC')
            print('Creating... %s' % dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_a.nc')
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
            clrsky_sw_nc = dataset.createVariable('clrsky_sw', np.float32, ('y','x'))
            allsky_sw_nc = dataset.createVariable('allsky_sw', np.float32, ('y','x'))
                       
            # Write data to layers
            Y[:] = ismip_lat
            X[:] = ismip_lon
            clrsky_sw_nc[:] = np.nanmean(clrsky_sw_a, axis=2)
            allsky_sw_nc[:] = np.nanmean(allsky_sw_a, axis=2)
            
            print('Writing data to %s' % dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_a.nc')
                
            # Close dataset
            dataset.close()
            
            ###############################################################################
            # Save 1 km dataset to NetCDF
            ###############################################################################
            dataset = netCDF4.Dataset(dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_b.nc', 
                                      'w', format='NETCDF4_CLASSIC')
            print('Creating... %s' % dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_b.nc')
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
            clrsky_sw_nc = dataset.createVariable('clrsky_sw', np.float32, ('y','x'))
            allsky_sw_nc = dataset.createVariable('allsky_sw', np.float32, ('y','x'))
                       
            # Write data to layers
            Y[:] = ismip_lat
            X[:] = ismip_lon
            clrsky_sw_nc[:] = np.nanmean(clrsky_sw_b, axis=2)
            allsky_sw_nc[:] = np.nanmean(allsky_sw_b, axis=2)
            
            print('Writing data to %s' % dest + 'MYD06_Fluxes_' + str(year) + '_' + p + '_b.nc')
                
            # Close dataset
            dataset.close()
