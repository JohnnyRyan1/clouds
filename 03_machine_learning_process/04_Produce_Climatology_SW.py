#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Produce summer SW radiative flux climatology

"""

# Import modules
import numpy as np
import glob
import os
import netCDF4

# Define destination
dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/'

years = np.arange(2003, 2021, 1)
for year in years:
    
    if os.path.exists(dest + 'MYD06_SW_Fluxes_' + str(year) + '.nc'):
        pass
    else:
        # Define good hours
        good_hours = ['06', '07', '08', '09', '10', '11', '12', '13', '14']
    
        # Define files
        files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_SW/*' + str(year) + '*.nc'))
        
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
        
        # Define hourly to daily corrections
        conversions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/ERA5_SW_Conversions.nc')
                
        # Define ice sheet grid
        ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
        ismip_lon = ismip.variables['lon'][:]
        ismip_lat = ismip.variables['lat'][:]
        
        # Define MODIS albedo data
        albedo_files = glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/Summer_Albedo_Climatologies/*')
        
        # Read MODIS albedo data
        albedo_match = [s for s in albedo_files if str(year) in s]
        albedo_file = netCDF4.Dataset(albedo_match[0])
        albedo = albedo_file.variables['albedo'][:, :, :].filled(np.nan).astype(float)
        albedo[albedo == 0] = np.nan
        
        # Define some empty arrays
        clrsky_sw = np.zeros(ismip_lat.shape)
        allsky_sw = np.zeros(ismip_lat.shape)
        net_sw_corrected = np.zeros(ismip_lat.shape)
        
        for i in range(len(good_files)):
            print('Processing... %.0f out of %.0f' %(i+1, len(good_files)))
        
            # Read netCDF
            f = netCDF4.Dataset(good_files[i])
            
            # Get path and filename seperately 
            infilepath, infilename = os.path.split(good_files[i])
            # Get file name without extension            
            infilehortname, extension = os.path.splitext(infilename)
            
            # Get corresponding hour
            hour = int(infilehortname[18:20])
            
            # Correct from hourly to daily       
            add_allsky_sw = conversions.variables['correction_allsky_add'][:,:,hour]
            add_clrsky_sw = conversions.variables['correction_clrsky_add'][:,:,hour]
            
            clrsky_sw_data = f.variables['clrsky_sw'][:] + add_clrsky_sw
            allsky_sw_data = f.variables['allsky_sw'][:] + add_allsky_sw
            
            # Stack
            clrsky_sw = np.dstack((clrsky_sw, clrsky_sw_data))
            allsky_sw = np.dstack((allsky_sw, allsky_sw_data))
            
            net_sw = allsky_sw_data - clrsky_sw_data
            net_sw_corrected = np.dstack((net_sw_corrected, net_sw))
            
        # Remove first layer
        clrsky_sw = clrsky_sw[:, :, 1:]
        allsky_sw = allsky_sw[:, :, 1:]
        net_sw_corrected = net_sw_corrected[:, :, 1:]
        
        # Convert spurious values to zero
        net_sw_corrected[net_sw_corrected >= 0] = 0
        
        # Average
        clrsky_sw_mean = np.nanmean(clrsky_sw, axis=2)
        allsky_sw_mean = np.nanmean(allsky_sw, axis=2)
        net_sw_corrected_mean = np.nanmean(net_sw_corrected, axis=2)
        albedo_mean = np.nanmean(albedo, axis=0)
        
        ###############################################################################
        # Save 1 km dataset to NetCDF
        ###############################################################################
        dataset = netCDF4.Dataset(dest + 'MYD06_SW_Fluxes_' + str(year) + '.nc', 
                                  'w', format='NETCDF4_CLASSIC')
        print('Creating... %s' % dest + 'MYD06_SW_Fluxes_' + str(year) + '.nc')
        dataset.Title = "All-sky fluxes, clear-sky fluxes, and albedo from MYD06_L2 product"
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
        albedo_nc = dataset.createVariable('albedo', np.float32, ('y','x'))
        net_sw_corrected_nc = dataset.createVariable('net_sw_corrected', np.float32, ('y','x'))
                   
        # Write data to layers
        Y[:] = ismip_lat
        X[:] = ismip_lon
        clrsky_sw_nc[:] = clrsky_sw_mean
        allsky_sw_nc[:] = allsky_sw_mean
        albedo_nc[:] = albedo_mean
        net_sw_corrected_nc[:] = net_sw_corrected_mean
        
        print('Writing data to %s' % dest + 'MYD06_SW_Fluxes_' + str(year) + '.nc')
            
        # Close dataset
        dataset.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
























