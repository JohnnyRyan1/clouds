#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Produce summer SW radiative flux climatology for blocking events.

"""

# Import modules
import numpy as np
import glob
import os
import netCDF4

regions = ['SW', 'SE', 'NW', 'NE']
for region in regions:
    
    # Define files
    files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/3_MYD06_Radiative_Fluxes_NC_SW_Blocking/MYD06_Fluxes_*' + region + '_Block.nc'))
    
    # Define hourly to daily corrections
    conversions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/ERA5_SW_Conversions.nc')
    
    # Define destination
    dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/'
    
    # Define ice sheet grid
    ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
    ismip_lon = ismip.variables['lon'][:]
    ismip_lat = ismip.variables['lat'][:]
    ismip_mask = ismip.variables['ICE'][:]
    
    # Define MODIS albedo data
    albedo_files = glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/Summer_Albedo_Climatologies/*')[0:16]
    
    # Produce mean albedo for 2003 to 2018 period
    albedo_all = np.zeros(ismip_lat.shape)
    for i in albedo_files:
        albedo_file = netCDF4.Dataset(i)
        albedo = albedo_file.variables['albedo'][:, :, :].filled(np.nan).astype(float)
        albedo[albedo == 0] = np.nan
        albedo_all = np.dstack((albedo_all, np.nanmean(albedo, axis=0)))
    
    albedo_all = albedo_all[:, :, 1:]
    albedo_mean = np.nanmean(albedo_all, axis=2) / 100
    
    # Define some empty arrays
    clrsky_sw = np.zeros(ismip_lat.shape)
    allsky_sw = np.zeros(ismip_lat.shape)
    net_sw = np.zeros(ismip_lat.shape)
    net_sw_corrected = np.zeros(ismip_lat.shape)
    
    for i in range(len(files)):
        print('Processing... %.0f out of %.0f' %(i+1, len(files)))
    
        # Read netCDF
        f = netCDF4.Dataset(files[i])
        
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(files[i])
        # Get file name without extension            
        infilehortname, extension = os.path.splitext(infilename)
        
        # Get corresponding hour
        hour = int(infilehortname[13:15])
        
        # Correct from hourly to daily        
        add_allsky_sw = conversions.variables['correction_allsky_add'][:,:,hour]
        add_clrsky_sw = conversions.variables['correction_clrsky_add'][:,:,hour]
       
        clrsky_sw_data = f.variables['clrsky_sw'][:] + add_clrsky_sw
        allsky_sw_data = f.variables['allsky_sw'][:] + add_allsky_sw
    
        # Stack
        clrsky_sw = np.dstack((clrsky_sw, clrsky_sw_data))
        allsky_sw = np.dstack((allsky_sw, allsky_sw_data))
        
        net_sw = (allsky_sw_data) - (clrsky_sw_data)
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
    
    # Compute some extra values
    cre_sw = net_sw_corrected_mean * (1 - albedo_mean)
    cldy_sw = np.abs(net_sw_corrected_mean / clrsky_sw_mean)
    
    # Mask data
    clrsky_sw_mean[~ismip_mask.astype(bool)] = np.nan
    allsky_sw_mean[~ismip_mask.astype(bool)] = np.nan
    net_sw_corrected_mean[~ismip_mask.astype(bool)] = np.nan
    cre_sw[~ismip_mask.astype(bool)] = np.nan
    cldy_sw[~ismip_mask.astype(bool)] = np.nan
    
    ###############################################################################
    # Save 1 km dataset to NetCDF
    ###############################################################################
    dataset = netCDF4.Dataset(dest + 'MYD06_SW_Fluxes_' + region + '.nc', 
                              'w', format='NETCDF4_CLASSIC')
    print('Creating... %s' % dest + 'MYD06_SW_Fluxes_' + region + '.nc')
    dataset.Title = "All-sky fluxes, clear-sky fluxes, and albedo from MYD06_L2 product for blocking events."
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
    cre_sw_nc = dataset.createVariable('cre_sw', np.float32, ('y','x'))
    cloudiness_nc = dataset.createVariable('cloudiness', np.float32, ('y','x'))
               
    # Write data to layers
    Y[:] = ismip_lat
    X[:] = ismip_lon
    clrsky_sw_nc[:] = clrsky_sw_mean
    allsky_sw_nc[:] = allsky_sw_mean
    albedo_nc[:] = albedo_mean
    net_sw_corrected_nc[:] = net_sw_corrected_mean
    cre_sw_nc[:] = cre_sw
    cloudiness_nc[:] = cldy_sw
    
    print('Writing data to %s' % dest + 'MYD06_SW_Fluxes_' + region + '.nc')
        
    # Close dataset
    dataset.close()
    
    
    
    
    
    
    
    
    





























