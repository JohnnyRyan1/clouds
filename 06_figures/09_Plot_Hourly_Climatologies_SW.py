#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Visualize radiative flux climatologies.

"""

# Import modules
import numpy as np
import os
import glob
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pyplot as plt

# Define year
year = str(2003)

# Folder
folder = '/media/johnny/Cooley_Data/Johnny/Clouds/3_MYD06_Radiative_Fluxes_NC_SW_V2/'

# Destination
dest = '/home/johnny/Documents/Clouds/Presentations/2020-12-21/'

# Define file
file_string = sorted(glob.glob(folder + 'MYD06_Fluxes_' + str(year) + '_*.nc'))

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Define MODIS albedo data
albedo_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_' + str(year) + '.nc')
albedo = albedo_file.variables['albedo'][:, :, :].filled(np.nan) / 100
albedo[albedo == 0] = np.nan
albedo_mean = np.nanmean(albedo, axis=0)
albedo_mean[~ismip_mask.astype(bool)] = np.nan

for i in file_string:
    # Import data
    f = netCDF4.Dataset(i)
    
    # Get path and filename seperately 
    infilepath, infilename = os.path.split(i)
    # Get file name without extension            
    infilehortname, extension = os.path.splitext(infilename)
    
    # Append hour
    hour = int(infilehortname[18:20])
    
    # Read data
    net_sw = f.variables['allsky_sw'][:] - f.variables['clrsky_sw'][:]
    lats = f.variables['latitude'][:]
    lons = f.variables['longitude'][:]
    
    # Mask data
    net_sw[~ismip_mask.astype(bool)] = np.nan
    
    # Resize for convenient plotting
    cre_sw = (net_sw*(1 - albedo_mean))[::5,::5]
    lons = lons[::5,::5]
    lats = lats[::5,::5]
    
    ###############################################################################
    # Plot cloud radiative effect LW
    ###############################################################################
    fig = plt.figure(figsize=(4, 4))
    v = np.arange(-50, 0, 2)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    plt.contourf(lons, lats, cre_sw, v, transform=ccrs.PlateCarree(), vmin=-50, vmax=0)
    ax.coastlines()
    cbar = plt.colorbar()
    cbar.set_label('Cloud radiative effect on SW (W m-2)', rotation=270, labelpad=12)
    plt.tight_layout()
    plt.savefig(dest + 'summer_cre_sw_' + year + '_' + str(hour).zfill(2) + '.png', dpi=200)










