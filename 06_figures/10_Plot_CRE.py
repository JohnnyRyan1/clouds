#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Visualize cloud radiative effects.

"""

# Import modules
import numpy as np
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pyplot as plt

# Define year
year = 2010

# Define destination
dest = '/home/johnny/Documents/Clouds/Presentations/2020-12-21/'

# Define LW
lw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds/4_MYD06_Radiative_Flux_Climatologies/MYD06_LW_Fluxes_' + str(year) + '.nc')

# Define SW
sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds/4_MYD06_Radiative_Flux_Climatologies/MYD06_SW_Fluxes_' + str(year) + '.nc')

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

# Derive CRE
cre = (lw.variables['allsky_lw'][:] - lw.variables['clrsky_lw'][:]) +\
    (sw.variables['net_sw_corrected'] * (1 - albedo_mean))
    
# Mask data
cre[~ismip_mask.astype(bool)] = np.nan
    
# Resize for convenient plotting
cre = cre[::5,::5]
lons = ismip_lon[::5,::5]
lats = ismip_lat[::5,::5]

###############################################################################
# Plot all-wave cloud radiative effect 
###############################################################################
fig = plt.figure(figsize=(4, 4))
v = np.arange(-45, 45, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, cre, v, transform=ccrs.PlateCarree(), vmin=-45, vmax=45,
             cmap='coolwarm')
ax.coastlines()
cbar = plt.colorbar()
cbar.set_label('Cloud radiative effect (W m-2)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig(dest + 'summer_cre_allwave_' + str(year) + '.png', dpi=200)











