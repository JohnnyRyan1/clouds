#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Visualize radiative flux climatologies.

"""

# Import modules
import numpy as np
import cartopy.crs as ccrs
import netCDF4
import matplotlib.pyplot as plt

# Define year
year = str(2006)

# Define file
file_string = '/media/johnny/Cooley_Data/Johnny/Clouds/4_MYD06_Radiative_Flux_Climatologies/MYD06_SW_Fluxes_' + year + '.nc'

# Import data
f = netCDF4.Dataset(file_string)

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Read data
net_sw_corrected = f.variables['net_sw_corrected'][:]
albedo = f.variables['albedo'][:] / 100
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]

# Mask data
net_sw_corrected[~ismip_mask.astype(bool)] = np.nan
albedo[~ismip_mask.astype(bool)] = np.nan

# Compute CRE LW at Summit as a sanity check
summit_lat, summit_lon = 72.68, -38.58
abslat = np.abs(ismip_lat - summit_lat)
abslon= np.abs(ismip_lon - summit_lon)
c = np.maximum(abslon, abslat)
x, y = np.where(c == np.min(c))
summit_cre_sw = (net_sw_corrected*(1-(albedo)))[x[0], y[0]]

# Resize for convenient plotting
cre_sw = (net_sw_corrected*(1-(albedo)))[::5,::5]
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
plt.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-21/summer_cre_sw_' + year + '.png', dpi=200)










