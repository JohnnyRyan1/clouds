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
year = str(2011)

# Define file
file_string = '/media/johnny/Cooley_Data/Johnny/Clouds/4_MYD06_Radiative_Flux_Climatologies/MYD06_LW_Fluxes_' + year + '.nc'

# Import data
f = netCDF4.Dataset(file_string)

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Read data
clrsky_lw = f.variables['clrsky_lw'][:]
allsky_lw = f.variables['allsky_lw'][:]
lats = f.variables['latitude'][:]
lons = f.variables['longitude'][:]

# Mask data
clrsky_lw[~ismip_mask.astype(bool)] = np.nan
allsky_lw[~ismip_mask.astype(bool)] = np.nan

# Compute CRE LW at Summit as a sanity check
summit_lat, summit_lon = 72.68, -38.58
abslat = np.abs(ismip_lat - summit_lat)
abslon= np.abs(ismip_lon - summit_lon)
c = np.maximum(abslon, abslat)
x, y = np.where(c == np.min(c))
summit_cre_lw = (allsky_lw - clrsky_lw)[x[0], y[0]]

# Resize for convenient plotting
clrsky_lw = clrsky_lw[::5,::5]
allsky_lw = allsky_lw[::5,::5]
lons = lons[::5,::5]
lats = lats[::5,::5]

###############################################################################
# Plot downward clear-sky LW
###############################################################################
fig = plt.figure(figsize=(4, 4))
v = np.arange(170, 310, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
cs = plt.contourf(lons, lats, clrsky_lw, v, transform=ccrs.PlateCarree(), vmin=170, vmax=310)
ax.coastlines()
cbar = plt.colorbar()
cbar.set_label('Clear-sky downward LW (W m-2)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-21/summer_clrsky_lw_' + year + '.png', dpi=200)

###############################################################################
# Plot downward all-sky LW
###############################################################################
fig = plt.figure(figsize=(4, 4))
v = np.arange(170, 310, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, allsky_lw, v, transform=ccrs.PlateCarree(), vmin=170, vmax=310)
ax.coastlines()
cbar = plt.colorbar()
cbar.set_label('All-sky downward LW (W m-2)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-21/summer_allsky_lw_' + year + '.png', dpi=200)

###############################################################################
# Plot cloud radiative effect LW
###############################################################################
fig = plt.figure(figsize=(4, 4))
v = np.arange(0, 60, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, allsky_lw - clrsky_lw, v, transform=ccrs.PlateCarree(), vmin=0, vmax=60)
ax.coastlines()
cbar = plt.colorbar()
cbar.set_label('Cloud radiative effect on LW (W m-2)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-21/summer_cre_lw_' + year + '.png', dpi=200)










