#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Draft "blocking" figures for paper

"""

# Import modules
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import data
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

###############################################################################
# Figure 3a. Cloudiness during all Greenland blocking events
###############################################################################
block_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SE.nc')
block_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')
block_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NE.nc')
block_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NW.nc')

mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

lons = mod.variables['longitude'][:]
lats = mod.variables['latitude'][:]
mod_cldy = np.nanmean(mod.variables['cloudiness'][:], axis=2) * 100
block_cldy_se = block_se.variables['cloudiness'][:] * 100
block_cldy_sw = block_sw.variables['cloudiness'][:] * 100
block_cldy_ne = block_ne.variables['cloudiness'][:] * 100
block_cldy_nw = block_nw.variables['cloudiness'][:] * 100

block_cldy = (block_cldy_se + block_cldy_sw + block_cldy_ne + block_cldy_nw) / 4

cldy = block_cldy - mod_cldy

# Resize for more convenient plotting
cldy = cldy[::5,::5]
lons = lons[::5,::5]
lats = lats[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(-15, 20, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, cldy, v, transform=ccrs.PlateCarree(), vmin=-15, vmax=20,
             cmap='coolwarm')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-15, -10, -5, 0, 5, 10, 15])
cbar.ax.set_yticklabels([-15, -10, -5, 0, 5, 10, 15]) 
cbar.set_label('Cloudiness during blocks (Block - Control) (%)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_3a_Cloudiness_Blocking.png', dpi=200)

###############################################################################
# Figure 3b. Net shortwave energy flux during all Greenland blocking events
###############################################################################
block_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SE.nc')
block_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')
block_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NE.nc')
block_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NW.nc')

mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

lons = mod.variables['longitude'][:]
lats = mod.variables['latitude'][:]
mod_sw_flux = np.nanmean(mod.variables['allsky_sw'][:], axis=2) /  np.nanmean(mod.variables['clrsky_sw'][:], axis=2) 
block_se_flux = block_se.variables['allsky_sw'][:] /  block_se.variables['clrsky_sw'][:]
block_sw_flux = block_sw.variables['allsky_sw'][:] /  block_sw.variables['clrsky_sw'][:]
block_ne_flux = block_ne.variables['allsky_sw'][:] /  block_ne.variables['clrsky_sw'][:]
block_nw_flux = block_nw.variables['allsky_sw'][:] /  block_nw.variables['clrsky_sw'][:]

block_flux = (block_se_flux + block_sw_flux + block_ne_flux + block_nw_flux) / 4

sw_flux_diff = (block_flux - mod_sw_flux) * np.nanmean(mod.variables['clrsky_sw'][:], axis=2)

# Resize for more convenient plotting
sw_flux_diff = sw_flux_diff[::5,::5]
lons = lons[::5,::5]
lats = lats[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(-75, 75, 5)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, sw_flux_diff, v, transform=ccrs.PlateCarree(), vmin=-75, vmax=75,
             cmap='coolwarm')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-50, -30, -10, 10, 30, 50])
cbar.ax.set_yticklabels([-50, -30, -10, 10, 30, 50]) 
cbar.set_label('SW$_{net}$ during blocks (Block - Control) (W m$^{-2}$)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_3b_SW_Flux_Blocking.png', dpi=200)

###############################################################################
# Figure 3c. Downward longwave energy flux during all Greenland blocking events
###############################################################################
block_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SE.nc')
block_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SW.nc')
block_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NE.nc')
block_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NW.nc')

mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

lons = mod.variables['longitude'][:]
lats = mod.variables['latitude'][:]
mod_lw_flux = np.nanmean(mod.variables['allsky_lw'][:], axis=2) /  np.nanmean(mod.variables['clrsky_lw'][:], axis=2) 
block_se_flux = block_se.variables['allsky_lw'][:] /  block_se.variables['clrsky_lw'][:]
block_sw_flux = block_sw.variables['allsky_lw'][:] /  block_sw.variables['clrsky_lw'][:]
block_ne_flux = block_ne.variables['allsky_lw'][:] /  block_ne.variables['clrsky_lw'][:]
block_nw_flux = block_nw.variables['allsky_lw'][:] /  block_nw.variables['clrsky_lw'][:]

block_flux = (block_se_flux + block_sw_flux + block_ne_flux + block_nw_flux) / 4

lw_flux_diff = (block_flux - mod_lw_flux) * np.nanmean(mod.variables['clrsky_lw'][:], axis=2)

# Resize for more convenient plotting
lw_flux_diff = lw_flux_diff[::5,::5]
lons = lons[::5,::5]
lats = lats[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(-75, 75, 5)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, lw_flux_diff, v, transform=ccrs.PlateCarree(), vmin=-75, vmax=75,
             cmap='coolwarm')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-50, -30, -10, 10, 30, 50])
cbar.ax.set_yticklabels([-50, -30, -10, 10, 30, 50]) 
cbar.set_label('LW flux during blocks (Block - Control) (W m$^{-2}$)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_3c_LW_Flux_Blocking.png', dpi=200)

###############################################################################
# Figure 3d. Allwave energy flux during all Greenland blocking events
###############################################################################
flux_diff = lw_flux_diff + sw_flux_diff

fig = plt.figure(figsize=(4, 4))
v = np.arange(-75, 75, 5)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, flux_diff, v, transform=ccrs.PlateCarree(), vmin=-75, vmax=75,
             cmap='coolwarm')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-50, -30, -10, 10, 30, 50])
cbar.ax.set_yticklabels([-50, -30, -10, 10, 30, 50])
cbar.set_label('Allwave energy flux (Block - Control) (W m$^{-2}$)', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_3d_Allwave_Flux_Blocking.png', dpi=200)



