#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION 

Compute statistics for mean climatology vs. blocking climatology.

"""

# Import modules
import numpy as np
import netCDF4
#import pyresample
#from pyproj import Transformer

# Import data
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')
block = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')

# Define destination
dest = '/home/johnny/Documents/Clouds/Data/Model_Evaluation/'
# =============================================================================
# 
# # Define ice sheet grid
# ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
# ismip_lon = ismip.variables['lon'][:]
# ismip_lat = ismip.variables['lat'][:]
# ismip_mask = ismip.variables['ICE'][:]
# =============================================================================

# Define maximum snowline
snowline_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_2012.nc')
snowline = snowline_file.variables['bare_ice'][1, :, :].filled(np.nan)
max_snowline = (snowline > 0.1)

# =============================================================================
# # Define region mask
# regions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/Ultimate_Mask.nc')
# regions_lon = regions.variables['x'][:]
# regions_lat = regions.variables['y'][:]
# regions_1 = regions.variables['North'][:]
# regions_2 = regions.variables['NorthEast'][:]
# regions_3 = regions.variables['East'][:]
# regions_4 = regions.variables['SouthEast'][:]
# regions_5 = regions.variables['South'][:]
# regions_6 = regions.variables['SouthWest'][:]
# regions_7 = regions.variables['West'][:]
# regions_8 = regions.variables['NorthWest'][:]
# 
# regions_2[regions_2 > 0] = 2
# regions_3[regions_3 > 0] = 3
# regions_4[regions_4 > 0] = 4
# regions_5[regions_5 > 0] = 5
# regions_6[regions_6 > 0] = 6
# regions_7[regions_7 > 0] = 7
# regions_8[regions_8 > 0] = 8
# 
# regions_mask = regions_1 + regions_2 + regions_3 + regions_4 + regions_5 + regions_6 +\
#     regions_7 + regions_8
#     
# # Convert from stereographic to WGS84
# transformer = Transformer.from_crs(3413, 4326)
# lat, lon = transformer.transform(regions_lon, regions_lat)
# 
# # Define grid using pyresample       
# grid = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
# swath = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
# 
# # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
# region_swath = pyresample.kd_tree.resample_nearest(source_geo_def=swath, 
#                                              target_geo_def=grid, 
#                                              data=regions_mask, 
#                                              radius_of_influence=50000)
# =============================================================================

# Import data
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

# SW
sw_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')
sw_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SE.nc')
sw_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NW.nc')
sw_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NE.nc')

# LW
lw_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SW.nc')
lw_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SE.nc')
lw_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NW.nc')
lw_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NE.nc')

# Read data
# SW
mod_sw = np.nanmean(np.nanmean(mod.variables['allsky_sw'][:], axis=2) [max_snowline])
block_sw_sw = np.nanmean(sw_sw.variables['allsky_sw'][:][max_snowline])
block_sw_se = np.nanmean(sw_se.variables['allsky_sw'][:][max_snowline])
block_sw_nw = np.nanmean(sw_nw.variables['allsky_sw'][:][max_snowline])
block_sw_ne = np.nanmean(sw_ne.variables['allsky_sw'][:][max_snowline])

mod_sw_clr = np.nanmean(np.nanmean(mod.variables['clrsky_sw'][:], axis=2) [max_snowline])
block_sw_sw_clr = np.nanmean(sw_sw.variables['clrsky_sw'][:][max_snowline])
block_sw_se_clr = np.nanmean(sw_se.variables['clrsky_sw'][:][max_snowline])
block_sw_nw_clr = np.nanmean(sw_nw.variables['clrsky_sw'][:][max_snowline])
block_sw_ne_clr = np.nanmean(sw_ne.variables['clrsky_sw'][:][max_snowline])

# LW
mod_lw = np.nanmean(np.nanmean(mod.variables['allsky_lw'][:], axis=2)[max_snowline])
block_lw_sw = np.nanmean(lw_sw.variables['allsky_lw'][:][max_snowline])
block_lw_se = np.nanmean(lw_se.variables['allsky_lw'][:][max_snowline])
block_lw_nw = np.nanmean(lw_nw.variables['allsky_lw'][:][max_snowline])
block_lw_ne = np.nanmean(lw_ne.variables['allsky_lw'][:][max_snowline])

mod_lw_clr = np.nanmean(np.nanmean(mod.variables['clrsky_lw'][:], axis=2)[max_snowline])
block_lw_sw_clr = np.nanmean(lw_sw.variables['clrsky_lw'][:][max_snowline])
block_lw_se_clr = np.nanmean(lw_se.variables['clrsky_lw'][:][max_snowline])
block_lw_nw_clr = np.nanmean(lw_nw.variables['clrsky_lw'][:][max_snowline])
block_lw_ne_clr = np.nanmean(lw_ne.variables['clrsky_lw'][:][max_snowline])

# Scale according to clear-sky values
mod_sw_flux = mod_sw /  mod_sw_clr 
block_sw_flux = block_sw_sw /  block_sw_sw_clr
block_se_flux = block_sw_se /  block_sw_se_clr
block_nw_flux = block_sw_nw /  block_sw_nw_clr
block_ne_flux = block_sw_ne /  block_sw_ne_clr

block_flux = (block_se_flux + block_sw_flux + block_ne_flux + block_nw_flux) / 4

sw_flux_diff = (block_flux - mod_sw_flux) * mod_sw_clr

# Compute net radiative fluxes for blocking events in ablation zone
mod_sw_currrent = (mod_sw_flux * mod_sw_clr) * (1 - 0.62)
block_sw_current = (block_flux * mod_sw_clr) * (1 - 0.62)

mod_sw_future = (mod_sw_flux * mod_sw_clr) * (1 - 0.43)
block_sw_future = (block_flux * mod_sw_clr) * (1 - 0.43)

# Scale according to clear-sky values
mod_lw_flux = mod_lw /  mod_lw_clr 
block_sw_flux = block_lw_sw /  block_lw_sw_clr
block_se_flux = block_lw_se /  block_lw_se_clr
block_nw_flux = block_lw_nw /  block_lw_nw_clr
block_ne_flux = block_lw_ne /  block_lw_ne_clr

block_flux = (block_se_flux + block_sw_flux + block_ne_flux + block_nw_flux) / 4

lw_flux_diff = (block_flux - mod_lw_flux) * mod_lw_clr

# Compute net radiative fluxes for blocking events in ablation zone
mod_lw_currrent = (mod_lw_flux * mod_lw_clr)
block_lw_current = (block_flux * mod_lw_clr)

# Compute cloudiness for blocking events in the ablation zone
mod_cldy = np.abs((mod_sw - mod_sw_clr) /  mod_sw_clr)
sw_cldy = np.abs((block_sw_sw - block_sw_sw_clr) /  block_sw_sw_clr)
se_cldy = np.abs((block_sw_se - block_sw_se_clr) /  block_sw_se_clr)
nw_cldy = np.abs((block_sw_nw - block_sw_nw_clr) /  block_sw_nw_clr)
ne_cldy = np.abs((block_sw_ne - block_sw_ne_clr) /  block_sw_ne_clr)













