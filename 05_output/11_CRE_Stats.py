#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Cloud radiative effect statistics.

"""

# Import modules
import pandas as pd
import numpy as np
import netCDF4

# Define years
years = np.arange(2003, 2021, 1)

# Define destination
dest = '/Users/jryan4/Dropbox (University of Oregon)/research/clouds/data/'

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Define maximum snowline
snowline_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_2012.nc')
snowline = snowline_file.variables['bare_ice'][1, :, :].filled(np.nan)
max_snowline = (snowline > 0.1)

# Define some empty lists
cre_gre = []
cre_abl = []
cre_acc = []

cre_lw_gre = []
cre_lw_abl = []
cre_lw_acc = []

cre_sw_gre = []
cre_sw_abl = []
cre_sw_acc = []

alb_gre = []
alb_abl = []
alb_acc = []

cldy_gre = []
cldy_abl = []
cldy_acc = []

allsky_sw_gre = []
allsky_sw_abl = []
allsky_sw_acc = []

allsky_lw_gre = []
allsky_lw_abl = []
allsky_lw_acc = []

cre_net_map = np.zeros(ismip_mask.shape)
cre_lw_map = np.zeros(ismip_mask.shape)
cre_sw_map = np.zeros(ismip_mask.shape)
cloudiness_map = np.zeros(ismip_mask.shape)
albedo_map = np.zeros(ismip_mask.shape)
allsky_sw_map = np.zeros(ismip_mask.shape)
allsky_lw_map = np.zeros(ismip_mask.shape)
clrsky_sw_map = np.zeros(ismip_mask.shape)
clrsky_lw_map = np.zeros(ismip_mask.shape)

for i in years:
    
    # Define LW
    lw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/MYD06_LW_Fluxes_' + str(i) + '.nc')
    
    # Define SW
    sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/MYD06_SW_Fluxes_' + str(i) + '.nc')
    
    # Define MODIS albedo data
    albedo = sw.variables['albedo'][:]
    albedo[~ismip_mask.astype(bool)] = np.nan
    albedo = albedo / 100
    
    allsky_sw = sw.variables['allsky_sw'][:]
    allsky_lw = lw.variables['allsky_lw'][:]
    clrsky_sw = sw.variables['clrsky_sw'][:]
    clrsky_lw = lw.variables['clrsky_lw'][:]
    
    net_sw = sw.variables['net_sw_corrected'][:]
    
    # Derive CRE SW and LW
    cre_lw = allsky_lw - clrsky_lw
    cre_sw = net_sw * (1 - albedo)
    
    # Derive CRE allwave
    cre = cre_lw + cre_sw
    
    # Derive cloudiness
    cldy_sw = np.abs(net_sw / clrsky_sw)
    
    # Define new mask
    new_mask = np.isfinite(albedo) & (ismip_mask == 1)
    
    # Mask data
    cre[~new_mask] = np.nan
    cre_lw[~new_mask] = np.nan
    cre_sw[~new_mask] = np.nan
    albedo[~new_mask] = np.nan
    cldy_sw[~new_mask] = np.nan
    allsky_sw[~new_mask] = np.nan
    allsky_lw[~new_mask] = np.nan
    clrsky_sw[~new_mask] = np.nan
    clrsky_lw[~new_mask] = np.nan
    
# =============================================================================
#     # Resize for more convenient plotting
#     lons = sw.variables['longitude'][:]
#     lats = sw.variables['latitude'][:]
#     lons = lons[::5,::5]
#     lats = lats[::5,::5]
#     mod_cre_sw = cre_sw[::5, ::5]
#     
#     fig = plt.figure(figsize=(4, 4))
#     v = np.arange(-50, 1, 2)
#     ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
#     plt.contourf(lons, lats, mod_cre_sw, v, transform=ccrs.PlateCarree(), vmin=-50, vmax=1,
#                  cmap='Blues_r')
#     ax.coastlines(resolution='50m', color='black', linewidth=0.5)
#     ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
#     ax.outline_patch.set_edgecolor('white')
#     cbar = plt.colorbar(ticks=[-50, -40, -30, -20, -10, 0, 10])
#     cbar.ax.set_yticklabels([-50, -40, -30, -20, -10, 0, 10]) 
#     cbar.set_label('Mean CRE SW (W m$^{-2}$)', rotation=270, labelpad=12)
#     plt.tight_layout()
# =============================================================================
    
    # Get CRE for different areas
    cre_gre.append(np.nanmean(cre))
    cre_abl.append(np.nanmean(cre[max_snowline]))
    cre_acc.append(np.nanmean(cre[~max_snowline]))
    
    cldy_gre.append(np.nanmean(cldy_sw))
    cldy_abl.append(np.nanmean(cldy_sw[max_snowline]))
    cldy_acc.append(np.nanmean(cldy_sw[~max_snowline]))
    
    alb_gre.append(np.nanmean(albedo))
    alb_abl.append(np.nanmean(albedo[max_snowline]))
    alb_acc.append(np.nanmean(albedo[~max_snowline]))
    
    cre_lw_gre.append(np.nanmean(cre_lw))
    cre_lw_abl.append(np.nanmean(cre_lw[max_snowline]))
    cre_lw_acc.append(np.nanmean(cre_lw[~max_snowline]))
    
    cre_sw_gre.append(np.nanmean(cre_sw))
    cre_sw_abl.append(np.nanmean(cre_sw[max_snowline]))
    cre_sw_acc.append(np.nanmean(cre_sw[~max_snowline]))
    
    allsky_sw_gre.append(np.nanmean(allsky_sw))
    allsky_sw_abl.append(np.nanmean(allsky_sw[max_snowline]))
    allsky_sw_acc.append(np.nanmean(allsky_sw[~max_snowline]))
    
    allsky_lw_gre.append(np.nanmean(allsky_lw))
    allsky_lw_abl.append(np.nanmean(allsky_lw[max_snowline]))
    allsky_lw_acc.append(np.nanmean(allsky_lw[~max_snowline]))
    
    cre_net_map = np.dstack((cre_net_map, cre))
    cre_lw_map = np.dstack((cre_lw_map, cre_lw))
    cre_sw_map = np.dstack((cre_sw_map, cre_sw))
    albedo_map = np.dstack((albedo_map, albedo))
    cloudiness_map = np.dstack((cloudiness_map, cldy_sw))
    allsky_sw_map = np.dstack((allsky_sw_map, allsky_sw))
    allsky_lw_map = np.dstack((allsky_lw_map, allsky_lw))
    
    clrsky_sw_map = np.dstack((clrsky_sw_map, clrsky_sw))
    clrsky_lw_map = np.dstack((clrsky_lw_map, clrsky_lw))

# Remove first layer
cre_net_map = cre_net_map[:, :, 1:]
cre_lw_map = cre_lw_map[:, :, 1:]
cre_sw_map = cre_sw_map[:, :, 1:]
albedo_map = albedo_map[:, :, 1:]
cloudiness_map = cloudiness_map[:, :, 1:]
allsky_sw_map = allsky_sw_map[:, :, 1:]
allsky_lw_map = allsky_lw_map[:, :, 1:]
clrsky_sw_map = clrsky_sw_map[:, :, 1:]
clrsky_lw_map = clrsky_lw_map[:, :, 1:]

df = pd.DataFrame([cre_gre, cre_abl, cre_acc, cldy_gre, cldy_abl, cldy_acc,
                   cre_sw_gre, cre_sw_abl, cre_sw_acc, cre_lw_gre, cre_lw_abl, 
                   cre_lw_acc, alb_gre, alb_abl, alb_acc, allsky_sw_gre,
                   allsky_sw_abl, allsky_sw_acc, allsky_lw_gre, allsky_lw_abl,
                   allsky_lw_acc]).T
df.columns = ['cre_all', 'cre_abl', 'cre_acc', 'cldy_all', 'cldy_abl', 'cldy_acc', 
              'cre_sw_all', 'cre_sw_abl', 'cre_sw_acc', 'cre_lw_all', 'cre_lw_abl', 
              'cre_lw_acc', 'alb_all', 'alb_abl', 'alb_acc', 'allsky_sw_all',
              'allsky_sw_abl', 'allsky_sw_acc', 'allsky_lw_all', 'allsky_lw_abl',
              'allsky_lw_acc']
df.insert(0, 'year', years)

# Save to csv
df.to_csv(dest + 'cre_spreadsheet.csv')

###############################################################################
# Save 1 km dataset to NetCDF
###############################################################################
dataset = netCDF4.Dataset(dest + 'final_climatologies.nc', 
                          'w', format='NETCDF4_CLASSIC')
print('Creating... %s' % dest + 'final_climatologies.nc')
dataset.Title = "Net CRE, SW CRE, LW CRE, albedo and cloudiness from MODIS"
import time
dataset.History = "Created " + time.ctime(time.time())
dataset.Projection = "WGS 84"
dataset.Reference = "Ryan, J. C., Smith. L. C., Cooley, S. W., and Pearson, B. (in review), Emerging importance of clouds for Greenland Ice Sheet energy balance and meltwater production."
dataset.Contact = "jryan4@uoregon.edu"
    
# Create new dimensions
lat_dim = dataset.createDimension('y', ismip_lat.shape[0])
lon_dim = dataset.createDimension('x', ismip_lat.shape[1])
data_dim = dataset.createDimension('z', cre_net_map.shape[2])

# Define variable types
Y = dataset.createVariable('latitude', np.float32, ('y','x'))
X = dataset.createVariable('longitude', np.float32, ('y','x'))
    
# Define units
Y.units = "degrees"
X.units = "degrees"
   
# Create the actual 3D variable
cre_nc = dataset.createVariable('cre', np.float32, ('y','x','z'))
cre_sw_nc = dataset.createVariable('cre_sw', np.float32, ('y','x','z'))
cre_lw_nc = dataset.createVariable('cre_lw', np.float32, ('y','x','z'))
albedo_nc = dataset.createVariable('albedo', np.float32, ('y','x','z'))
cloudiness_nc = dataset.createVariable('cloudiness', np.float32, ('y','x','z'))
allsky_sw_nc = dataset.createVariable('allsky_sw', np.float32, ('y','x','z'))
allsky_lw_nc = dataset.createVariable('allsky_lw', np.float32, ('y','x','z'))
clrsky_sw_nc = dataset.createVariable('clrsky_sw', np.float32, ('y','x','z'))
clrsky_lw_nc = dataset.createVariable('clrsky_lw', np.float32, ('y','x','z'))

# Write data to layers
Y[:] = ismip_lat
X[:] = ismip_lon
cre_nc[:] = cre_net_map
cre_lw_nc[:] = cre_lw_map
cre_sw_nc[:] = cre_sw_map
albedo_nc[:] = albedo_map
cloudiness_nc[:] = cloudiness_map
allsky_sw_nc[:] = allsky_sw_map
allsky_lw_nc[:] = allsky_lw_map
clrsky_sw_nc[:] = clrsky_sw_map
clrsky_lw_nc[:] = clrsky_lw_map

print('Writing data to %s' % dest + 'final_climatologies.nc')
    
# Close dataset
dataset.close()



















