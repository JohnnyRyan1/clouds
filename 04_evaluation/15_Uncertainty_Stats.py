#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Compute uncertainties

"""

# Import modules
import pandas as pd
import numpy as np
import netCDF4

# Define years
years = np.arange(2003, 2021, 1)

# Define f uncertainties
f_lw_mean = 1.37
f_lw_uncert = 0.036
lw_clrsky_uncert = 13.7 # W m-2

f_sw_mean = 0.64
f_sw_uncert = 0.075
sw_clrsky_uncert = 20.8 # W m-2
albedo_uncert = 0.03

# Define ice sheet grid
ismip = netCDF4.Dataset(path + 'masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

sw_clrsky_mean = []
lw_clrsky_mean = []

cre_lw_uncert = []
cre_sw_uncert = []
cldy_uncert = []

cre_lw_mean = []
cre_sw_mean = []
cre_mean = []
cldy_mean = []

for i in years:
    print(i)
    
    # Define LW
    lw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/MYD06_LW_Fluxes_' + str(i) + '.nc')
    
    # Define SW
    sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies/MYD06_SW_Fluxes_' + str(i) + '.nc')
    
    # Derive LW uncertainties
    lw_allsky = lw.variables['allsky_lw'][:]
    lw_allsky[~ismip_mask.astype(bool)] = np.nan
    
    lw_clrsky = lw.variables['clrsky_lw'][:]
    lw_clrsky[~ismip_mask.astype(bool)] = np.nan
    
    lw_clrsky_mean.append(np.nanmean(lw_clrsky))
    cre_lw_mean.append(np.nanmean(lw_allsky - lw_clrsky))
    
    lw_clrsky_uncert = lw_uncert / np.nanmean(lw_clrsky)
    lw_allsky_uncert = lw_clrsky_uncert + (f_lw_uncert / f_lw_mean)
    cre_lw_uncert.append(lw_clrsky_uncert + lw_allsky_uncert)
    
    # Derive SW uncertainties
    sw_allsky = sw.variables['allsky_sw'][:]
    sw_allsky[~ismip_mask.astype(bool)] = np.nan
    
    sw_clrsky = sw.variables['clrsky_sw'][:]
    sw_clrsky[~ismip_mask.astype(bool)] = np.nan
    
    albedo = sw.variables['albedo'][:]
    albedo[~ismip_mask.astype(bool)] = np.nan
    albedo = albedo / 100
    
    sw_allsky_uncert = (f_sw_uncert / f_sw_mean)
    cre_sw_uncert.append(sw_allsky_uncert + (albedo_uncert / (1 - np.nanmean(albedo))))
    cre_sw_mean.append(np.nanmean((sw_allsky - sw_clrsky)*(1-albedo)))
    
    # Derive cloudiness uncertainties
    cldy_uncert.append(sw_allsky_uncert + sw_allsky_uncert)
    

df = pd.DataFrame([cre_lw_uncert, cre_sw_uncert, cldy_uncert]).T
df.columns = ['cre_lw_uncertainty', 'cre_sw_uncertainty', 'cldy_uncertainty']
df.insert(0, 'year', years)
df.insert(1, 'cre_uncertainty', np.sqrt(df['cre_lw_uncertainty']**2 + df['cre_sw_uncertainty']**2))

# Print results
print('CRE SW uncertainty = %0.1f %%' %(df['cre_sw_uncertainty'].mean() * 100))
print('CRE LW uncertainty = %0.1f %%' %(df['cre_lw_uncertainty'].mean() * 100))
print('CRE uncertainty = %0.1f %%' %(df['cre_uncertainty'].mean() * 100))
print('Cloudiness uncertainty = %0.1f %%' %(df['cldy_uncertainty'].mean() * 100))
















