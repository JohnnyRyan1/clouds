#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary figures

"""

# Import modules
import pandas as pd
import numpy as np
import netCDF4
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter
import glob
from functions import hdf_read
import pyresample
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Transformer

# Import data
df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/CRE_Regional_Spreadsheet.csv')
era = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_summer_climatologies.nc')
sw_preds = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/sw_prediction_results.csv')
lw_preds = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/lw_prediction_results.csv')
era_aws = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Hourly_t2m.csv')
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

# Define maximum snowline
snowline_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_2012.nc')
snowline = snowline_file.variables['bare_ice'][1, :, :].filled(np.nan)
max_snowline = (snowline > 0.1)
mask = snowline_file.variables['mask'][:].astype('bool')

# Define years
n = np.arange(2003, 2021, 1)

# Define CloudSat
files = glob.glob('/home/johnny/Documents/Clouds/Data/1_Merged_Data/*')

# Read data
data = hdf_read.data_read_machine_learning(files)

# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]

# Define region mask
regions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/Ultimate_Mask.nc')
regions_lon = regions.variables['x'][:]
regions_lat = regions.variables['y'][:]
regions_1 = regions.variables['North'][:]
regions_2 = regions.variables['NorthEast'][:]
regions_3 = regions.variables['East'][:]
regions_4 = regions.variables['SouthEast'][:]
regions_5 = regions.variables['South'][:]
regions_6 = regions.variables['SouthWest'][:]
regions_7 = regions.variables['West'][:]
regions_8 = regions.variables['NorthWest'][:]

regions_2[regions_2 > 0] = 2
regions_3[regions_3 > 0] = 3
regions_4[regions_4 > 0] = 4
regions_5[regions_5 > 0] = 5
regions_6[regions_6 > 0] = 6
regions_7[regions_7 > 0] = 7
regions_8[regions_8 > 0] = 8

regions_mask = regions_1 + regions_2 + regions_3 + regions_4 + regions_5 + regions_6 +\
    regions_7 + regions_8
    
# Convert from stereographic to WGS84
transformer = Transformer.from_crs(3413, 4326)
lat, lon = transformer.transform(regions_lon, regions_lat)

# Define grid using pyresample       
grid = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
swath = pyresample.geometry.GridDefinition(lons=lon, lats=lat)

# Determine nearest (w.r.t. great circle distance) neighbour in the grid.
region_swath = pyresample.kd_tree.resample_nearest(source_geo_def=swath, 
                                             target_geo_def=grid, 
                                             data=regions_mask, 
                                             radius_of_influence=50000)


###############################################################################
# # Figure S2: Interannual variation in cloudiness and surface albedo in the 
# accumulation and ablation zone.
###############################################################################

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(13, 6))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Axis 1
ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.plot(n, df['cldy_abl'], color=c1, zorder=2, lw=3, alpha=0.8, label='')
#ax1.set_ylim(0, 10.5)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(n[::2])
ax1.set_xticklabels(n[::2])

# Axis 2
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.plot(n, df['cldy_acc'], color=c2, zorder=2, lw=3, alpha=0.8, label='')
#ax2.set_ylim(0, 10.5)
ax2.set_ylabel('Cloudiness (%)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xticks(n[::2])
ax2.set_xticklabels(n[::2])
ax2.set_ylim(0.09, 0.17)

# Axis 3
ax3.grid(linestyle='dotted', lw=1, zorder=1)
ax3.plot(n, df['alb_abl'], color=c1, zorder=2, lw=3, alpha=0.8, label='')
ax3.set_ylim(0.56, 0.68)
ax3.set_ylabel('Surface albedo (unitless)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xticks(n[::2])
ax3.set_xticklabels(n[::2])

# Axis 4
ax4.grid(linestyle='dotted', lw=1, zorder=1)
ax4.plot(n, df['alb_acc'], color=c2, zorder=2, lw=3, alpha=0.8, label='')
#ax4.set_ylim(0, 10.5)
ax4.set_ylabel('Surface albedo (unitless)', fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xticks(n[::2])
ax4.set_xticklabels(n[::2])
ax4.set_ylim(0.72, 0.84)

ax1.text(0.03, 0.85, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.03, 0.85, "b", fontsize=24, transform=ax2.transAxes)
ax3.text(0.03, 0.85, "c", fontsize=24, transform=ax3.transAxes)
ax4.text(0.03, 0.85, "d", fontsize=24, transform=ax4.transAxes)

ax1.text(0.35, 0.90, "Ablation zone", fontsize=16, transform=ax1.transAxes)
ax2.text(0.3, 0.90, "Accumulation zone", fontsize=16, transform=ax2.transAxes)
ax3.text(0.35, 0.90, "Ablation zone", fontsize=16, transform=ax3.transAxes)
ax4.text(0.3, 0.90, "Accumulation zone", fontsize=16, transform=ax4.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S2_Cloudiness_Albedo.png', dpi=200)

###############################################################################
# Figure S3: (a) MODIS albedo vs. ERA5 air temperature in the ablation zone and,
# (b) MODIS albedo vs. ERA5 air temperature in the accumulation zone
###############################################################################   
era_t = era.variables['t2m'][:]
era_t_abl = []
era_t_acc = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_abl.append(np.nanmean(era_t[:,:,i][max_snowline]))
    era_t_acc.append(np.nanmean(era_t[:,:,i][(~max_snowline & mask)]))
    
baseline_abl = np.mean(era_t_abl[12:])
baseline_acc = np.mean(era_t_acc[12:])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_t_abl - baseline_abl, df['cldy_abl']*100)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(era_t_acc - baseline_acc, df['cldy_acc']*100)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(era_t_abl - baseline_abl, df['alb_abl'])
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(era_t_acc - baseline_acc, df['alb_acc'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_t_abl - baseline_abl, df['cldy_abl']*100, color=c2, zorder=2, s=100, alpha=0.8)
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(era_t_acc - baseline_acc, df['cldy_acc']*100, color=c2, zorder=2, s=100, alpha=0.8)
ax3.grid(linestyle='dotted', lw=1, zorder=1)
ax3.scatter(era_t_abl - baseline_abl, df['alb_abl'], color=c2, zorder=2, s=100, alpha=0.8)
ax4.grid(linestyle='dotted', lw=1, zorder=1)
ax4.scatter(era_t_acc - baseline_acc, df['alb_acc'], color=c2, zorder=2, s=100, alpha=0.8)

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(12, 20)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(9, 17)
ax3.set_xlim(-1.5, 1.5)
ax3.set_ylim(0.56, 0.68)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(0.72, 0.84)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'y = (%.2f * x) + %.2f' % (slope2, intercept2),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value3**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax3.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'y = (%.2f * x) + %.2f' % (slope3, intercept3),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax3.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value4**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'y = (%.2f * x) + %.2f' % (slope4, intercept4),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax2.set_ylabel('Cloudiness (%)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax3.set_ylabel('Albedo (unitless)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax4.set_ylabel('Albedo (unitless)', fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.text(0.3, 1.01, "Ablation zone", fontsize=16, transform=ax1.transAxes)
ax2.text(0.3, 1.01, "Accumulation zone", fontsize=16, transform=ax2.transAxes)
ax3.text(0.3, 1.01, "Ablation zone", fontsize=16, transform=ax3.transAxes)
ax4.text(0.3, 1.01, "Accumulation zone", fontsize=16, transform=ax4.transAxes)

ax1.text(0.03, 0.89, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.03, 0.89, "b", fontsize=24, transform=ax2.transAxes)
ax3.text(0.03, 0.89, "c", fontsize=24, transform=ax3.transAxes)
ax4.text(0.03, 0.89, "d", fontsize=24, transform=ax4.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S3_T_vs_Albedo.svg', dpi=200)

###############################################################################
# Figure S4: Relationships used for computing downward shortwave and longwave clear-sky radiation.
###############################################################################

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(data['ssrdc']/3600, data['sw_cs'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(data['strdc']/3600, data['lw_cs'])

mae1 = np.mean(np.abs(data['sw_cs'] - data['ssrdc']/3600))
mae2 = np.mean(np.abs(data['lw_cs'] - data['strdc']/3600))

# Generate points for linear relationship
x_sw = np.arange(data['ssrdc'].min()/3600, data['ssrdc'].max()/3600, 1)
y_sw = (slope1*x_sw) + intercept1

x_lw = np.arange(data['strdc'].min()/3600, data['strdc'].max()/3600, 1)
y_lw = (slope2*x_lw) + intercept2

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.hist2d(data['ssrdc']/3600, data['sw_cs'], bins=(70, 70), cmap=plt.cm.BuPu)
ax1.plot(x_sw, y_sw, color='k', lw=2, ls='dashed')
ax2.hist2d(data['strdc']/3600, data['lw_cs'], bins=(70, 70), cmap=plt.cm.BuPu)
ax2.plot(x_lw, y_lw, color='k', lw=2, ls='dashed')

ax1.set_xlim(150, 900)
ax1.set_ylim(150, 900)
ax2.set_xlim(150, 280)
ax2.set_ylim(150, 280)

# Add stats
textstr = r'R$^{2}$ = %.2f' % (r_value1**2, )
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

textstr = r'y = (%.2f*x) + %.1f' % (slope1, intercept1, )
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

textstr = r'R$^{2}$ = %.2f' % (r_value2**2, )
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

textstr = r'y = (%.2f*x) + %.1f' % (slope2, intercept2, )
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

ax1.set_ylabel('CloudSat Clear-sky SW$\downarrow$ (W m$^{-2}$)', fontsize=14)
ax1.set_xlabel('ERA5 Clear-sky SW$\downarrow$ (W m$^{-2}$)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.set_ylabel('CloudSat Clear-sky LW$\downarrow$ (W m$^{-2}$)', fontsize=14)
ax2.set_xlabel('ERA5 Clear-sky LW$\downarrow$ (W m$^{-2}$)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

ax1.text(0.05, 0.90, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.05, 0.90, "b", fontsize=24, transform=ax2.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S4_Clearsky_Evaluation.png', dpi=200)

###############################################################################
# Figure S5: Evaluation of cloud enhancement factor (F) uncertainty. 
###############################################################################
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(sw_preds['y_test'], sw_preds['predictions'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(lw_preds['y_test'], lw_preds['predictions'])

# Calculate the absolute errors
mae_sw = abs(sw_preds['y_test'] - sw_preds['predictions']).mean()
mae_lw = abs(lw_preds['y_test'] - lw_preds['predictions']).mean()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.hist2d(sw_preds['y_test'], sw_preds['predictions'], bins=(70, 70), cmap=plt.cm.BuPu)
ax1.plot([0.2,1.0], [0.2,1.0], color='k', lw=2, ls='dashed')

ax2.hist2d(lw_preds['y_test'], lw_preds['predictions'], bins=(170, 170), cmap=plt.cm.BuPu)
ax2.plot([1,1.6], [1,1.6], color='k', lw=2, ls='dashed')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),
    r'MAE = %.3f' % (mae_sw, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),
    r'MAE = %.3f' % (mae_lw, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

ax1.set_ylabel('Predicted F$_{SW}$', fontsize=14)
ax1.set_xlabel('Observed F$_{SW}$', fontsize=14)
ax1.set_xlim(0.3,1.0)
ax1.set_ylim(0.3,1.0)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.set_ylabel('Predicted F$_{LW}$', fontsize=14)
ax2.set_xlabel('Observed F$_{LW}$', fontsize=14)
ax2.set_xlim(1.2, 1.5)
ax2.set_ylim(1.2, 1.5)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax1.text(0.05, 0.90, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.05, 0.90, "b", fontsize=24, transform=ax2.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S5_F_Evaluation.png', dpi=200)

###############################################################################
# Figure S6: Map showing average number of MODIS retrievals across the ice sheet per day.
###############################################################################
# Define ice sheet grid
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:] > 0.9

grid_count = np.zeros(ismip_lon.shape)
# Get a list of all lat/lons for 5 days in August
sample = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_LW/MYD06_L2.A2018*.csv'))
for i in range(len(sample)):
    
    print('Resampling %.0f out of %.0f' %(i+1, len(sample)))
    # Read csv
    coords = pd.read_csv(sample[i])
    coords['count'] = 1
    
    # Define source and target grids
    swath_con = pyresample.geometry.GridDefinition(lons=ismip_lon, lats=ismip_lat)
    swath_def = pyresample.geometry.SwathDefinition(coords['lon'].values, coords['lat'].values)

    result = pyresample.kd_tree.resample_nearest(source_geo_def=swath_def, 
                                                       target_geo_def=swath_con, 
                                                       data=coords['count'].values, 
                                                       radius_of_influence=5000)

    grid_count = grid_count + result

count_per_day = grid_count / 92
count_per_day[~ismip_mask] = np.nan

# Resize for more convenient plotting
count_per_day = count_per_day[::5,::5]
lons = ismip_lon[::5,::5]
lats = ismip_lat[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(0, 7, 0.25)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, count_per_day, v, transform=ccrs.PlateCarree(), vmin=0, vmax=7,
             cmap='viridis')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5, 6])
cbar.ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6]) 
cbar.set_label('Sample count per day', rotation=270, labelpad=12)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S6.png', dpi=200)
























