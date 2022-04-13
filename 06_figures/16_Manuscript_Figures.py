#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Manuscript figures

"""

# Import modules
import pandas as pd
import numpy as np
import netCDF4
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FormatStrFormatter

# Define filepath 
fp = '/Users/jryan4/Dropbox (University of Oregon)/projects/clouds/data/'

# Import data
df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/CRE_Regional_Spreadsheet.csv')
era = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_summer_climatologies.nc')
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')
cmip = netCDF4.Dataset(fp + 'cmip6/GIS_Atmospheric_Temperature_Data.nc')
ele = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')

# Import blocking events
# SW
blocking_sw_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')
blocking_sw_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NW.nc')
blocking_sw_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SE.nc')
blocking_sw_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NE.nc')

# LW
blocking_lw_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SW.nc')
blocking_lw_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NW.nc')
blocking_lw_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_SE.nc')
blocking_lw_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_LW_Fluxes_NE.nc')

# Define maximum snowline
snowline_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_2012.nc')
snowline = snowline_file.variables['bare_ice'][1, :, :].filled(np.nan)
max_snowline = (snowline > 0.1)
mask = snowline_file.variables['mask'][:].astype('bool')

# Define years
n = np.arange(2003, 2021, 1)

###############################################################################
# Fig. 2: (a) Allsky SW vs. Cloudiness and (b) Allsky LW vs. Cloudiness
###############################################################################   
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df['cldy_acc']*100, df['allsky_sw_acc']* (1 - 0.78))
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df['cldy_abl']*100, df['allsky_sw_abl']* (1 - 0.62))
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df['cldy_acc']*100, df['allsky_lw_acc'])
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(df['cldy_abl']*100, df['allsky_lw_abl'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(df['cldy_acc']*100, df['allsky_sw_acc'] * (1 - 0.78), color=c2, zorder=2, s=75, alpha=0.8)
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(df['cldy_abl']*100, df['allsky_sw_abl'] * (1 - 0.62), color=c2, zorder=2, s=75, alpha=0.8)
ax3.grid(linestyle='dotted', lw=1, zorder=1)
ax3.scatter(df['cldy_acc']*100, df['allsky_lw_acc'], color=c2, zorder=2, s=75, alpha=0.8)
ax4.grid(linestyle='dotted', lw=1, zorder=1)
ax4.scatter(df['cldy_abl']*100, df['allsky_lw_abl'], color=c2, zorder=2, s=75, alpha=0.8)

ax1.set_xlim(9, 17)
ax1.set_ylim(63, 80)
ax2.set_xlim(12, 20)
ax2.set_ylim(109, 126)

ax3.set_xlim(9, 17)
ax3.set_ylim(219, 236)
ax4.set_xlim(12, 20)
ax4.set_ylim(248, 265)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'y = (%.2f * x) + %.2f' % (slope1, intercept1),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
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
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax3.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value4**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'y = (%.2f * x) + %.2f' % (slope4, intercept4),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)

ax1.set_xlabel('Cloudiness (%)', fontsize=14)
ax1.set_ylabel('SW$_{net}$ (W m$^{-2}$)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xlabel('Cloudiness (%)', fontsize=14)
ax2.set_ylabel('SW$_{net}$ (W m$^{-2}$)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax3.set_xlabel('Cloudiness (%)', fontsize=14)
ax3.set_ylabel('Downward LW (W m$^{-2}$)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax4.set_xlabel('Cloudiness (%)', fontsize=14)
ax4.set_ylabel('Downward LW (W m$^{-2}$)', fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax1.text(0.03, 0.89, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.03, 0.89, "b", fontsize=24, transform=ax2.transAxes)
ax3.text(0.03, 0.89, "c", fontsize=24, transform=ax3.transAxes)
ax4.text(0.03, 0.89, "d", fontsize=24, transform=ax4.transAxes)

ax1.text(0.3, 1.01, "Accumulation zone", fontsize=16, transform=ax1.transAxes)
ax2.text(0.3, 1.01, "Ablation zone", fontsize=16, transform=ax2.transAxes)
ax3.text(0.35, 1.01, "Accumulation zone", fontsize=16, transform=ax3.transAxes)
ax4.text(0.35, 1.01, "Ablation zone", fontsize=16, transform=ax4.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig2_Allsky_vs_Cloudiness.svg')

###############################################################################
# Fig. 4: Future SWnet response to cloudiness
############################################################################### 
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][max_snowline]))

baseline = np.mean(era_t_values[12:])
data_current = pd.DataFrame(list(zip(n, era_t_values-baseline)), columns=['year', 'T'])
data_current['albedo'] = (-0.05 * data_current['T']) + 0.63

cmip_time = np.nanmean(cmip.variables['TIME'][:].filled(), axis=1)
summer_months = []
for i in range(cmip_time.shape[0]):
    decimal_diff = cmip_time[i] -  int(cmip_time[i])
    if (decimal_diff > 0.4) & (decimal_diff < 0.6):
        summer_months.append(i)

n_future = np.arange(2015, 2101, 1)

def cmip_read(variable):
    
    cmip_t = np.nanmean(cmip.variables[variable][summer_months, :].filled(), axis=1)
    summer = np.mean(cmip_t.reshape(-1, 3), axis=1)
    baseline = summer[0:6].mean()
    summer_norm = summer - baseline
    
    cmip_range = cmip.variables[variable][summer_months, :].filled()
    cmip_iqr = np.percentile(cmip_range, [75 ,25], axis=1)
    cmip_lower = np.mean(cmip_iqr[0].reshape(-1, 3), axis=1) - baseline
    cmip_higher = np.mean(cmip_iqr[1].reshape(-1, 3), axis=1) - baseline
    
    return summer_norm, cmip_higher, cmip_lower

ssp585_mean, ssp585_lower, ssp585_higher  = cmip_read('T_GIS_SSP585')
ssp370_mean, ssp370_lower, ssp370_higher = cmip_read('T_GIS_SSP370')
ssp245_mean, ssp245_lower, ssp245_higher = cmip_read('T_GIS_SSP245')
ssp126_mean, ssp126_lower, ssp126_higher = cmip_read('T_GIS_SSP126')

#data_current = pd.DataFrame(list(zip(n, areas_percent)), columns=['year', 'area'])
data_future = pd.DataFrame(list(zip(n_future, ssp585_mean, ssp370_mean, 
                                    ssp245_mean, ssp126_mean,
                                    ssp585_lower, ssp585_higher, ssp370_lower,
                                    ssp370_higher, ssp245_lower, ssp245_higher,
                                    ssp126_lower, ssp126_higher)), 
                           columns=['year', 'ssp585_mean', 'ssp370_mean', 
                                    'ssp245_mean', 'ssp126_mean', 'ssp585_lower',
                                    'ssp585_higher', 'ssp370_lower', 'ssp370_higher',
                                    'ssp245_lower', 'ssp245_higher', 'ssp126_lower',
                                    'ssp126_higher'])

data_future['ssp585_albedo_mean'] = (-0.05 * ssp585_mean) + 0.63
data_future['ssp585_albedo_low'] = (-0.05 * ssp585_lower) + 0.63
data_future['ssp585_albedo_high'] = (-0.05 * ssp585_higher) + 0.63

data_future['ssp245_albedo_mean'] = (-0.05 * ssp245_mean) + 0.63
data_future['ssp245_albedo_low'] = (-0.05 * ssp245_lower) + 0.63
data_future['ssp245_albedo_high'] = (-0.05 * ssp245_higher) + 0.63


data_future['ssp126_albedo_mean'] = (-0.05 * ssp126_mean) + 0.63
data_future['ssp126_albedo_low'] = (-0.05 * ssp126_lower) + 0.63
data_future['ssp126_albedo_high'] = (-0.05 * ssp126_higher) + 0.63

def find_slopes(albedos):
    
    slopes = []
    for i in range(albedos.shape[0]):
        slope, inte, r, p, se = stats.linregress(df['cldy_abl']*100, df['allsky_sw_abl']* (1 - albedos.iloc[i]))
        slopes.append(slope)
    return np.abs(np.array(slopes))

data_current['slope_mean'] = find_slopes(data_current['albedo'])

data_future['ssp585_slope_mean'] = find_slopes(data_future['ssp585_albedo_mean'])
data_future['ssp585_slope_low'] = find_slopes(data_future['ssp585_albedo_low'])
data_future['ssp585_slope_high'] = find_slopes(data_future['ssp585_albedo_high'])

data_future['ssp245_slope_mean'] = find_slopes(data_future['ssp245_albedo_mean'])
data_future['ssp245_slope_low'] = find_slopes(data_future['ssp245_albedo_low'])
data_future['ssp245_slope_high'] = find_slopes(data_future['ssp245_albedo_high'])

data_future['ssp126_slope_mean'] = find_slopes(data_future['ssp126_albedo_mean'])
data_future['ssp126_slope_low'] = find_slopes(data_future['ssp126_albedo_low'])
data_future['ssp126_slope_high'] = find_slopes(data_future['ssp126_albedo_high'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c3 = '#616E96'
c2 = '#F8A557'
c4 = '#3CBEDD'

ax1.plot(data_future['year'], data_future['ssp126_slope_mean'], color=c4, lw=3, 
         zorder=2, label='SSP1-2.6')
ax1.fill_between(data_future['year'], data_future['ssp126_slope_low'], 
                 data_future['ssp126_slope_high'], color=c4, alpha=0.5)

ax1.plot(data_future['year'], data_future['ssp245_slope_mean'], color=c2, lw=3, 
         zorder=2, label='SSP2-4.5')
ax1.fill_between(data_future['year'], data_future['ssp245_slope_low'], 
                 data_future['ssp245_slope_high'], color=c2, alpha=0.5)

ax1.plot(data_future['year'], data_future['ssp585_slope_mean'], color=c1, lw=3, 
         zorder=2, label='SSP5-8.5')
ax1.fill_between(data_future['year'], data_future['ssp585_slope_low'], 
                 data_future['ssp585_slope_high'], color=c1, alpha=0.5)

ax1.grid(linestyle='dotted', lw=1.5, zorder=0)
ax1.plot(data_current['year'], data_current['slope_mean'], color='k', lw=3, zorder=2)
ax1.fill_between(data_current['year'],  data_current['slope_mean'] +  data_current['slope_mean']*0.20, 
                 data_current['slope_mean'] - data_current['slope_mean']*0.20, color='k', alpha=0.3)

#ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('SW$_{net}$ sensitivity to to $\Delta$ cloudiness \n in the ablation zone', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc=1)
ax1.set_xlim(2000, 2100)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig3_Future_SWnet.png', dpi=200)

###############################################################################
# Table S1
############################################################################### 

def produce_table(block_file_sw):
    block_cldy = np.nanmean((block_file_sw.variables['cloudiness'][:][max_snowline]) * 100)
    blk_sw_current = np.nanmean((block_file_sw.variables['allsky_sw'][:][max_snowline]) * (1 - 0.62))
    blk_sw_future_1 = np.nanmean((block_file_sw.variables['allsky_sw'][:][max_snowline]) * (1 - 0.57))
    blk_sw_future_2 = np.nanmean((block_file_sw.variables['allsky_sw'][:][max_snowline]) * (1 - 0.50))
    blk_sw_future_5 = np.nanmean((block_file_sw.variables['allsky_sw'][:][max_snowline]) * (1 - 0.35))
    return block_cldy, blk_sw_current, blk_sw_future_1, blk_sw_future_2, blk_sw_future_5    
    
# Get mean cloudiness
mod_cldy = np.nanmean((np.nanmean(mod.variables['cloudiness'][:], axis=2)[max_snowline]) * 100)

# Get mean radiative flux values
mean_sw_current = np.nanmean((np.nanmean(mod.variables['allsky_sw'][:], axis=2)[max_snowline]) * (1 - 0.62))
mean_sw_future_1 = np.nanmean((np.nanmean(mod.variables['allsky_sw'][:], axis=2)[max_snowline]) * (1 - 0.57))
mean_sw_future_2 = np.nanmean((np.nanmean(mod.variables['allsky_sw'][:], axis=2)[max_snowline]) * (1 - 0.50))
mean_sw_future_5 = np.nanmean((np.nanmean(mod.variables['allsky_sw'][:], axis=2)[max_snowline]) * (1 - 0.35))
mean_lw_current = np.nanmean(np.nanmean(mod.variables['allsky_lw'][:], axis=2)[max_snowline])

# Get block longwave radiative fluxes
blk_lw_sw_current = blocking_lw_sw.variables['allsky_lw'][:][max_snowline]
blk_lw_nw_current = blocking_lw_nw.variables['allsky_lw'][:][max_snowline]
blk_lw_se_current = blocking_lw_se.variables['allsky_lw'][:][max_snowline]
blk_lw_ne_current = blocking_lw_ne.variables['allsky_lw'][:][max_snowline]

# Join LW blocking events
blk_lw_current_all = np.nanmean(np.nanmean((blk_lw_sw_current, blk_lw_nw_current, 
                                            blk_lw_se_current, blk_lw_ne_current), axis=0))

# Get SW blocking events
block_sw = produce_table(blocking_sw_sw)
block_nw = produce_table(blocking_sw_nw)
block_ne = produce_table(blocking_sw_ne)
block_se = produce_table(blocking_sw_se)

# Get radiative difference
blk_southwest_current = (block_sw[1] + blk_lw_current_all) - (mean_sw_current + mean_lw_current)
blk_northwest_current = (block_nw[1] + blk_lw_current_all) - (mean_sw_current + mean_lw_current)
blk_northeast_current = (block_ne[1] + blk_lw_current_all) - (mean_sw_current + mean_lw_current)
blk_southeast_current = (block_se[1] + blk_lw_current_all) - (mean_sw_current + mean_lw_current)

blk_southwest_future_1 = (block_sw[2] + blk_lw_current_all) - (mean_sw_future_1 + mean_lw_current)
blk_northwest_future_1 = (block_nw[2] + blk_lw_current_all) - (mean_sw_future_1 + mean_lw_current)
blk_northeast_future_1 = (block_ne[2] + blk_lw_current_all) - (mean_sw_future_1 + mean_lw_current)
blk_southeast_future_1 = (block_se[2] + blk_lw_current_all) - (mean_sw_future_1 + mean_lw_current)

blk_southwest_future_2 = (block_sw[3] + blk_lw_current_all) - (mean_sw_future_2 + mean_lw_current)
blk_northwest_future_2 = (block_nw[3] + blk_lw_current_all) - (mean_sw_future_2 + mean_lw_current)
blk_northeast_future_2 = (block_ne[3] + blk_lw_current_all) - (mean_sw_future_2 + mean_lw_current)
blk_southeast_future_2 = (block_se[3] + blk_lw_current_all) - (mean_sw_future_2 + mean_lw_current)

blk_southwest_future_5 = (block_sw[4] + blk_lw_current_all) - (mean_sw_future_5 + mean_lw_current)
blk_northwest_future_5 = (block_nw[4] + blk_lw_current_all) - (mean_sw_future_5 + mean_lw_current)
blk_northeast_future_5 = (block_ne[4] + blk_lw_current_all) - (mean_sw_future_5 + mean_lw_current)
blk_southeast_future_5 = (block_se[4] + blk_lw_current_all) - (mean_sw_future_5 + mean_lw_current)

cloudiness = list((block_sw[0]-mod_cldy, block_nw[0]-mod_cldy, block_ne[0]-mod_cldy, block_se[0]-mod_cldy))
blk_current = list((blk_southwest_current, blk_northwest_current,
                    blk_northeast_current, blk_southeast_current))
blk_future_1 = list((blk_southwest_future_1, blk_northwest_future_1,
                    blk_northeast_future_1, blk_southeast_future_1))
blk_future_2 = list((blk_southwest_future_2, blk_northwest_future_2,
                    blk_northeast_future_2, blk_southeast_future_2))
blk_future_5 = list((blk_southwest_future_5, blk_northwest_future_5,
                    blk_northeast_future_5, blk_southeast_future_5))

blk_df = pd.DataFrame((cloudiness, blk_current, blk_future_1, blk_future_2, blk_future_5))
blk_df.columns = ['Southwest', 'Northwest', 'Northeast', 'Southeast']

###############################################################################
# Some extra statistics
############################################################################### 
mod_cre = mod.variables['cre'][:]
areas_percent = []
for i in range(18):
    areas_percent.append((mod_cre[:,:,i] < 0).sum() / max_snowline.sum() * 100)
    
print('Area of ablation zone with negative CRE = %.1f' % (np.mean(areas_percent)))











