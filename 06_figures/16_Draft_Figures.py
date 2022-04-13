#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Draft figures for paper

"""

# Import modules
import pandas as pd
import numpy as np
import netCDF4
import glob
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.gridspec import GridSpec

# Import data
df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/CRE_Regional_Spreadsheet.csv')
era = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_summer_climatologies.nc')
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')
cmip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/CMIP6/GIS_Atmospheric_Temperature_Data.nc')

# Define maximum snowline
snowline_file = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/SciAdv_Products/Monthly_Bare_Ice_2012.nc')
snowline = snowline_file.variables['bare_ice'][1, :, :].filled(np.nan)
max_snowline = (snowline > 0.1)
mask = snowline_file.variables['mask'][:].astype('bool')

# Define years
n = np.arange(2003, 2021, 1)
###############################################################################
# Plot interannual CRE in ablation zone
###############################################################################
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][max_snowline]))

baseline = np.mean(era_t_values[12:])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_t_values - baseline, df['cre_abl'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(era_t_values - baseline, df['cre_sw_abl'])
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(era_t_values - baseline, df['cre_lw_abl'])

fig = plt.figure(figsize=(10, 6))

gs = GridSpec(2,3) # 2 rows, 3 columns

ax1 = fig.add_subplot(gs[0,:]) # First row, span all columns
ax2 = fig.add_subplot(gs[1,0]) # Second row, first column
ax3 = fig.add_subplot(gs[1,1]) # Second row, second column
ax4 = fig.add_subplot(gs[1,2]) # Second row, third column

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Axis 1
ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.plot(n, df['cre_abl'], color=c3, zorder=2, lw=3, alpha=0.8, label='')
#ax1.set_ylim(0, 10.5)
ax1.set_ylabel('CRE (W m$^{-2}$)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xticks(n[::2])
ax1.set_xticklabels(n[::2])

# Axis 2
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(era_t_values - baseline, df['cre_abl'], color=c3, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)
ax2.set_ylabel('CRE (W m$^{-2}$)', fontsize=14)
ax2.set_xlabel('Summer air temp. \n anomaly (K)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)

# Axis 3
ax3.grid(linestyle='dotted', lw=1, zorder=1)
ax3.scatter(era_t_values - baseline, df['cre_sw_abl'], color=c2, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax3.add_artist(text_box)
ax3.set_ylabel('CRE SW (W m$^{-2}$)', fontsize=14)
ax3.set_xlabel('Summer air temp.\n anomaly (K)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.set_yticks(np.arange(-33, -20, 3))
ax3.set_yticklabels(np.arange(-33, -20, 3))

# Axis 4
ax4.grid(linestyle='dotted', lw=1, zorder=1)
ax4.scatter(era_t_values - baseline, df['cre_lw_abl'], color=c1, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value3**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)
ax4.set_ylabel('CRE LW (W m$^{-2}$)', fontsize=14)
ax4.set_xlabel('Summer air temp. \n anomaly (K)', fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=12)
ax4.set_yticks(np.arange(26, 40, 3))
ax4.set_yticklabels(np.arange(26, 40, 3))

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/Ablation_Zone_CRE_NET.png', dpi=200)

###############################################################################
# Plot interannual CRE in accumulation zone
###############################################################################
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][(~max_snowline & mask)]))

baseline = np.mean(era_t_values[12:])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_t_values, df['cre_acc'] - baseline)
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(era_t_values, df['cre_sw_acc'] - baseline)
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(era_t_values, df['cre_lw_acc'] - baseline)

fig = plt.figure(figsize=(10, 6))

gs = GridSpec(2,3) # 2 rows, 3 columns

ax1 = fig.add_subplot(gs[0,:]) # First row, span all columns
ax2 = fig.add_subplot(gs[1,0]) # Second row, first column
ax3 = fig.add_subplot(gs[1,1]) # Second row, second column
ax4 = fig.add_subplot(gs[1,2]) # Second row, third column

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Axis 1
ax1.grid(linestyle='dotted', lw=1, zorder=1)
#ax1.plot(n, df['cre_lw_acc'], color=c1, zorder=2, lw=3, alpha=0.8, label='LW')
ax1.plot(n, df['cre_acc'], color=c3, zorder=2, lw=3, alpha=0.8, label='net')
#ax1.plot(n, df['cre_sw_acc'], color=c2, zorder=2, lw=3, alpha=0.8, label='SW')
#ax1.axhline(y=0, ls='dashed', lw=2, color='k', zorder=1)
ax1.set_ylabel('CRE (W m-2)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc=3)

# Axis 2
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(era_t_values, df['cre_acc'], color=c3, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)
ax2.set_ylabel('CRE (W m-2)', fontsize=14)
ax2.set_xlabel('Mean air temp (K)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

# Axis 3
ax3.grid(linestyle='dotted', lw=1, zorder=1)
ax3.scatter(era_t_values, df['cre_sw_acc'], color=c2, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax3.add_artist(text_box)
ax3.set_ylabel('CRE SW (W m-2)', fontsize=14)
ax3.set_xlabel('Mean air temp (K)', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)

# Axis 4
ax4.grid(linestyle='dotted', lw=1, zorder=1)
ax4.scatter(era_t_values, df['cre_lw_acc'], color=c1, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value3**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax4.add_artist(text_box)
ax4.set_ylabel('CRE LW (W m-2)', fontsize=14)
ax4.set_xlabel('Mean air temp (K)', fontsize=14)
ax4.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot a) area of ice sheet that experiences net cloud radiative cooling and b) vs. T
###############################################################################
mod_cre = mod.variables['cre'][:]
areas_percent = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    areas_percent.append((mod_cre[:,:,i] < 0).sum() / mask.sum() * 100)

era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i]))

baseline = np.mean(era_t_values[12:])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_t_values - baseline, areas_percent)

fig = plt.figure(figsize=(10, 4))

gs = GridSpec(1,3) # 1 rows, 3 columns

ax1 = fig.add_subplot(gs[0,0:2]) # First row, span first two columns
ax2 = fig.add_subplot(gs[0,2]) # First row, last column

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Axis 1
ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.plot(n, areas_percent, color=c2, zorder=2, lw=3, alpha=0.8, label='')
ax1.set_ylabel('Fraction of ice sheet with \n negative CRE (%)', fontsize=14)
ax1.set_xlabel('Year', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xticks(n[::2])
ax1.set_xticklabels(n[::2])

# Axis 2
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(era_t_values - baseline, areas_percent, color=c2, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)
ax2.set_ylabel('Fraction of ice sheet with \n negative CRE (%)', fontsize=14)
ax2.set_xlabel('Summer air temp. \n anomaly (K)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/CRE_vs_Area.png', dpi=200)

###############################################################################
# Plot a) melt potential due to clouds and b) vs. T
###############################################################################

"""
29.5 W m-2 provides enough energy to melt 90 Gt of ice in the GrIS ablation
area during July and August (van Tricht et al., 2016)

"""
hof = 333.55 # kJ kg−1

gt_melt_abl = []
for i in range(df.shape[0]):
    # J s-1 * seconds in summer * joules in kilajoules / heat of fusion * m2 to km2 * area of ablation zone / kg to Gt
    gt_melt_abl.append(df['cre_abl'].iloc[i] * (86400 * 92) * 0.001 / hof * 1e+6 * max_snowline.sum() / 1e+12)

era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i]))

-30.661150 * (86400 * 92) * 0.001 / hof * 1e+6 * max_snowline.sum() / 1e+12
32.013912 * (86400 * 92) * 0.001 / hof * 1e+6 * max_snowline.sum() / 1e+12
# =============================================================================
# era_t_uncert = 2.9 / np.mean(era_t_values)
# gt_melt_uncert = np.sqrt(era_t_uncert**2 + 0.351**2)
# =============================================================================

baseline = np.mean(era_t_values[12:])

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_t_values - baseline, gt_melt_abl)

fig = plt.figure(figsize=(10, 4))

gs = GridSpec(1,3) # 1 rows, 3 columns

ax1 = fig.add_subplot(gs[0,0:2]) # First row, span first two columns
ax2 = fig.add_subplot(gs[0,2]) # First row, last column

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

# Axis 1
ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.plot(n, gt_melt_abl, color=c2, zorder=2, lw=3, alpha=0.8, label='')
ax1.set_ylabel('Melt attributed to clouds \n in ablation zone (Gt)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
#ax1.legend(fontsize=14, loc=3)
ax1.set_xlabel('Year', fontsize=14)
ax1.set_xticks(n[::2])
ax1.set_xticklabels(n[::2])

# Axis 2
ax2.grid(linestyle='dotted', lw=1, zorder=1)
ax2.scatter(era_t_values - baseline, gt_melt_abl, color=c2, zorder=2, s=50, alpha=0.8)
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)
ax2.set_ylabel('Melt attributed to clouds \n in ablation zone (Gt)', fontsize=14)
ax2.set_xlabel('Summer air temp. \n anomaly (K)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/CRE_vs_Melt.png', dpi=200)

###############################################################################
# Plot cloudiness vs. T
###############################################################################
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][mask]))

baseline = np.mean(era_t_values[12:])

slope, intercept, r_value, p_value, std_err = stats.linregress(era_t_values - baseline, df['cldy_all'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_t_values - baseline, df['cldy_all']*100, color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n):
    ax1.annotate(txt, (era_t_values[i] - baseline, df['cldy_all'].values[i]*100), fontsize=14)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot cloudiness vs. T in the ablation zone
###############################################################################
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][max_snowline]))

baseline = np.mean(era_t_values[12:])

slope, intercept, r_value, p_value, std_err = stats.linregress(era_t_values - baseline, df['cldy_abl'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_t_values - baseline, df['cldy_abl'] * 100, color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n):
    ax1.annotate(txt, (era_t_values[i] - baseline, df['cldy_abl'].values[i]*100), fontsize=14)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/Cloudiness_vs_T_Ablation_Zone.png', dpi=200)

###############################################################################
# Plot cloudiness vs. T in the accumulation zone
###############################################################################
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][(~max_snowline & mask)]))

baseline = np.mean(era_t_values[12:])

slope, intercept, r_value, p_value, std_err = stats.linregress(era_t_values - baseline, df['cldy_acc']*100)

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_t_values - baseline, df['cldy_acc']*100, color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n):
    ax1.annotate(txt, (era_t_values[i] - baseline, df['cldy_acc'].values[i]*100), fontsize=14)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot MODIS vs ERA CRE in ablation zone
###############################################################################
# Define MODIS albedo data
albedo_files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/Summer_Albedo_Climatologies/*'))

era_cre_sw = era.variables['cre_sw'][:]
era_cre_lw = era.variables['cre_lw'][:]
era_cre = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    y = n[i]
    albedo_match = [s for s in albedo_files if str(y) in s]
    albedo_file = netCDF4.Dataset(albedo_match[0])
    albedo = albedo_file.variables['albedo'][:, :, :].filled(np.nan).astype(float)
    albedo[albedo == 0] = np.nan
    albedo_mean = np.nanmean(albedo, axis=0) / 100
    era_sw_values = era_cre_sw[:,:,i] * (1 - albedo_mean)
    era_cre.append(np.nanmean(era_sw_values[max_snowline]) + np.nanmean(era_cre_lw[:,:,i][max_snowline]))

slope, intercept, r_value, p_value, std_err = stats.linregress(era_cre, df['cre_abl'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_cre, df['cre_abl'], color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n):
    ax1.annotate(txt, (era_cre[i], df['cre_abl'].values[i]), fontsize=14)

ax1.set_xlabel('ERA CRE (W m-2)', fontsize=14)
ax1.set_ylabel('MODIS CRE (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot MODIS vs ERA CRE in accumulation zone
###############################################################################
# Define MODIS albedo data
albedo_files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/Summer_Albedo_Climatologies/*'))

era_cre_sw = era.variables['cre_sw'][:]
era_cre_lw = era.variables['cre_lw'][:]
era_cre = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    y = n[i]
    albedo_match = [s for s in albedo_files if str(y) in s]
    albedo_file = netCDF4.Dataset(albedo_match[0])
    albedo = albedo_file.variables['albedo'][:, :, :].filled(np.nan).astype(float)
    albedo[albedo == 0] = np.nan
    albedo_mean = np.nanmean(albedo, axis=0) / 100
    era_sw_values = era_cre_sw[:,:,i] * (1 - albedo_mean)
    era_cre.append(np.nanmean(era_sw_values[~max_snowline]) + np.nanmean(era_cre_lw[:,:,i][~max_snowline]))

slope, intercept, r_value, p_value, std_err = stats.linregress(era_cre, df['cre_acc'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_cre, df['cre_acc'], color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n):
    ax1.annotate(txt, (era_cre[i], df['cre_acc'].values[i]), fontsize=14)

ax1.set_xlabel('ERA CRE (W m-2)', fontsize=14)
ax1.set_ylabel('MODIS CRE (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot future melt prevented by clouds using CMIP6 data for Greenland
###############################################################################
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
    gt_mean = (summer_norm * -31.16) + 39.28
    
    cmip_range = cmip.variables[variable][summer_months, :].filled()
    cmip_iqr = np.percentile(cmip_range, [75 ,25], axis=1)
    cmip_lower = np.mean(cmip_iqr[0].reshape(-1, 3), axis=1) - baseline
    cmip_higher = np.mean(cmip_iqr[1].reshape(-1, 3), axis=1) - baseline
    
    gt_lower = (cmip_lower * -31.16) + 39.28
    gt_higher = (cmip_higher * -31.16) + 39.28
    
    return gt_mean, gt_lower, gt_higher

ssp585_mean, ssp585_lower, ssp585_higher  = cmip_read('T_GIS_SSP585')
ssp370_mean, ssp370_lower, ssp370_higher = cmip_read('T_GIS_SSP370')
ssp245_mean, ssp245_lower, ssp245_higher = cmip_read('T_GIS_SSP245')
ssp126_mean, ssp126_lower, ssp126_higher = cmip_read('T_GIS_SSP126')

data_current = pd.DataFrame(list(zip(n, gt_melt_abl)), columns=['year', 'melt'])
data_future = pd.DataFrame(list(zip(n_future, ssp585_mean, ssp370_mean, 
                                    ssp245_mean, ssp126_mean,
                                    ssp585_lower, ssp585_higher, ssp370_lower,
                                    ssp370_higher, ssp245_lower, ssp245_higher,
                                    ssp126_lower, ssp126_higher)), 
                           columns=['year', 'melt_ssp585', 'melt_ssp370', 
                                    'melt_ssp245', 'melt_ssp126', 'ssp585_lower',
                                    'ssp585_higher', 'ssp370_lower', 'ssp370_higher',
                                    'ssp245_lower', 'ssp245_higher', 'ssp126_lower',
                                    'ssp126_higher'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c3 = '#616E96'
c2 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1.5, zorder=0)
ax1.axhline(y=0, linestyle='dashed', lw=2, zorder=1, color='k')
ax1.plot(data_current['year'], data_current['melt'], color='k', lw=2, zorder=3, label='Observed')
ax1.plot(data_future['year'], data_future['melt_ssp126'], color=c4, lw=3, zorder=2, label='SSP1-2.6')
ax1.plot(data_future['year'], data_future['melt_ssp245'], color=c3, lw=3, zorder=2, label='SSP2-4.5')
ax1.plot(data_future['year'], data_future['melt_ssp370'], color=c2, lw=3, zorder=2, label='SSP3-7.0')
ax1.plot(data_future['year'], data_future['melt_ssp585'], color=c1, lw=3, zorder=2, label='SSP5-8.5')

ax1.fill_between(data_current['year'], data_current['melt'] - data_current['melt']*0.351, 
                 data_current['melt'] + data_current['melt']*0.351, color='grey', alpha=0.5)
ax1.fill_between(data_future['year'], data_future['ssp585_lower'], data_future['ssp585_higher'],
                 color=c1, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp370_lower'], data_future['ssp370_higher'],
                 color=c2, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp245_lower'], data_future['ssp245_higher'],
                 color=c3, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp126_lower'], data_future['ssp126_higher'],
                 color=c4, alpha=0.3)

#ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Melt attributed to clouds \n in ablation zone (Gt)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc=3)
ax1.set_xlim(2000, 2100)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/Future_Melt.png', dpi=200)

###############################################################################
# Plot area of ice sheet that experiences negative CRE using CMIP6 data for Greenland
###############################################################################
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
    gt_mean = (summer_norm * 2.135) + 3.139
    
    cmip_range = cmip.variables[variable][summer_months, :].filled()
    cmip_iqr = np.percentile(cmip_range, [75 ,25], axis=1)
    cmip_lower = np.mean(cmip_iqr[0].reshape(-1, 3), axis=1) - baseline
    cmip_higher = np.mean(cmip_iqr[1].reshape(-1, 3), axis=1) - baseline
    
    gt_lower = (cmip_lower * 2.135) + 3.139
    gt_higher = (cmip_higher * 2.135) + 3.139
    
    return gt_mean, gt_lower, gt_higher

ssp585_mean, ssp585_lower, ssp585_higher  = cmip_read('T_GIS_SSP585')
ssp370_mean, ssp370_lower, ssp370_higher = cmip_read('T_GIS_SSP370')
ssp245_mean, ssp245_lower, ssp245_higher = cmip_read('T_GIS_SSP245')
ssp126_mean, ssp126_lower, ssp126_higher = cmip_read('T_GIS_SSP126')

data_current = pd.DataFrame(list(zip(n, areas_percent)), columns=['year', 'area'])
data_future = pd.DataFrame(list(zip(n_future, ssp585_mean, ssp370_mean, 
                                    ssp245_mean, ssp126_mean,
                                    ssp585_lower, ssp585_higher, ssp370_lower,
                                    ssp370_higher, ssp245_lower, ssp245_higher,
                                    ssp126_lower, ssp126_higher)), 
                           columns=['year', 'area_ssp585', 'area_ssp370', 
                                    'area_ssp245', 'area_ssp126', 'ssp585_lower',
                                    'ssp585_higher', 'ssp370_lower', 'ssp370_higher',
                                    'ssp245_lower', 'ssp245_higher', 'ssp126_lower',
                                    'ssp126_higher'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c3 = '#616E96'
c2 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1.5, zorder=0)
ax1.plot(data_current['year'], data_current['area'], color='k', lw=2, zorder=3, label='Observed')
ax1.plot(data_future['year'], data_future['area_ssp585'], color=c1, lw=3, zorder=2, label='SSP5-8.5')
ax1.plot(data_future['year'], data_future['area_ssp370'], color=c2, lw=3, zorder=2, label='SSP3-7.0')
ax1.plot(data_future['year'], data_future['area_ssp245'], color=c3, lw=3, zorder=2, label='SSP2-4.5')
ax1.plot(data_future['year'], data_future['area_ssp126'], color=c4, lw=3, zorder=2, label='SSP1-2.6')

ax1.fill_between(data_current['year'], data_current['area'] - data_current['area']*0.351, 
                 data_current['area'] + data_current['area']*0.351, color='grey', alpha=0.5)
ax1.fill_between(data_future['year'], data_future['ssp585_lower'], data_future['ssp585_higher'],
                 color=c1, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp370_lower'], data_future['ssp370_higher'],
                 color=c2, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp245_lower'], data_future['ssp245_higher'],
                 color=c3, alpha=0.3)
ax1.fill_between(data_future['year'], data_future['ssp126_lower'], data_future['ssp126_higher'],
                 color=c4, alpha=0.3)

#ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Fraction of ice sheet with negative CRE (%)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14, loc=2)
ax1.set_xlim(2000, 2100)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2021-01-11/Future_Areas.png', dpi=200)

###############################################################################
# Runoff from MAR3.9 vs air temperature from ERA5
###############################################################################
mar_files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/MAR39/*'))

runoff = []
for i in mar_files:
    f = netCDF4.Dataset(i)
    ru = np.nanmean(f.variables['RU'][152:244,0,:,:], axis=0)
    msk = f.variables['MSK'][:] > 50
    ru[~msk] = np.nan
    runoff_gt = ((ru * 92) * 225) / 1000000
    runoff.append(np.nansum(runoff_gt))
    
era_t = era.variables['t2m'][:]
era_t_values = []
for i in range(18): # NOTE TO CHANGE THIS ONCE WE HAVE ALL YEARS
    era_t_values.append(np.nanmean(era_t[:,:,i][mask]))

baseline = np.mean(era_t_values[12:])
n_mar = np.arange(2003, 2018, 1)
era_t_values = era_t_values[0:15]

slope, intercept, r_value, p_value, std_err = stats.linregress(era_t_values - baseline, runoff)

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(era_t_values - baseline, runoff, color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

for i, txt in enumerate(n_mar):
    ax1.annotate(txt, (era_t_values[i] - baseline, runoff[i]), fontsize=14)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Meltwater runoff (Gt)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Runoff vs CRE
###############################################################################
mar_files = sorted(glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/MAR39/*'))

runoff = []
for i in mar_files:
    f = netCDF4.Dataset(i)
    ru = np.nanmean(f.variables['RU'][152:244,0,:,:], axis=0)
    msk = f.variables['MSK'][:] > 50
    ru[~msk] = np.nan
    runoff_gt = ((ru * 92) * 225) / 1000000
    runoff.append(np.nansum(runoff_gt))

cre = df['cre__abl'][0:15]
n_mar = np.arange(2003, 2018, 1)

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(cre, runoff)

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(cre, runoff, color=c2, zorder=2, s=100, alpha=0.8)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

textstr = r'y = (%.1f*x) + %.0f' % (slope1, intercept1, )
text_box = AnchoredText(textstr, frameon=True, loc=1, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_xlabel('CRE (W m$^{-2}$)', fontsize=14)
ax1.set_ylabel('Meltwater runoff (Gt)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)



block_se = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SE.nc')
block_sw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_SW.nc')
block_ne = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NE.nc')
block_nw = netCDF4.Dataset('/media/johnny/Cooley_Data/Johnny/Clouds_Data/4_MYD06_Radiative_Flux_Climatologies_Blocking/MYD06_SW_Fluxes_NW.nc')

mod_sw_flux = np.nanmean(mod.variables['allsky_sw'][:], axis=2) - np.nanmean(mod.variables['clrsky_sw'][:], axis=2) 
block_se_flux = block_se.variables['allsky_sw'][:] -  block_se.variables['clrsky_sw'][:]
block_sw_flux = block_sw.variables['allsky_sw'][:] -  block_sw.variables['clrsky_sw'][:]
block_ne_flux = block_ne.variables['allsky_sw'][:] -  block_ne.variables['clrsky_sw'][:]
block_nw_flux = block_nw.variables['allsky_sw'][:] -  block_nw.variables['clrsky_sw'][:]

block_flux = (block_se_flux + block_sw_flux + block_ne_flux + block_nw_flux) / 4

sw_flux_diff = (block_flux - mod_sw_flux)
mod_albedo = np.nanmean(mod.variables['albedo'][:], axis=2)
future_albedo = (-0.05*4) + 0.63

np.nanmean(sw_flux_diff[max_snowline]) * (1 - np.nanmean(mod_albedo[max_snowline]))
np.nanmean(sw_flux_diff[max_snowline]) * (1 - future_albedo)

###############################################################################
# Figure S1: Relationships between cloud radiative effects (CRE), cloudiness and surface albedo.
###############################################################################
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(df['cre_acc'], df['cldy_acc'])
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df['cre_acc'], df['alb_acc'])
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(df['cre_abl'], df['cldy_abl'])
slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(df['cre_abl'], df['alb_abl'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.scatter(df['cre_acc'],df['cldy_acc'] * 100, color=c2, zorder=2, s=100, alpha=0.8,
            label='Accumulation zone')
ax2.scatter(df['cre_acc'],df['alb_acc'], color=c2, zorder=2, s=100, alpha=0.8,
            label='Accumulation zone')
ax3.scatter(df['cre_abl'],df['cldy_abl'] * 100, color=c2, zorder=2, s=100, alpha=0.8,
            label='Ablation zone')
ax4.scatter(df['cre_abl'],df['alb_abl'], color=c2, zorder=2, s=100, alpha=0.8,
            label='Ablation zone')

ax1.set_yticks(np.arange(10, 20, 2))
ax1.set_yticklabels(np.arange(10, 20, 2))
ax2.set_yticks(np.arange(0.76, 0.8, 0.01))
ax2.set_yticklabels(np.arange(0.76, 0.8, 0.01))
ax3.set_yticks(np.arange(10, 24, 3))
ax3.set_yticklabels(np.arange(10, 24, 3))
ax4.set_yticks(np.arange(0.56, 0.72, 0.03))
ax4.set_yticklabels(np.arange(0.56, 0.72, 0.03))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.text(0.05, 0.85, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.05, 0.85, "b", fontsize=24, transform=ax2.transAxes)
ax3.text(0.05, 0.85, "c", fontsize=24, transform=ax3.transAxes)
ax4.text(0.05, 0.85, "d", fontsize=24, transform=ax4.transAxes)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
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
    r'R$^{2}$ = %.2f' % (r_value3**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=3, pad=0.5, prop=dict(size=14))
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

ax1.set_ylabel('Cloudiness (%)', fontsize=14)

ax2.set_ylabel('Surface albedo', fontsize=14)

ax3.set_xlabel('CRE (W m$^{-2}$)', fontsize=14)
ax3.set_ylabel('Cloudiness (%)', fontsize=14)

ax4.set_xlabel('CRE (W m$^{-2}$)', fontsize=14)
ax4.set_ylabel('Surface albedo', fontsize=14)

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(linestyle='dotted', lw=1, zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc=1, markerscale=0, handlelength=0, frameon=False)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S1.png', dpi=200)



###############################################################################
# Figure S2: Relationship between cloudiness and mean summer air temperature.
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

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.scatter(era_t_acc - baseline_acc, df['cldy_acc']*100, color=c2, zorder=2, 
            s=100, alpha=0.8, label='Accumulation zone')
ax2.scatter(era_t_abl - baseline_abl, df['cldy_abl']*100, color=c2, zorder=2, 
            s=100, alpha=0.8, label='Ablation zone')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value2**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax2.add_artist(text_box)

ax1.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax1.set_ylabel('Cloudiness (%)', fontsize=14)
ax2.set_xlabel('Summer air temp. anomaly (K)', fontsize=14)
ax2.set_ylabel('Cloudiness (%)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax2.grid(linestyle='dotted', lw=1, zorder=1)

ax1.legend(fontsize=14, loc=1, markerscale=0, handlelength=0, frameon=False)
ax2.legend(fontsize=14, loc=1, markerscale=0, handlelength=0, frameon=False)
ax1.text(0.05, 0.90, "a", fontsize=24, transform=ax1.transAxes)
ax2.text(0.05, 0.90, "b", fontsize=24, transform=ax2.transAxes)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S2.png', dpi=200)

###############################################################################
# Figure S5: Evaluation of ERA5 2 m air temperatures versus PROMICE automated weather stations.
###############################################################################
slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(era_aws['aws_t2m'], era_aws['era_t2m'])
rmse = np.sqrt(np.mean((era_aws['aws_t2m'] - era_aws['era_t2m'])**2))
bias = np.mean(era_aws['aws_t2m'] - era_aws['era_t2m'])
mae = np.sum(np.abs(era_aws['aws_t2m'] - (era_aws['era_t2m']+bias))) / era_aws.shape[0]

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

#ax1.hist2d(era_aws['aws_t2m'], era_aws['era_t2m'], bins=(100, 100), cmap=plt.cm.BuPu)
ax1.scatter(era_aws['aws_t2m'], era_aws['era_t2m'], s=50, color=c2, alpha=0.5)
ax1.plot([150, 400], [150, 400], lw=2, color='k', ls='dashed')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value1**2, ),
    r'MAE = %.1f K' % (mae, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 2 m air temperature (K)', fontsize=14)
ax1.set_xlabel('AWS 2 m temperature (K)', fontsize=14)

ax1.set_xlim(250, 295)
ax1.set_ylim(250, 295)
ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_S5.png', dpi=200)