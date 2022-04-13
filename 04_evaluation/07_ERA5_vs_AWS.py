#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
DESCRIPTION

Plot AWS vs ERA5 radiative fluxes

"""


# Import modules
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Define matched DataFrame
df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Hourly.csv', parse_dates=['datetime'])

# Remove zeros
df = df[df['aws_sw'] > 100]
df = df[df['era_sw'] > 100]

df_daily = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Daily.csv', parse_dates=['datetime'])
df_daily = df_daily.dropna()

###############################################################################
# Plot LW hourly
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_lw'], df['era_lw'])
rmse = np.sqrt(np.mean((df['aws_lw'] - df['era_lw'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['aws_lw'], df['era_lw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([150, 400], [150, 400], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 LW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS LW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_LW_Hourly.png', dpi=200)

###############################################################################
# Plot SW hourly
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_sw'], df['era_sw'])
rmse = np.sqrt(np.mean((df['aws_sw'] - df['era_sw'])**2))
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['aws_sw'], df['era_sw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([0, 700], [0, 700], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 SW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS SW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xlim(100, 700)
ax1.set_ylim(100, 700)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_SW_Hourly.png', dpi=200)

###############################################################################
# Plot LW daily
###############################################################################

slope, intercept, r_value, p_value, std_err = stats.linregress(df_daily['aws_lw'], df_daily['era_lw'])
rmse = np.sqrt(np.mean((df_daily['aws_lw'] - df_daily['era_lw'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df_daily['aws_lw'], df_daily['era_lw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([150, 400], [150, 400], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 LW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS LW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_LW_Daily.png', dpi=200)

###############################################################################
# Plot SW daily
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df_daily['aws_sw'], df_daily['era_sw'])
rmse = np.sqrt(np.mean((df_daily['aws_sw'] - df_daily['era_sw'])**2))
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df_daily['aws_sw'], df_daily['era_sw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([0, 400], [0, 400], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 SW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS SW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xlim(100, 400)
ax1.set_ylim(100, 400)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_SW_Daily.png', dpi=200)

# =============================================================================
# ###############################################################################
# # Plot LW Monthly (Hourly)
# ###############################################################################
# df['month'] = df['datetime'].dt.month
# df_month = df.groupby(['month','station']).mean().reset_index()
# 
# slope, intercept, r_value, p_value, std_err = stats.linregress(df_month['aws_lw'], df_month['era_lw'])
# rmse = np.sqrt(np.mean((df_month['aws_lw'] - df_month['era_lw'])**2))
# 
# fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
# 
# ax1.scatter(df_month['aws_lw'], df_month['era_lw'])
# ax1.plot([225, 325], [225, 325], lw=2, color='k')
# 
# # Add stats
# textstr = '\n'.join((
#     r'R$^{2}$ = %.2f' % (r_value**2, ),
#     r'RMSE = %.1f W m-2' % (rmse, )))
# text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
# text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# plt.setp(text_box.patch, facecolor='white', alpha=0.7)
# ax1.add_artist(text_box)
# 
# ax1.set_ylabel('ERA5 LW down (W m-2)', fontsize=14)
# ax1.set_xlabel('AWS LW down (W m-2)', fontsize=14)
# 
# ax1.tick_params(axis='both', which='major', labelsize=14)
# 
# fig.tight_layout()
# fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_LW_Monthly.png', dpi=200)
# 
# ###############################################################################
# # Plot SW Monthly (Hourly)
# ###############################################################################
# slope, intercept, r_value, p_value, std_err = stats.linregress(df_month['aws_sw'], df_month['era_sw'])
# rmse = np.sqrt(np.mean((df_month['aws_sw'] - df_month['era_sw'])**2))
# 
# fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
# 
# ax1.scatter(df_month['aws_sw'], df_month['era_sw'])
# #ax1.plot([225, 325], [225, 325], lw=2, color='k')
# 
# # Add stats
# textstr = '\n'.join((
#     r'R$^{2}$ = %.2f' % (r_value**2, ),
#     r'RMSE = %.1f W m-2' % (rmse, )))
# text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
# text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
# plt.setp(text_box.patch, facecolor='white', alpha=0.7)
# ax1.add_artist(text_box)
# 
# ax1.set_ylabel('ERA5 SW down (W m-2)', fontsize=14)
# ax1.set_xlabel('AWS SW down (W m-2)', fontsize=14)
# 
# ax1.tick_params(axis='both', which='major', labelsize=14)
# 
# fig.tight_layout()
# fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_SW_Monthly.png', dpi=200)
# 
# ###############################################################################
# # Plot daily LW cycle
# ###############################################################################
# # Read data again
# df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5.csv', parse_dates=['datetime'])
# 
# # Group by station at a daily time-scale
# df['hour'] = df['datetime'].dt.hour
# df_group = df.groupby(['station', 'hour']).mean().reset_index()
# =============================================================================






















                  

                  

                  

                  

    
    
    