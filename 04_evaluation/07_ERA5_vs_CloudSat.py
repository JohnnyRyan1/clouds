#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
DESCRIPTION

Plot ERA5 vs CloudSat radiative fluxes

"""


# Import modules
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Define matched DataFrame
df = pd.read_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/CS_vs_ERA5.csv')

# Remove columns with only one CloudSat value
df = df[df['count'] > 5]

###############################################################################
# Plot LW
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['cs_lw'], df['era_lw'])
rmse = np.sqrt(np.mean((df['cs_lw'] - df['era_lw'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['cs_lw'], df['era_lw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([150, 320], [150, 320], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=2, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_xlabel('CloudSat LW down (W m-2)', fontsize=14)
ax1.set_ylabel('ERA5 LW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(150, 320)
ax1.set_xlim(150, 320)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/ERA5_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot SW
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['cs_sw'], df['era_sw'])
rmse = np.sqrt(np.mean((df['cs_sw'] - df['era_sw'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['cs_sw'], df['era_sw'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([100, 800], [100, 800], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=2, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_xlabel('CloudSat SW down (W m-2)', fontsize=14)
ax1.set_ylabel('ERA5 SW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(100, 800)
ax1.set_xlim(100, 800)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/ERA5_vs_CloudSat_SW.png', dpi=200)
                  
###############################################################################
# Plot LW clear-sky
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['cs_lw_cs'], df['era_lw_cs'])
rmse = np.sqrt(np.mean((df['cs_lw_cs'] - df['era_lw_cs'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['cs_lw_cs'], df['era_lw_cs'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([150, 320], [150, 320], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=2, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_xlabel('CloudSat LW down (W m-2)', fontsize=14)
ax1.set_ylabel('ERA5 LW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(150, 320)
ax1.set_xlim(150, 320)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/ERA5_vs_CloudSat_LW_ClearSky.png', dpi=200)

###############################################################################
# Plot SW clear-sky
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['cs_sw_cs'], df['era_sw_cs'])
rmse = np.sqrt(np.mean((df['cs_sw_cs'] - df['era_sw_cs'])**2))

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['cs_sw_cs'], df['era_sw_cs'], bins=(50, 50), cmap=plt.cm.BuPu)
ax1.plot([100, 800], [100, 800], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=2, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_xlabel('CloudSat SW down (W m-2)', fontsize=14)
ax1.set_ylabel('ERA5 SW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_ylim(100, 800)
ax1.set_xlim(100, 800)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/ERA5_vs_CloudSat_SW_ClearSky.png', dpi=200)
          

                  

                  

                  

    
    
    