#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
DESCRIPTION

Plot AWS vs CloudSat radiative fluxes

"""


# Import modules
import pandas as pd
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

# Define matched DataFrame
matches = pd.read_csv('/home/johnny/Documents/Clouds/Data/Validation/CloudSat_AWS_Pairs.csv')

# Define AWS data files
aws_files = glob.glob('/home/johnny/Documents/Clouds/Data/PROMICE/Stations_Matched/*.txt')

aws_lw_as = []
aws_sw_as = []
#aws_sw_as_jaws = []
cs_lw_as = []
cs_sw_as = []
station = []
time_diff = []
filename = []

for i in range(matches.shape[0]):
    
    # Append CloudSat data
    cs_lw_as.append(matches['lw_as'].iloc[i])
    cs_sw_as.append(matches['sw_as'].iloc[i])
    time_diff.append(matches['time_diff'].iloc[i])
    station.append(matches['station'].iloc[i])
    filename.append(matches['filename'].iloc[i])
    
    # Get datetime string
    date_string = matches['datetime'].iloc[i]
    date_string = date_string.replace(' ', '_')
    date_string = date_string.replace(':', '_')
    date_string = date_string.replace('.', '_')
    
    # Find corresponding file
    aws_match = [s for s in aws_files if date_string in s]
    
    # Read AWS data
    #aws = netCDF4.Dataset(aws_match[0])
    aws = pd.read_csv(aws_match[0], skiprows=1, header=None, delim_whitespace=True)
    header = pd.read_csv(aws_match[0], nrows=0, delim_whitespace=True)
    aws.columns = list(header)
    
    # Append AWS data
    aws_sw_as.append(aws['ShortwaveRadiationDown_Cor(W/m2)'].iloc[0])
    #aws_sw_as_jaws.append(aws.variables['fsds_adjusted'][:])
    aws_lw_as.append(aws['LongwaveRadiationDown(W/m2)'].iloc[0])

# Put back into DataFrame
df = pd.DataFrame(list(zip(aws_lw_as, aws_sw_as, cs_lw_as, cs_sw_as, station, 
                           time_diff, filename)))
df.columns = ['aws_lw_as', 'aws_sw_as', 'cs_lw_as', 'cs_sw_as', 'station', 
              'time_diff', 'filename']           

# Convert no data to NaNs
df[df['aws_lw_as'] == -999] = np.nan

# Add a difference column
df['sw_diff'] = df['cs_sw_as'] - df['aws_sw_as']

###############################################################################
# Plot LW
###############################################################################
df = df.dropna()
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_lw_as'], df['cs_lw_as'])
rmse = np.sqrt(np.mean((df['aws_lw_as'] - df['cs_lw_as'])**2))
mae = np.sum(np.abs(df['aws_lw_as'] - df['cs_lw_as'])) / df.shape[0]

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(df['aws_lw_as'], df['cs_lw_as'], color=c2, zorder=2, label='SW', 
             alpha=0.8)
ax1.plot([200, 350], [200, 350], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('CloudSat LW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS LW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_LW.png', dpi=200)

###############################################################################
# Plot SW
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_sw_as'], df['cs_sw_as'])
rmse = np.sqrt(np.mean((df['aws_sw_as'] - df['cs_sw_as'])**2))
mae = np.sum(np.abs(df['aws_sw_as'] - df['cs_sw_as'])) / df.shape[0]

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.scatter(df['aws_sw_as'], df['cs_sw_as'], color=c2, zorder=2, label='SW', 
             alpha=0.8)
ax1.plot([0, 700], [0, 700], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f W m-2' % (rmse, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('CloudSat SW down (W m-2)', fontsize=14)
ax1.set_xlabel('AWS SW down (W m-2)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_CloudSat_SW.png', dpi=200)
                  

                  

                  

                  

                  

    
    
    