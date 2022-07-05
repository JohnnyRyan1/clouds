#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Compare AWS against ERA5

"""

# Import modules
import numpy as np
import pandas as pd
import glob
import netCDF4
from datetime import timedelta, datetime
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from functions import hdf_read

# Define data
files = glob.glob('/home/johnny/Documents/Clouds/Data/Merged_Data/*')

# Read data
data = hdf_read.data_read_machine_learning(files)

# Remove clear skies
data = data[data['type'] > 0]

# Define AWS coordinates
promice = pd.read_csv('/home/johnny/Documents/Clouds/Data/PROMICE_Coordinates.csv')

# Define AWS files
aws_files = glob.glob('/home/johnny/Documents/Clouds/Data/PROMICE/Stations/*')

# Define ERA5 data
era5 = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_t2m_2002_2020.nc')
era5_lon = era5.variables['longitude'][:]
era5_lat = era5.variables['latitude'][:]
era5_xx, era5_yy = np.meshgrid(era5_lon, era5_lat)

# Get time
base = datetime(1900,1,1)
era5_time = pd.DataFrame(era5.variables['time'][:], columns=['hours'])
era5_time['datetime'] = era5_time['hours'].apply(lambda x: base + timedelta(hours=x))
era5_time['index'] = era5_time.index

# Loop over every CloudSat profile
aws_t2m = []
era_t2m = []
timing = []
station = []

for i in range(promice.shape[0]):
    
    print('Matching %.0f of %.0f' %(i+1, promice.shape[0]))
    
    # Get station name and filename separately
    station_id = promice['station'].iloc[i]
    
    # Replace underscore with hyphen
    station_id = station_id.replace('_', '-')
    
    # Find corresponding file
    aws_match = [s for s in aws_files if station_id in s]
    
    if len(aws_match) == 1:
        
        # Read AWS
        aws = pd.read_csv(aws_match[0], skiprows=1, header=None, delim_whitespace=True)
        header = pd.read_csv(aws_match[0], nrows=0, delim_whitespace=True)
        aws.columns = list(header)
        
        # Add a datetime column
        aws_datetime = aws[['Year', 'MonthOfYear', 'DayOfMonth', 'HourOfDay(UTC)']]
        aws_datetime.columns = ['year', 'month', 'day', 'hour']
        aws['datetime'] = pd.to_datetime(aws_datetime[['year', 'month', 'day', 'hour']])
        aws['datetime'] = aws['datetime'] + pd.DateOffset(hours=1)
        
        # Remove no data
        aws = aws[aws['AirTemperature(C)'] != -999]
        
        # Resample to daily keeping only days with a full 24 hours of samples
        aws['number'] = 1
        aws_daily = aws.resample('D', on='datetime').mean()
        aws_daily['dt'] = aws_daily.index
        aws_daily_sum = aws.resample('D', on='datetime').sum()
        aws_daily['number'] = aws_daily_sum['number']
        aws_daily = aws_daily[aws_daily['number'] == 24]
        
        # Concatanate based on time
        aws_clip = pd.merge(aws, era5_time, how='inner', on='datetime')
        
        # Get a nearest ERA5 grid cell
        abslat = np.abs(era5_yy - promice['lat'].iloc[i])
        abslon= np.abs(era5_xx - promice['lon'].iloc[i])
        c = np.maximum(abslon,abslat)
        idx_x, idx_y = np.where(c == np.min(c))
        
        # Get hourly ERA5 values corresponding to AWS record 
        era5_t2m_hourly = era5.variables['t2m'][aws_clip['index'].values, idx_x[0], idx_y[0]]
                
        # Append to list
        era_t2m.append(list(era5_t2m_hourly))
        aws_t2m.append(aws_clip['AirTemperature(C)'].values + 273.15)
        station.append(np.repeat(station_id, aws_clip.shape[0]))
        timing.append(aws_clip['datetime'].values)
        
    else:
        pass
    
# Flatten
aws_t2m_flat = [item for sublist in aws_t2m for item in sublist]
era_t2m_flat = [item for sublist in era_t2m for item in sublist]
station_flat = [item for sublist in station for item in sublist]
timing_flat = [item for sublist in timing for item in sublist]

# Put back into DataFrame
df = pd.DataFrame(list(zip(timing_flat, station_flat, aws_t2m_flat, 
                           era_t2m_flat)))

df.columns = ['datetime', 'station', 'aws_t2m', 'era_t2m']

# Remove rows with no data
df = df[df['aws_t2m'] != -999]
df = df[df['era_t2m'] != -999]

# Save as csv
df.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Hourly_t2m.csv')

###############################################################################
# Plot t2m hourly
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_t2m'], df['era_t2m'])
rmse = np.sqrt(np.mean((df['aws_t2m'] - df['era_t2m'])**2))
bias = np.mean(df['aws_t2m'] - df['era_t2m'])
mae = np.sum(np.abs(df['aws_t2m'] - (df['era_t2m']+bias))) / df.shape[0]


fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['aws_t2m'], df['era_t2m'], bins=(50, 50), cmap=plt.cm.BuPu)
#ax1.plot([150, 400], [150, 400], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f K' % (mae, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 2 m air temperature (K)', fontsize=14)
ax1.set_xlabel('AWS 2 m temperature (K)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_LW_Hourly.png', dpi=200)

###############################################################################
# Plot t2m hourly
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(df['aws_t2m'], df['era_t2m'])
rmse = np.sqrt(np.mean((df['aws_t2m'] - df['era_t2m'])**2))
mae = np.sum(np.abs(df['aws_t2m'] - (df['era_t2m']+0.691))) / df.shape[0]
bias = np.mean(df['aws_t2m'] - df['era_t2m'])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

ax1.hist2d(df['aws_t2m'], df['era_t2m'], bins=(50, 50), cmap=plt.cm.BuPu)
#ax1.plot([150, 400], [150, 400], lw=2, color='k')

# Add stats
textstr = '\n'.join((
    r'R$^{2}$ = %.2f' % (r_value**2, ),
    r'RMSE = %.1f K' % (mae, )))
text_box = AnchoredText(textstr, frameon=True, loc=4, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('ERA5 2 m air temperature (K)', fontsize=14)
ax1.set_xlabel('AWS 2 m temperature (K)', fontsize=14)

ax1.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-07/AWS_vs_ERA5_LW_Hourly.png', dpi=200)

###############################################################################
# Clear-sky prediction scatterplot
###############################################################################
slope, intercept, r_value, p_value, std_err = stats.linregress(data['t2m'], data['lw_cs'])
slope_high, intercept_high, r_value_high, p_value_high, std_err_high = stats.linregress(data['t2m']+mae, data['lw_cs'])
slope_low, intercept_low, r_value_low, p_value_low, std_err_low = stats.linregress(data['t2m']-mae, data['lw_cs'])

# Generate points for linear relationship
x = np.arange(data['t2m'].min(), data['t2m'].max(), 1)
y = (slope*x) + intercept
y_high = (slope_high*x) + intercept_high
y_low = (slope_low*x) + intercept_low

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

#ax1.grid(linestyle='dotted', lw=1, zorder=1)
#ax1.hist2d(data['t2m'], data['lw_cs'], bins=(70, 70), cmap=plt.cm.BuPu)
#ax1.scatter(data['t2m'], data['lw_cs'], color=c2, zorder=2, alpha=0.5)
ax1.plot(x, y, color='k', lw=2, ls='dashed')
ax1.plot(x, y_high, color='k', lw=2, ls='dashed')
ax1.plot(x, y_low, color='k', lw=2, ls='dashed')

# Add stats
textstr = r'R$^{2}$ = %.2f' % (r_value**2, )
text_box = AnchoredText(textstr, frameon=True, loc=2, pad=0.5, prop=dict(size=14))
text_box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
plt.setp(text_box.patch, facecolor='white', alpha=0.7)
ax1.add_artist(text_box)

ax1.set_ylabel('Clear-sky longwave (W m-2)', fontsize=14)
ax1.set_xlabel('2 m air temperature (K)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

#fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-14/LW_vs_t2m.png', dpi=200)


