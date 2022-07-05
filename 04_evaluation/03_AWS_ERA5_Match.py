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
import xarray as xr 
from datetime import timedelta, datetime

# Define AWS coordinates
promice = pd.read_csv('/home/johnny/Documents/Clouds/Data/PROMICE_Coordinates.csv')

# Define AWS files
aws_files = glob.glob('/home/johnny/Documents/Clouds/Data/PROMICE/Stations/*')

# Define ERA5 data
era5 = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/adaptor.mars.internal-1607354850.4485557-16817-5-e984cfba-dce3-4c51-bf5c-bc92cc20f892.nc')
era5_lon = era5.variables['longitude'][:]
era5_lat = era5.variables['latitude'][:]
era5_xx, era5_yy = np.meshgrid(era5_lon, era5_lat)

# Get time
base = datetime(1900,1,1)
era5_time = pd.DataFrame(era5.variables['time'][:], columns=['hours'])
era5_time['datetime'] = era5_time['hours'].apply(lambda x: base + timedelta(hours=x))
era5_time['index'] = era5_time.index

ds = xr.open_dataset('/home/johnny/Documents/Clouds/Data/ERA5/adaptor.mars.internal-1607354850.4485557-16817-5-e984cfba-dce3-4c51-bf5c-bc92cc20f892.nc')
ssrd_daily = ds['ssrd'].resample(time='1D').mean()
strd_daily = ds['strd'].resample(time='1D').mean()

# Loop over every CloudSat profile
aws_sw = []
aws_lw = []
era_sw = []
era_lw = []
timing = []
station = []

aws_sw_daily = []
aws_lw_daily = []
era_sw_daily = []
era_lw_daily = []
timing_daily = []
station_daily = []

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
        aws = aws[aws['ShortwaveRadiationDown_Cor(W/m2)'] != -999]
        aws = aws[aws['LongwaveRadiationDown(W/m2)'] != -999]
        
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
        era5_strd_hourly = era5.variables['strd'][aws_clip['index'].values, idx_x[0], idx_y[0]]
        era5_ssrd_hourly = era5.variables['ssrd'][aws_clip['index'].values, idx_x[0], idx_y[0]]
        
        # Get daily ERA5 values corresponding to AWS record
        era5_ssrd = ssrd_daily[:, idx_x[0], idx_y[0]]
        era5_strd = strd_daily[:, idx_x[0], idx_y[0]]
        era5_daily = pd.DataFrame(list(zip(strd_daily['time'].values, 
                                           era5_ssrd.values, era5_strd.values)), 
                          columns=['dt', 'ssrd', 'strd'])
        aws_clip_daily = pd.merge(aws_daily, era5_daily, how='inner', on='dt')
        
        # Append to list
        era_sw.append(list(era5_ssrd_hourly / 3600))
        era_lw.append(list(era5_strd_hourly / 3600))
        aws_sw.append(aws_clip['ShortwaveRadiationDown_Cor(W/m2)'].values)
        aws_lw.append(aws_clip['LongwaveRadiationDown(W/m2)'].values)    
        station.append(np.repeat(station_id, aws_clip.shape[0]))
        timing.append(aws_clip['datetime'].values)
        
        aws_sw_daily.append(aws_clip_daily['ShortwaveRadiationDown_Cor(W/m2)'].values)
        aws_lw_daily.append(aws_clip_daily['LongwaveRadiationDown(W/m2)'].values)
        era_sw_daily.append(aws_clip_daily['ssrd'].values / 3600)
        era_lw_daily.append(aws_clip_daily['strd'].values / 3600)
        timing_daily.append(aws_clip_daily['dt'].values)
        station_daily.append(np.repeat(station_id, aws_clip_daily.shape[0]))
        
    else:
        pass
    
# Flatten
aws_sw_flat = [item for sublist in aws_sw for item in sublist]
aws_lw_flat = [item for sublist in aws_lw for item in sublist]
era_sw_flat = [item for sublist in era_sw for item in sublist]
era_lw_flat = [item for sublist in era_lw for item in sublist]
station_flat = [item for sublist in station for item in sublist]
timing_flat = [item for sublist in timing for item in sublist]

# Put back into DataFrame
df = pd.DataFrame(list(zip(timing_flat, station_flat, aws_sw_flat, 
                           aws_lw_flat, era_sw_flat, era_lw_flat)))

df.columns = ['datetime', 'station', 'aws_sw', 'aws_lw','era_sw', 'era_lw']

# Remove rows with no data
df = df[df['aws_sw'] != -999]
df = df[df['aws_lw'] != -999]

# Save as csv
df.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Hourly.csv')

# Flatten
aws_sw_flat = [item for sublist in aws_sw_daily for item in sublist]
aws_lw_flat = [item for sublist in aws_lw_daily for item in sublist]
era_sw_flat = [item for sublist in era_sw_daily for item in sublist]
era_lw_flat = [item for sublist in era_lw_daily for item in sublist]
station_flat = [item for sublist in station_daily for item in sublist]
timing_flat = [item for sublist in timing_daily for item in sublist]

# Put back into DataFrame
df = pd.DataFrame(list(zip(timing_flat, station_flat, aws_sw_flat, 
                           aws_lw_flat, era_sw_flat, era_lw_flat)))

df.columns = ['datetime', 'station', 'aws_sw', 'aws_lw','era_sw', 'era_lw']

# Remove rows with no data
df = df[df['aws_sw'] != -999]
df = df[df['aws_lw'] != -999]

# Save as csv
df.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/AWS_vs_ERA5_Daily.csv')








