#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

DESCRIPTION

Compute daily cycle in downward LW and SW AWS data.

"""

# Import modules
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Define AWS data files
aws_files = glob.glob('/home/johnny/Documents/Clouds/Data/PROMICE/Stations/*.txt')

# Read AWS
aws = pd.read_csv(aws_files[0], skiprows=1, header=None, delim_whitespace=True)
header = pd.read_csv(aws_files[0], nrows=0, delim_whitespace=True)
aws.columns = list(header)

# Add a datetime column
aws_datetime = aws[['Year', 'MonthOfYear', 'DayOfMonth', 'HourOfDay(UTC)']]
aws_datetime.columns = ['year', 'month', 'day', 'hour']
aws['datetime'] = pd.to_datetime(aws_datetime[['year', 'month', 'day', 'hour']])

# Get columns of interest
aws = aws[['datetime', 'ShortwaveRadiationDown_Cor(W/m2)','LongwaveRadiationDown(W/m2)']]

# Remove no data
aws = aws[aws['ShortwaveRadiationDown_Cor(W/m2)'] != -999]
aws = aws[aws['LongwaveRadiationDown(W/m2)'] != -999]

# Add hour columns
aws['hour'] = aws['datetime'].dt.hour

# Grouby hour
aws_group = aws.groupby('hour').mean().reset_index()

###############################################################################
# Plot
###############################################################################
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

# Define colour map
c1 = '#E05861'
c2 = '#616E96'
c3 = '#F8A557'
c4 = '#3CBEDD'

ax1.grid(linestyle='dotted', lw=1, zorder=1)
ax1.plot(aws_group['hour'], aws_group['ShortwaveRadiationDown_Cor(W/m2)'], 
         color=c2, zorder=2, alpha=0.8, lw=3, label='SW')
ax1.plot(aws_group['hour'], aws_group['LongwaveRadiationDown(W/m2)'], 
         color=c1, zorder=2, alpha=0.8, lw=3, label='LW')

ax1.set_ylabel('Downward flux (W m-2)', fontsize=14)
ax1.set_xlabel('Hour of day', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=14)
fig.tight_layout()
fig.savefig('/home/johnny/Documents/Clouds/Presentations/2020-12-14/LW_SW_Daily_Cycle.png', dpi=200)















