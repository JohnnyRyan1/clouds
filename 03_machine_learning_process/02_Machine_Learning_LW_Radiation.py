#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Machine learning F_LW from MODIS cloud properties.

"""

# Import modules
import pandas as pd
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from functions import hdf_read
import netCDF4
from datetime import timedelta, datetime

# Define data
files = glob.glob('/home/johnny/Documents/Clouds/Data/Merged_Data_v2/*')

# Read data
df = hdf_read.data_read_machine_learning(files)

# Remove clear skies
data = df[df['type'] > 0]

# Define feature list
feature_list = ['modis_cot', 'modis_ctp', 'modis_phase', 'modis_cer', 
                'modis_ctt', 'modis_cth', 'modis_cwp', 't2m']

# Define labels and targets
y = data['f_lw']
X = data[['modis_cot', 'modis_ctp', 'modis_phase', 'modis_cer', 'modis_ctt', 
          'modis_cth', 'modis_cwp', 't2m']]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Define classifier
classifier = RandomForestRegressor(n_estimators=100)

# Train classifier
classifier.fit(X_train, y_train)

# Predict
predictions = classifier.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 3))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(classifier.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

df_preds = pd.DataFrame(list(zip(y_test, predictions)), 
                        columns=['y_test', 'predictions'])
df_preds.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/lw_prediction_results.csv')

# =============================================================================
# # Add test predictions to original DataFrame and save as csv
# df_preds = pd.DataFrame(list(zip(y_test.index, y_test, predictions)), 
#                         columns=['idx', 'y_test', 'y_preds'])
# df_preds.set_index('idx', inplace=True)
# df_out = pd.merge(data, df_preds, how='right', left_index=True, right_index=True)
# df_out['diff'] = np.abs(df_out['y_test'] - df_out['y_preds'])
# df_out['poor'] = (df_out['diff'] > 0.1).astype(int)
# df_out.to_csv('/home/johnny/Documents/Clouds/Data/Model_Evaluation/LW_Predictions.csv')
# =============================================================================

###############################################################################
# Apply model to entire of Greenland
###############################################################################

# Define files
modis_list = glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/MYD06_L2/*.hdf')

# Define and read ERA5 data
era_t = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_t2m_2003_2020.nc')
era_fluxes = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/era_clrsky_2003_2020.nc')
era_lon = era_t.variables['longitude'][:]
era_lat = era_t.variables['latitude'][:]
era_xx, era_yy = np.meshgrid(era_lon, era_lat)

# Get time
base = datetime(1900,1,1)
era_time = pd.DataFrame(era_t.variables['time'][:], columns=['hours'])
era_time['datetime'] = era_time['hours'].apply(lambda x: base + timedelta(hours=x))

# Define destination
dest = '/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_LW_V2/'

years = np.arange(2003, 2021, 1)
 
for p in years[::-1]:
    # Get MODIS files
    modis_list_by_years = []
    for j in range(len(modis_list)):
        
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(modis_list[j])
        # Get file name without extension            
        infileshortname, extension = os.path.splitext(infilename)
        
        if infileshortname[10:14] == str(p):
            modis_list_by_years.append(modis_list[j])
            
    print('Processing year... %s' %str(p))        
    for h in range(len(modis_list_by_years)):
        
        print('Processing... %.0f out of %.0f' %(h, len(modis_list_by_years)))
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(modis_list_by_years[h]) 
        # Get file name without extension            
        infileshortname, extension = os.path.splitext(infilename)
        
        if os.path.exists(dest + infileshortname[0:22] + '.csv'):
            pass
        else:
        
            # Day of year
            dayofyear = int(infileshortname[14:17])
            modis_hour = int(infileshortname[18:20])
            modis_min = int(infileshortname[20:22])
            
            # Convert to datetime
            modis_time = datetime(p, 1, 1) + timedelta(days = dayofyear - 1) +\
                timedelta(hours = modis_hour, minutes = modis_min)
                
            # Read MODIS file and combine with ERA5 surface temperature
            df, df_clearsky = hdf_read.MYD06_L2_Read_LW(modis_list_by_years[h],
                                                     era_t, era_fluxes, era_time, 
                                                     era_xx, era_yy, modis_time)
            
            if df.shape[0] == 0:
                pass
            else:
                
                # Make predictions
                predictions = classifier.predict(df[['modis_cot', 'modis_ctp', 'modis_phase', 'modis_cer', 
                'modis_ctt', 'modis_cth', 'modis_cwp', 't2m']])
                df['f_lw'] = predictions
                
                # Compute clear-sky flux based on strdc
                df['clearsky_lw'] = (0.9 * df['strdc']) + 25.7
                
                # Compute all-sky flux 
                df['allsky_lw'] = df['f_lw'] * df['clearsky_lw']
                            
                # Save new DataFrame as csv
                new_df = df[['lon', 'lat', 'clearsky_lw', 'allsky_lw']]
                new_df.insert(2, 'value', 0)
                all_df = pd.concat((new_df, df_clearsky))
                all_df = all_df.astype(np.float32)
                all_df.to_csv(dest + infileshortname[0:22] + '.csv')
    
    
# Count how many files in each year to check progress
modis_list = glob.glob('/media/johnny/Cooley_Data/Johnny/Clouds_Data/2_MYD06_Radiative_Fluxes_CSV_SW_V2/*.csv')
years = np.arange(2003, 2021, 1)
count = []
for p in years:
    # Get MODIS files
    modis_list_by_years = []
    for j in range(len(modis_list)):
        
        # Get path and filename seperately 
        infilepath, infilename = os.path.split(modis_list[j])
        # Get file name without extension            
        infileshortname, extension = os.path.splitext(infilename)
        
        if infileshortname[10:14] == str(p):
            modis_list_by_years.append(modis_list[j])
    
    count.append(len(modis_list_by_years))
    
    








