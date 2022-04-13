#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Extract MODIS cloud properties and ERA5 surface temperature for CloudSat profiles.

"""

# Import modules
import pandas as pd
import numpy as np
import geopandas as gpd
import glob
import os
import netCDF4
from functions import hdf_read
from shapely.geometry import Point
from Functions import geopandas_utils
from datetime import timedelta, datetime

# Define coordinate system
df_crs = 'EPSG:4326'

# Define destination
dest = '/home/johnny/Documents/Clouds/Data/Merged_Data_v2/'

# Define 2B-FLXHR-LIDAR data
all_flxhr = sorted(glob.glob('/media/johnny/Seagate Backup Plus Drive/Clouds/Data/2B-FLXHR-LIDAR/*.hdf'))

# Define 2B-CLDCLSS-LIDAR data
all_cldclss = sorted(glob.glob('/media/johnny/Seagate Backup Plus Drive/Clouds/Data/2B-CLDCLSS-LIDAR/*.hdf'))

# Define MOD06 cloud property data
all_mod06 = sorted(glob.glob('/media/johnny/Seagate Backup Plus Drive/Clouds/Data/MAC06SO/*.hdf'))

# Define MODIS monthly surface albedo data
all_mod10 = sorted(glob.glob('/media/johnny/Seagate Backup Plus Drive/Clouds/Data/SciAdv_Products/*.nc'))

# Define ice sheet mask and elevations
ismip = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/1km-ISMIP6.nc')
ismip_lon = ismip.variables['lon'][:]
ismip_lat = ismip.variables['lat'][:]
ismip_mask = ismip.variables['ICE'][:]
ismip_elev = ismip.variables['SRF'][:]

# Define region mask
regions = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Masks/Ultimate_Mask.nc')
regions_y = regions.variables['y'][:]
regions_x = regions.variables['x'][:]
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

# Define ERA5 data
era5 = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/ERA5/adaptor.mars.internal-1612837523.5205054-27920-1-3edca96f-8976-4cff-81f9-d14c96f7feba.nc')
era5_lon = era5.variables['longitude'][:]
era5_lat = era5.variables['latitude'][:]
era5_xx, era5_yy = np.meshgrid(era5_lon, era5_lat)

# Get time
base = datetime(1900,1,1)
era5_time = pd.DataFrame(era5.variables['time'][:], columns=['hours'])
era5_time['datetime'] = era5_time['hours'].apply(lambda x: base + timedelta(hours=x))

# Convert to GeoDataFrame
era5_df = pd.DataFrame(list(zip((np.ravel(era5_xx)), np.ravel(era5_yy))), columns = ['lon', 'lat'])
geometry = [Point(xy) for xy in zip(era5_df['lon'], era5_df['lat'])]
era_gdf = gpd.GeoDataFrame(era5_df, crs=df_crs, geometry=geometry)

# Reverse list
all_flxhr = all_flxhr[::-1]

###############################################################################
# Extract CloudSat data from HDF
###############################################################################
#start = 828310
for i in range(len(all_flxhr)):
    
    # Get path and filename separately
    infilepath1, infilename1 = os.path.split(all_flxhr[i])
    # Get file name without extension
    infileshortname1, extension1 = os.path.splitext(infilename1)
    
    if os.path.exists(dest + infileshortname1[0:19] + '.csv'):
        print('Skipping... %s' %(infileshortname1[0:19] + '.csv'))
    
    else:
    
        # Get date string
        date_string = infileshortname1[0:7]
        
        # Get Julian Day
        julian = int(date_string[4:])
        
        # Only process if summer
        if (julian > 151) & (julian < 244):
        
            print('Matching %.0f of %.0f' %(i+1, len(all_flxhr)))
            
            # Find corresponding 2B-CLDCLSS-LIDAR profile
            cldclss_match = [s for s in all_cldclss if infileshortname1[0:19] in s]
                
            # Merge 2B-FLXHR and 2B-CLDCLSS-LIDAR profiles
            df = hdf_read.CloudSat_Product_Merge(all_flxhr[i], cldclss_match[0])
            
            ###########################################################################
            # Get MODIS cloud properties from MOD06
            ###########################################################################
            
            if df.shape[0] > 0:
                #print('Appending MOD06 cloud properties...')
                # Convert to GeoDataFrame
                geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
                gdf = gpd.GeoDataFrame(df, crs=df_crs, geometry=geometry)
                        
                # Find corresponding MODIS matches
                mod06_files = [s for s in all_mod06 if date_string in s]
                
                # Get MODIS swaths +/- 2 hours of CloudSat profile
                mod06_files_matched = []
                for j in mod06_files:
                    # Get path and filename separately
                    infilepath2, infilename2 = os.path.split(j)
                    # Get file name without extension
                    infileshortname2, extension2 = os.path.splitext(infilename2)
                    
                    # Get hour 
                    hour = int(infileshortname2[17:19])
                    
                    if np.abs(hour - df['datetime'].iloc[0].hour) < 3:
                        mod06_files_matched.append(j)
                
                final_df = pd.DataFrame()
                # Add MOD06 data from each corresponding tile
                for k in range(len(mod06_files_matched)):
                    
                    # Read MODIS data
                    modis_df = hdf_read.MOD06_Read(mod06_files_matched[k], df)
                                
                    if modis_df.shape[0] > 0:
                        
                        #print('Appending MOD06 cloud properties...')
                        # Concatenate with original CloudSat data
                        geometry = [Point(xy) for xy in zip(modis_df['lon'], modis_df['lat'])]
                        modis_gdf = gpd.GeoDataFrame(modis_df, crs=df_crs, geometry=geometry)
                                    
                        # Call function
                        closest_point, idx = geopandas_utils.nearest_neighbor(gdf.reset_index(drop=True), 
                                                                         modis_gdf.reset_index(drop=True), 
                                                                         return_dist=True)
                                
                        # Rename the geometry of closest stops gdf so that we can identify it
                        closest_point = closest_point.rename(columns={'geometry': 'geom_modis'})
                                
                        # Merge the datasets by index (for this, it is good to use '.join()' -function)
                        combined = gdf.reset_index(drop=True).join(closest_point, rsuffix='_modis')
                        
                        # Filter only close distances
                        combined = combined[combined['distance'] < 1500]
                        
                        # Drop some columns
                        combined = combined.drop('geom_modis', axis=1)
                        combined = combined.drop('distance', axis=1)
                        combined = combined.drop('lon_modis', axis=1)
                        combined = combined.drop('lat_modis', axis=1)
                        combined = combined.drop('sza_modis', axis=1)
                        
                        if combined.shape[0] > 0:
                            # Append to final DataFrame           
                            final_df = final_df.append(combined)
                        
                        else:
                            pass
                        
                if final_df.shape[0] > 0:
                    
                    # Sort DataFrame
                    final_df = final_df.sort_values(by=['datetime'])
                    
                    ###################################################################
                    # Get near-surface air temperature (proxy for cloud bottom 
                    # temperature from ERA5 
                    ###################################################################
                    
                    #print('Appending ERA5 near-surface air temperatures...')
                    # Get a list of ERA5 lat/lons to extract temperature data
                    closest_era, idx = geopandas_utils.nearest_neighbor(final_df.reset_index(drop=True), 
                                                                   era_gdf.reset_index(drop=True), 
                                                                   return_dist=True)
                    
                    # Get ERA5 corresponding to nearest datetime
                    date_match = np.abs(era5_time['datetime'] - final_df['datetime'].iloc[0]).argmin()
                    era5_t = era5.variables['t2m'][date_match, :, :]
                    era5_d2m = era5.variables['d2m'][date_match, :, :]
                    era5_ssrdc = era5.variables['ssrdc'][date_match, :, :]
                    era5_strdc = era5.variables['strdc'][date_match, :, :]
                    
                    # Add 2m temperature to DataFrame
                    final_df['t2m'] = np.ravel(era5_t)[idx]
                    final_df['d2m'] = np.ravel(era5_d2m)[idx]
                    final_df['ssrdc'] = np.ravel(era5_ssrdc)[idx]
                    final_df['strdc'] = np.ravel(era5_strdc)[idx]
                    
                    ###################################################################
                    # Get surface albedo from MOD10A1 
                    ###################################################################
                    #print('Appending MOD10A1 surface albedo...')
            
                    # Find corresponding MODIS matches
                    mod_file = [s for s in all_mod10 if date_string[0:4] in s]
                    
                    # Get month
                    if julian < 182:
                        month = 0
                    if (julian > 182) & (julian < 213):
                        month = 1
                    if julian > 212:
                        month = 2
                    
                    # Read MODIS data
                    final_df = hdf_read.MODIS_Monthly_Read(mod_file[0], final_df, month)
                    
                    ###################################################################
                    # Get surface elevation from GIMP
                    ###################################################################
                    #print('Appending surface elevation and mask...')
                    
                    # Read physical data
                    final_df = hdf_read.ISMIP_Read(ismip_lon, ismip_lat, ismip_elev, ismip_mask, final_df)
                    
                    ###################################################################
                    # Get region
                    ###################################################################
                    #print('Appending region...')
                    
                    # Read regional data
                    final_df = hdf_read.Regions_Read(regions_x, regions_y, regions_mask, final_df)
                    
                    # Filter values outside mask
                    final_df = final_df[final_df['mask'] != 0]
                    
                    if final_df.shape[0] > 0:
                        
                        # Add filename column
                        final_df['filename'] = np.repeat(infileshortname1[0:19], final_df.shape[0])
                                            
                        # Remove column
                        final_df.drop('geometry', axis=1, inplace=True)
                        
                        # Save as csv
                        final_df.to_csv(dest + infileshortname1[0:19] + '.csv')
                    else:
                        pass
                else:
                    pass
            else:
                pass
        else:
            pass
        

          





























