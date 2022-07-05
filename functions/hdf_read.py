#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for CloudSat radiative flux paper.

"""

# Import modules
import pandas as pd
import numpy as np
from pyhdf import HDF, VS
from pyhdf.SD import SD, SDC
import datetime as dt
from scipy.interpolate import griddata
from pyproj import Proj, Transformer
import os
import pyresample
import netCDF4

"""

Check which datasets exist.

from pyhdf.HDF import *
from pyhdf.VS import *
test = HDF(i, HC.READ).vstart()
vdata = test.vdatainfo()
print(vdata)

f = SD(i, SDC.READ)
datasets_dic = file.datasets()
for idx,sds in enumerate(datasets_dic.keys()):
    print(idx,sds)

"""
   
def vs_read(dataset, variable):
    
    # Read a dataset from VS
    variable_id = dataset.attach(dataset.find(variable))
    variable_id.setfields(variable)
    nrecs, _, _, _, _ = variable_id.inquire()
    data = variable_id.read(nRec=nrecs)
    variable_id.detach()
    
    return np.asarray(data)    

def CloudSat_Product_Merge(k, j):
    
    flxhr = HDF.HDF(k)
    flxhr_vs = flxhr.vstart()
    flxhr_sds = SD(k, SDC.READ)
    
    #cldclss = HDF.HDF(j)
    #cldclss_vs = cldclss.vstart()
    cldclss_sds = SD(j, SDC.READ)
    
    # Coordinates
    latitude = vs_read(flxhr_vs, 'Latitude')
    longitude = vs_read(flxhr_vs, 'Longitude')
    
    # Time in seconds since 00:00:00 Jan 1 1993
    time = vs_read(flxhr_vs, 'TAI_start')
    time_since = vs_read(flxhr_vs, 'Profile_time')
       
    # Get time of first data point
    time_seconds = time_since + time
    
    # Solar zenith angle
    sza = vs_read(flxhr_vs, 'Solar_zenith_angle') / 10

    # Albedo
    albedo = vs_read(flxhr_vs, 'Albedo')
    
    # Flags indicating 2B-FLXHR-LIDAR data quality. If 0, then data is of good quality.
    flxhr_quality = vs_read(flxhr_vs, 'Data_quality')
    
    # Surface height bin
    surface_bin = vs_read(flxhr_vs, 'SurfaceHeightBin')
    surface_bin = surface_bin[:,0]
    
# =============================================================================
#     # Flags indicating 2B-CLDCLSS_LIDAR data quality. If 0, then data is of good quality. 
#     cldclss_quality = vs_read(cldclss_vs, 'Data_quality')
# =============================================================================
       
# =============================================================================
#     # the total number of cloud layers by combining radar and lidar measurements 
#     cldclss_cld_layer = vs_read(cldclss_vs, 'Cloudlayer')
# =============================================================================
    
    ###########################################################################
    # Get SDS datasets
    ###########################################################################
    # Cloud impact on long- and shortwave fluxes at the bottom of the atmosphere (i.e. the surface)   
    boacre = flxhr_sds.select('BOACRE')
    cre_sw = boacre.get()[0]
    cre_lw = boacre.get()[1]

    # Downwelling long- and short-wave fluxes in the absence of clouds
    fd_nc = flxhr_sds.select('FD_NC')
    sw_down_cs = fd_nc.get()[0] / 10
    lw_down_cs = fd_nc.get()[1] / 10

    # Downwelling long- and shortwave fluxes for each pixel.
    fd_nc = flxhr_sds.select('FD')
    sw_down_as = fd_nc.get()[0] / 10
    lw_down_as = fd_nc.get()[1] / 10
            		
# =============================================================================
#     # Combined cloud base height
#     cld_base_height = cldclss_sds.select('CloudLayerBase').get()
#     mask = (cld_base_height == -99)
#     cld_base_height[mask] = np.nan
# =============================================================================
    
# =============================================================================
#     # Combined cloud top height
#     cld_top_height = cldclss_sds.select('CloudLayerTop').get()
#     mask = (cld_top_height == -99)
#     cld_top_height[mask] = np.nan
# =============================================================================
    
    # Cloud type for each layer. 
    # 0 = Not determined, 1 = cirrus, 2 = altostratus, 3 = altocumulus, 4 = status, 
    # 5 = stratocumulus, 6 = cumulus, 7 = nimbostratus, 8 = deep convection
    cld_type = cldclss_sds.select('CloudLayerType').get().astype(float)
    mask = (cld_type == -9)
    cld_type[mask] = np.nan
    
    # Cloud phase identified by using CALIPSO feature, temperature, and radar reflectivity
    # 1 - ice, 2 - mixed, 3 - water
    cld_phase = cldclss_sds.select('CloudPhase').get().astype(float)
    mask = (cld_phase == -9)
    cld_phase[mask] = np.nan
    
# =============================================================================
#     # Confidence level assigned to the cloud phase for each layer. It has a 
#     # value ranging from 0 to 10. 10 indicates the highest confidence level. 
#     # If confidence level is below 5, use the cloud phase with a caution.
#     cld_phase_conf = cldclss_sds.select('CloudPhaseConfidenceLevel').get().astype(float)
# =============================================================================
    
# =============================================================================
#     # Cloud fraction within CloudSat foot print determined from CALIPSO lidar measurements.
#     cld_fraction = cldclss_sds.select('CloudFraction').get().astype(float)
#     mask = (cld_fraction == -99)
#     cld_fraction[mask] = np.nan
# =============================================================================
           
    # Find out if single-layer vs. multi-layer cloud type
    cld_type_sorted = np.sort(cld_type,axis=1)
    cld_type_unique = (cld_type_sorted[:,1:] != cld_type_sorted[:,:-1]).sum(axis=1) + 1
    cld_type_single = np.max(cld_type, axis=1)
    mask = (cld_type_unique > 2)
    
    # Flag multi-layered as 9
    cld_type_single[mask] = 9
    
    # Find out if single- or multi-phased cloud
    cld_phase_sorted = np.sort(cld_phase,axis=1)
    cld_phase_unique = (cld_phase_sorted[:,1:] != cld_phase_sorted[:,:-1]).sum(axis=1) + 1
    cld_phase_single = np.max(cld_phase, axis=1)
    mask = (cld_phase_unique > 2)
    
    # Flag multi-layered as 9
    cld_phase_single[mask] = 9
    
    ###########################################################################
    # Extract only values over Greenland
    ###########################################################################
    lat_mask = (latitude > 59)
    long_mask = (longitude < -12) & (longitude > -74)
    valid = (lat_mask == True) & (long_mask == True)
    valid_1d = valid[:,0]
    
    valid_lon = longitude[valid]
    valid_lat = latitude[valid]
    valid_time = time_seconds[valid]
    valid_sza = sza[valid]
    valid_albedo = albedo[valid]
    valid_flxhr_quality = flxhr_quality[valid]
    valid_cre_sw = cre_sw[valid_1d]
    valid_cre_lw = cre_lw[valid_1d]
    #valid_cld_layer = cldclss_cld_layer[valid]
    valid_cld_type = cld_type_single[valid_1d]
    valid_cld_phase = cld_phase_single[valid_1d]
      
    # Get row corresponding to surface bin
    valid_sw_down_cs = sw_down_cs[np.arange(len(sw_down_cs)), surface_bin][valid_1d]
    valid_lw_down_cs = lw_down_cs[np.arange(len(lw_down_cs)), surface_bin][valid_1d]
    valid_sw_down_as = sw_down_as[np.arange(len(sw_down_as)), surface_bin][valid_1d]
    valid_lw_down_as = lw_down_as[np.arange(len(lw_down_as)), surface_bin][valid_1d]
    
    # Put into DataFrame
    df = pd.DataFrame(list(zip(valid_lon, valid_lat, valid_time, valid_sza, valid_albedo,
                               valid_flxhr_quality, valid_cre_sw, valid_cre_lw, 
                               valid_sw_down_cs, valid_lw_down_cs, valid_sw_down_as, 
                               valid_lw_down_as, valid_cld_type, valid_cld_phase)))
    
    if df.shape[0] > 0:
        df.columns = ['lon', 'lat', 'time', 'sza_cloudsat', 'albedo_cloudsat', 
                      'quality', 'cre_sw', 'cre_lw', 'sw_down_cs', 'lw_down_cs', 
                      'sw_down_as', 'lw_down_as', 'cloud_type', 'cloud_phase_cloudsat']
        
        # Set some stuff to NaN
        df.loc[(df['sw_down_cs'] == -999), 'sw_down_cs'] = np.nan
        df.loc[(df['sw_down_as'] == -999), 'sw_down_as'] = np.nan
        df.loc[(df['lw_down_cs'] == -999), 'lw_down_cs'] = np.nan
        df.loc[(df['lw_down_as'] == -999), 'lw_down_as'] = np.nan
        df.loc[(df['cre_sw'] == -999), 'cre_sw'] = np.nan
        df.loc[(df['cre_lw'] == -999), 'cre_lw'] = np.nan
        
        # Some more filtering
        df.loc[((df['lw_down_cs']) > 1500), 'lw_down_cs'] = np.nan
        df.loc[((df['lw_down_cs']) < 0), 'lw_down_cs'] = np.nan
        
        df.loc[((df['lw_down_as']) > 1500), 'lw_down_as'] = np.nan
        df.loc[((df['lw_down_as']) < 0), 'lw_down_as'] = np.nan
        
        df.loc[((df['sw_down_cs']) > 1500), 'sw_down_cs'] = np.nan
        df.loc[((df['sw_down_cs']) < 0), 'sw_down_cs'] = np.nan
        
        df.loc[((df['sw_down_as']) > 1500), 'sw_down_as'] = np.nan
        df.loc[((df['sw_down_as']) < 0), 'sw_down_as'] = np.nan
        
        # Remove columns if all NaN
        df['Sum'] = pd.notnull(df['cre_sw']).astype(int) + pd.notnull(df['cre_lw']).astype(int) +\
            pd.notnull(df['sw_down_cs']).astype(int) + pd.notnull(df['sw_down_as']).astype(int) +\
                pd.notnull(df['lw_down_cs']).astype(int) + pd.notnull(df['lw_down_as']).astype(int)
        df = df[df['Sum'] > 1]
        
        # Add datetime column
        epoch = dt.datetime(1993,1,1).timestamp()
        time_values = []
        for j in range(df['time'].shape[0]):
            time_values.append(dt.datetime.fromtimestamp(df['time'].iloc[j] + epoch))
        df['datetime'] = pd.to_datetime(time_values)
        
        # Remove some columns
        df.drop('Sum', axis=1, inplace=True)
        df.drop('time', axis=1, inplace=True)
    
        return df
    
    else:
        
        return pd.DataFrame()

def sds_read(dataset, variable):
    
    # Read SDS
    sds = dataset.select(variable) 
    sds_scale = sds.attributes()['scale_factor']
    sds_offset = sds.attributes()['add_offset']
    
    mask = (sds.get() == sds.attributes()['_FillValue'])
    sds_float = sds.get().astype('float')
    sds_float[mask] = np.nan
    
    return (sds_float - sds_offset) * sds_scale

    
def MOD06_Read(k, df):

    # Read MODIS file
    f = SD(k, SDC.READ)
    
    # Get datasets
    sds_lat = f.select('Latitude')
    latitude = sds_lat.get()
    
    sds_lon = f.select('Longitude')
    longitude = sds_lon.get()
    
    # Check if file is contained within CloudSat profile
    index = (df['lon'] > longitude.min()) & (df['lat'] > latitude.min())\
    & (df['lon'] < longitude.max()) & (df['lat'] < latitude.max())
    
    if index.sum() > 0:
    
        # Solar Zenith Angle, Cell to Sun 
        sza = sds_read(f, 'Solar_Zenith') 
        
        # Cloud Top Height at 1-km resolution from LEOCAT, Geopotential Height at 
        # Retrieved Cloud Top Pressure Level rounded to nearest 50 m
        cth = sds_read(f, 'cloud_top_height_1km')
        
        # Cloud Top Temperature at 1-km resolution from LEOCAT, Temperature from 
        # Ancillary Data at Retrieved Cloud Top Pressure Level
        ctt = sds_read(f, 'cloud_top_temperature_1km')
        
        # Cloud Top Pressure at 1-km resolution from LEOCAT, Cloud Top Pressure 
        # Level rounded to nearest 5 mb
        ctp = sds_read(f, 'cloud_top_pressure_1km')
        
        # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
        cpop = sds_read(f, 'Cloud_Phase_Optical_Properties')
        
        # Cloud Optical Thickness two-channel retrieval using band 7(2.1um) and 
        # either band 1(0.65um), 2(0.86um), or 5(1.2um)
        cot = sds_read(f, 'Cloud_Optical_Thickness')
        
        # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
        cer = sds_read(f, 'Cloud_Effective_Radius')
        
        # Cloud Emissivity at 1-km resolution from LEOCAT Cloud Top Pressure Retrieval
        ce = sds_read(f, 'cloud_emissivity_1km')
        
        # Column Water Path two-channel retrieval using band 7(2.1um) and either 
        # band 1(0.65um), 2(0.86um), or 5(1.2um)
        cwp = sds_read(f, 'Cloud_Water_Path')
        
        # Cloud Mask, L2 MOD06 QA Plan
        cloud_mask = sds_read(f, 'Cloud_Mask_1km')
        
        # Interpolate some attributes from 5 km to 1 km
        grid_x_1km, grid_y_1km = np.meshgrid(np.arange(0, cwp.shape[1], 1), 
                                     np.linspace(0, cwp.shape[0]-5, 
                                                 cwp.shape[0]))
        
        grid_x_5km, grid_y_5km = np.meshgrid(np.arange(0, cwp.shape[1], 5), 
                                     np.arange(0, cwp.shape[0], 5))
        
        latitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                           np.ravel(latitude), (grid_x_1km, grid_y_1km), 
                           method='linear')
        longitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                           np.ravel(longitude), (grid_x_1km, grid_y_1km), 
                           method='linear')
        sza_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                           np.ravel(sza), (grid_x_1km, grid_y_1km), 
                           method='linear')

        # Ravel and put in big DataFrame
        modis_df = pd.DataFrame(list(zip(longitude_1km.flatten(), latitude_1km.flatten(),
                                         sza_1km.flatten(), cth.flatten(),
                                         ctt.flatten(), ctp.flatten(),
                                         cpop.flatten(), cot.flatten(),
                                         cer.flatten(), ce.flatten(),
                                         cwp.flatten(), cloud_mask.flatten())))
        modis_df.columns = ['lon', 'lat', 'sza_modis', 'cloud_top_height', 'cloud_top_temperature',
                            'cloud_top_pressure', 'cloud_phase_modis', 'cloud_optical_thickness',
                           'cloud_effective_radius', 'cloud_emissivity',  'cloud_water_path',
                           'cloud_mask']
        
        return modis_df
    
    else:
        
        return pd.DataFrame()

def MOD10_Read(files, df):

    # Define lat and lons of tile
    grid_cell = 463.31271653
    upper_left_grid = (-20015109.354, 10007554.677)
    lower_right_grid = (20015109.354, -10007554.677)
    
    t = []
    for k in files:
        
        # Get path and filename separately
        infilepath, infilename = os.path.split(k)
        # Get file name without extension
        infileshortname, extension = os.path.splitext(infilename)
        
        # Get tile number 
        tile = infileshortname[18:23]  
        
        ##########################################################################
        # Check if DataFrame is within MODIS tile
        ##########################################################################        
        upper_left_corner = (upper_left_grid[0] + (2400*int(tile[0:2])*grid_cell),
                             upper_left_grid[1] - (2400*int(tile[3:])*grid_cell))
        lower_right_corner = (upper_left_corner[0] + (2400*grid_cell),
                             (upper_left_corner[1] - (2400*grid_cell)))
        
        x = np.linspace(upper_left_corner[0], lower_right_corner[0], 2400)
        y = np.linspace(upper_left_corner[1], lower_right_corner[1], 2400)
        
        # Produce grid
        xv, yv = np.meshgrid(x, y)
        
        # Define MODIS grid in pyproj
        modis_grid = Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
        
        # Convert to lat, lons
        lon, lat = modis_grid(xv, yv, inverse=True)
        
        # Check if file is contained within CloudSat profile
        index = (df['lon'] > lon.min()) & (df['lat'] > lat.min())\
        & (df['lon'] < lon.max()) & (df['lat'] < lat.max())
        
        # If index is higher than zero, return DataFrame
        if index.sum() > 0:
            
            # Read MODIS file
            f = SD(k, SDC.READ)
            
            # Append
            t.append(tile)
            
            # Get datasets
            sds_albedo = f.select('Snow_Albedo_Daily_Tile')
            albedo = sds_albedo.get().astype('float')
            
            # Define grid using pyresample       
            grid = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
            
            # Define some sample points
            my_lons = df[index]['lon'].values
            my_lats = df[index]['lat'].values
            swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)
            
            # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
            data_swath = pyresample.kd_tree.resample_nearest(source_geo_def=grid, 
                                                             target_geo_def=swath, 
                                                             data=albedo, 
                                                             radius_of_influence=50000)
            
            # Make new column
            df[tile] = index.astype(int)
            df.loc[(df[tile] == 1), tile] = data_swath
        else:
            pass
            
    # Get single albedo value
    df['albedo_modis'] = df[t].max(axis=1)
    
    # Remove columns
    df = df.drop(t, axis=1)
    
    # Filter
    df.loc[(df['albedo_modis'] == 0), 'albedo_modis'] = np.nan
    df.loc[(df['albedo_modis'] > 100), 'albedo_modis'] = np.nan
    df.loc[(df['albedo_modis'] > 84), 'albedo_modis'] = 84
            
    return df
        
def ISMIP_Read(lon, lat, elev, mask, df):  
    
    
    # Define grid using pyresample       
    grid = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
    
    # Define some sample points
    my_lons = df['lon'].values
    my_lats = df['lat'].values
    swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)
    
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    elev_swath = pyresample.kd_tree.resample_nearest(source_geo_def=grid, 
                                                     target_geo_def=swath, 
                                                     data=elev, 
                                                     radius_of_influence=50000)
            
    mask_swath = pyresample.kd_tree.resample_nearest(source_geo_def=grid, 
                                                     target_geo_def=swath, 
                                                     data=mask, 
                                                     radius_of_influence=50000)
    
    df['elev'] = elev_swath
    df['mask'] = mask_swath
    
    return df

def Regions_Read(x, y, regions_masks, df):  
    
    # Convert from stereographic to WGS84
    transformer = Transformer.from_crs(3413, 4326)
    lat, lon = transformer.transform(x, y)
    
    # Define grid using pyresample       
    grid = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
    
    # Define some sample points
    my_lons = df['lon'].values
    my_lats = df['lat'].values
    swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)
    
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    region_swath = pyresample.kd_tree.resample_nearest(source_geo_def=grid, 
                                                     target_geo_def=swath, 
                                                     data=regions_masks, 
                                                     radius_of_influence=50000)
    
    
    df['region'] = region_swath
    
    return df
    

def MODIS_Monthly_Read(mod_file, df, month):  
    
    # Read NetCDF
    modis = netCDF4.Dataset(mod_file)
    x = modis.variables['x'][:]
    y = modis.variables['y'][:]
    albedo = modis.variables['albedo'][month, :, :]
    xx, yy = np.meshgrid(x, y)

    # Convert from stereographic to WGS84
    transformer = Transformer.from_crs(3413, 4326)
    lat, lon = transformer.transform(xx, yy)
    
    # Define grid using pyresample       
    grid = pyresample.geometry.GridDefinition(lons=lon, lats=lat)
    
    # Define some sample points
    my_lons = df['lon'].values
    my_lats = df['lat'].values
    swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)
    
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    region_swath = pyresample.kd_tree.resample_nearest(source_geo_def=grid, 
                                                     target_geo_def=swath, 
                                                     data=albedo, 
                                                     radius_of_influence=50000)
    
    
    df['albedo_modis'] = region_swath
    
    return df


def data_read_machine_learning(files):
    
    # Combine data   
    cloud_type = []
    cloud_phase = []
    sza_cloudsat = []
    region = []
    elev = []
    sw_down_cs = []
    sw_down_as = []
    lw_down_cs = []
    lw_down_as = []
    
    modis_cot = []
    modis_ctp = []
    modis_phase = []
    modis_cer = []
    modis_ctt = []
    modis_cth = []
    modis_cwp = []
    
    modis_albedo = []
    cloudsat_albedo = []
    t2m = []
    d2m = []
    ssrdc = []
    strdc = []
    
    lat = []
    lon = []
    
    for i in files:
        
        # Read data
        df = pd.read_csv(i)
        
        # Get path and filename separately
        infilepath1, infilename1 = os.path.split(i)
        # Get file name without extension
        infileshortname1, extension1 = os.path.splitext(infilename1)
        
        # Append
        lat.append(df['lat'].values)
        lon.append(df['lon'].values)
        sza_cloudsat.append(df['sza_cloudsat'].values)
        cloud_type.append(df['cloud_type'].values)
        region.append(df['region'].values)
        elev.append(df['elev'].values)
        sw_down_as.append(df['sw_down_as'].values)
        sw_down_cs.append(df['sw_down_cs'].values)
        lw_down_as.append(df['lw_down_as'].values)
        lw_down_cs.append(df['lw_down_cs'].values)
        cloud_phase.append(df['cloud_phase_cloudsat'].values)
        modis_cot.append(df['cloud_optical_thickness'].values)
        modis_ctp.append(df['cloud_top_pressure'].values)
        modis_phase.append(df['cloud_phase_modis'].values)
        modis_cer.append(df['cloud_effective_radius'].values)
        modis_ctt.append(df['cloud_top_temperature'].values)
        modis_cth.append(df['cloud_top_height'].values)
        modis_cwp.append(df['cloud_water_path'].values)
        modis_albedo.append(df['albedo_modis'].values)
        cloudsat_albedo.append(df['albedo_cloudsat'].values)
        t2m.append(df['t2m'].values)
        d2m.append(df['d2m'].values)
        ssrdc.append(df['ssrdc'].values)
        strdc.append(df['strdc'].values)
        
    lat_flat = [item for sublist in lat for item in sublist]
    lon_flat = [item for sublist in lon for item in sublist]
    cloud_type_flat = [item for sublist in cloud_type for item in sublist]
    cloud_phase_flat = [item for sublist in cloud_phase for item in sublist]
    sza_flat = [item for sublist in sza_cloudsat for item in sublist]
    region_flat = [item for sublist in region for item in sublist]
    elev_flat = [item for sublist in elev for item in sublist]
    sw_down_cs_flat = [item for sublist in sw_down_cs for item in sublist]
    sw_down_as_flat = [item for sublist in sw_down_as for item in sublist]
    lw_down_cs_flat = [item for sublist in lw_down_cs for item in sublist]
    lw_down_as_flat = [item for sublist in lw_down_as for item in sublist]
    
    modis_cot_flat = [item for sublist in modis_cot for item in sublist]
    modis_ctp_flat = [item for sublist in modis_ctp for item in sublist]
    modis_phase_flat = [item for sublist in modis_phase for item in sublist]
    modis_cer_flat = [item for sublist in modis_cer for item in sublist]
    modis_ctt_flat = [item for sublist in modis_ctt for item in sublist]
    modis_cth_flat = [item for sublist in modis_cth for item in sublist]
    modis_cwp_flat = [item for sublist in modis_cwp for item in sublist]
    
    modis_albedo_flat = [item for sublist in modis_albedo for item in sublist]
    cloudsat_albedo_flat = [item for sublist in cloudsat_albedo for item in sublist]
    t2m_flat = [item for sublist in t2m for item in sublist]
    d2m_flat = [item for sublist in d2m for item in sublist]
    ssrdc_flat = [item for sublist in ssrdc for item in sublist]
    strdc_flat = [item for sublist in strdc for item in sublist]

    # Put into DataFrame
    df = pd.DataFrame(list(zip(lon_flat, lat_flat, sza_flat,cloud_type_flat,
                               cloud_phase_flat,region_flat,
                               elev_flat, sw_down_cs_flat,sw_down_as_flat, 
                               lw_down_cs_flat,lw_down_as_flat,modis_cot_flat,
                               modis_ctp_flat,modis_phase_flat,modis_cer_flat,
                               modis_ctt_flat,modis_cth_flat, modis_cwp_flat,
                               modis_albedo_flat,cloudsat_albedo_flat, t2m_flat,
                               d2m_flat, ssrdc_flat, strdc_flat)))
    
    df.columns = ['lon', 'lat', 'sza', 'type', 'phase', 'region', 'elev','sw_cs', 'sw_as',
                  'lw_cs', 'lw_as','modis_cot', 'modis_ctp', 'modis_phase',
                  'modis_cer', 'modis_ctt', 'modis_cth', 'modis_cwp', 
                  'modis_albedo','cloudsat_albedo', 't2m', 'd2m', 'ssrdc', 'strdc']
    
    # Remove rows with no data
    df = df.dropna()
    
    # Remove rows with spurious longwave data
    df = df[df['lw_as'] < 400]
    df = df[df['lw_as'] > 150]
    df = df[df['lw_cs'] != 0]
    df = df[df['lw_cs'] > 150]
    
    # Remove rows with spurious shortwave data
    df = df[df['sw_as'] != 0]
    df = df[df['sw_cs'] != 0]
    
    # Remove if cloud detected but no effect on radiative fluxes
    clearsky = df[df['type'] == 0]
    valid = df[(df['lw_cs'] != df['lw_as']) | (df['sw_cs'] != df['sw_as'])]
    df = pd.concat((clearsky, valid))
    
    # Add factor column
    df['f_sw'] = np.divide(df['sw_as'], df['sw_cs'])
    df['f_lw'] = np.divide(df['lw_as'], df['lw_cs'])

    return df

def data_read_explore(files):
    
    # Combine data   
    cloud_type = []
    cloud_phase = []
    sza_cloudsat = []
    region = []
    elev = []
    sw_down_cs = []
    sw_down_as = []
    lw_down_cs = []
    lw_down_as = []

    modis_albedo = []
    t2m = []
    
    for i in files:
        
        # Read data
        df = pd.read_csv(i, parse_dates=(['datetime']))
        
        # Append
        sza_cloudsat.append(df['sza_cloudsat'].values)
        cloud_type.append(df['cloud_type'].values)
        region.append(df['region'].values)
        elev.append(df['elev'].values)
        sw_down_as.append(df['sw_down_as'].values)
        sw_down_cs.append(df['sw_down_cs'].values)
        lw_down_as.append(df['lw_down_as'].values)
        lw_down_cs.append(df['lw_down_cs'].values)
        cloud_phase.append(df['cloud_phase_cloudsat'].values)
        modis_albedo.append(df['albedo_modis'].values)
        t2m.append(df['t2m'].values)
        
        
    cloud_type_flat = [item for sublist in cloud_type for item in sublist]
    cloud_phase_flat = [item for sublist in cloud_phase for item in sublist]
    sza_flat = [item for sublist in sza_cloudsat for item in sublist]
    region_flat = [item for sublist in region for item in sublist]
    elev_flat = [item for sublist in elev for item in sublist]
    sw_down_cs_flat = [item for sublist in sw_down_cs for item in sublist]
    sw_down_as_flat = [item for sublist in sw_down_as for item in sublist]
    lw_down_cs_flat = [item for sublist in lw_down_cs for item in sublist]
    lw_down_as_flat = [item for sublist in lw_down_as for item in sublist]
        
    modis_albedo_flat = [item for sublist in modis_albedo for item in sublist]
    t2m_flat = [item for sublist in t2m for item in sublist]
    
    # Put into DataFrame
    df = pd.DataFrame(list(zip(sza_flat,cloud_type_flat,cloud_phase_flat,region_flat,
                               elev_flat, sw_down_cs_flat,sw_down_as_flat, 
                               lw_down_cs_flat,lw_down_as_flat,
                               modis_albedo_flat, t2m_flat)))
    
    df.columns = ['sza', 'type', 'phase', 'region', 'elev','sw_cs', 'sw_as',
                  'lw_cs', 'lw_as', 'modis_albedo','t2m']
    
    # Remove rows with no data
    df = df.dropna()
    
    # Remove rows with spurious longwave data
    df = df[df['lw_as'] < 400]
    df = df[df['lw_as'] > 150]
    df = df[df['lw_cs'] != 0]
    df = df[df['lw_cs'] > 150]
    
    # Remove if cloud detected but no effect on radiative fluxes
    clearsky = df[df['type'] == 0]
    valid = df[(df['lw_cs'] != df['lw_as']) | (df['sw_cs'] != df['sw_as'])]
    df = pd.concat((clearsky, valid))
    
    # Add factor column
    df['f_sw'] = np.divide(df['sw_as'], df['sw_cs'])
    df['f_lw'] = np.divide(df['lw_as'], df['lw_cs'])

    return df

def data_time_extract(files):
    
    # Define column to read data       
    time = []
    
    for i in files:
        
        # Read data
        df = pd.read_csv(i, parse_dates=(['datetime']))
        
        # Append
        time.append(df['datetime'].iloc[0].time())
           
    # Put into DataFrame
    df = pd.DataFrame(list(zip(time)))
    
    # Rename columns
    df.columns = ['time']
       
    return df

def MYD06_L2_Read_LW(myd, era_t, era_fluxes, era_time, era_xx, era_yy, modis_time):
    
    # Read MODIS file
    f = SD(myd, SDC.READ)
    
    # Get datasets
    sds_lat = f.select('Latitude')
    latitude = sds_lat.get()
    
    sds_lon = f.select('Longitude')
    longitude = sds_lon.get()

    # Cloud Top Height at 1-km resolution from LEOCAT, Geopotential Height at 
    # Retrieved Cloud Top Pressure Level rounded to nearest 50 m
    cth = sds_read(f, 'cloud_top_height_1km')
    
    # Cloud Top Temperature at 1-km resolution from LEOCAT, Temperature from 
    # Ancillary Data at Retrieved Cloud Top Pressure Level
    ctt = sds_read(f, 'cloud_top_temperature_1km')
    
    # Cloud Top Pressure at 1-km resolution from LEOCAT, Cloud Top Pressure 
    # Level rounded to nearest 5 mb
    ctp = sds_read(f, 'cloud_top_pressure_1km')
    
    # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
    phase = sds_read(f, 'Cloud_Phase_Optical_Properties')
    
    # Cloud Optical Thickness two-channel retrieval using band 7(2.1um) and 
    # either band 1(0.65um), 2(0.86um), or 5(1.2um)
    cot = sds_read(f, 'Cloud_Optical_Thickness')
    
    # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
    cer = sds_read(f, 'Cloud_Effective_Radius')
    
    # Column Water Path two-channel retrieval using band 7(2.1um) and either 
    # band 1(0.65um), 2(0.86um), or 5(1.2um)
    cwp = sds_read(f, 'Cloud_Water_Path')
    
    # Interpolate some attributes from 5 km to 1 km    
    grid_x_1km, grid_y_1km = np.meshgrid(np.linspace(0, latitude.shape[1]-1, cwp.shape[1]), 
                                 np.linspace(0, latitude.shape[0]-1, cwp.shape[0]))
    
    grid_x_5km, grid_y_5km = np.meshgrid(np.arange(0, latitude.shape[1], 1), 
                                 np.arange(0, latitude.shape[0], 1))
    
    latitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(latitude), (grid_x_1km, grid_y_1km), 
                       method='linear')
    longitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(longitude), (grid_x_1km, grid_y_1km), 
                       method='linear')

    # Regrid ERA5 to MODIS grid
    era_idx = np.abs(era_time['datetime'] - modis_time).argmin()
    era_t2m = era_t.variables['t2m'][era_idx, :, :]
    era_strdc = era_fluxes.variables['strdc'][era_idx, :, :] / 3600
  
    era_grid = pyresample.geometry.GridDefinition(lons=era_xx, lats=era_yy)
    modis_grid = pyresample.geometry.GridDefinition(lons=longitude_1km, lats=latitude_1km)
   
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    t2m = pyresample.kd_tree.resample_nearest(source_geo_def=era_grid, 
                                                     target_geo_def=modis_grid, 
                                                     data=era_t2m, 
                                                     radius_of_influence=50000)
    strdc = pyresample.kd_tree.resample_nearest(source_geo_def=era_grid, 
                                                     target_geo_def=modis_grid, 
                                                     data=era_strdc, 
                                                     radius_of_influence=50000)
    
    # Put into DataFrame
    df = pd.DataFrame(list(zip(np.ravel(longitude_1km), np.ravel(latitude_1km),
                               np.ravel(cot), np.ravel(ctp),
                               np.ravel(phase), np.ravel(cer),
                               np.ravel(ctt), np.ravel(cth),
                               np.ravel(cwp), np.ravel(t2m), np.ravel(strdc))))
    df.columns = ['lon', 'lat', 'modis_cot', 'modis_ctp', 'modis_phase', 'modis_cer', 
        'modis_ctt', 'modis_cth', 'modis_cwp', 't2m', 'strdc']
    
    # Set values to NaN
    df = df[df['t2m'] != 0]
    
    # Get clear-sky values
    df_clearsky = df[df['modis_phase'] == 1]
    df_clearsky.insert(0, 'value', 1)
    df_clearsky.insert(0, 'clearsky_lw', (0.9 * df_clearsky['strdc']) + 25.7)
    df_clearsky.insert(0, 'allsky_lw', df_clearsky['clearsky_lw'])
    new_df_clearsky = df_clearsky[['lon', 'lat', 'value', 'clearsky_lw', 'allsky_lw']]
    
    # Drop rows with NaNs
    df.dropna(inplace=True)

    return df, new_df_clearsky

def MYD06_L2_Read_SW(myd, era_t, era_fluxes, era_time, era_xx, era_yy, modis_time):
    
    # Read MODIS file
    f = SD(myd, SDC.READ)
    
    # Get datasets
    sds_lat = f.select('Latitude')
    latitude = sds_lat.get()
    
    sds_lon = f.select('Longitude')
    longitude = sds_lon.get()
    
    # Solar Zenith Angle, Cell to Sun 
    sza = sds_read(f, 'Solar_Zenith') 

    # Cloud Top Height at 1-km resolution from LEOCAT, Geopotential Height at 
    # Retrieved Cloud Top Pressure Level rounded to nearest 50 m
    cth = sds_read(f, 'cloud_top_height_1km')
    
    # Cloud Top Temperature at 1-km resolution from LEOCAT, Temperature from 
    # Ancillary Data at Retrieved Cloud Top Pressure Level
    ctt = sds_read(f, 'cloud_top_temperature_1km')
    
    # Cloud Top Pressure at 1-km resolution from LEOCAT, Cloud Top Pressure 
    # Level rounded to nearest 5 mb
    ctp = sds_read(f, 'cloud_top_pressure_1km')
    
    # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
    phase = sds_read(f, 'Cloud_Phase_Optical_Properties')
    
    # Cloud Optical Thickness two-channel retrieval using band 7(2.1um) and 
    # either band 1(0.65um), 2(0.86um), or 5(1.2um)
    cot = sds_read(f, 'Cloud_Optical_Thickness')
    
    # Cloud Phase Determination Used in Optical Thickness/Effective Radius Retrieval
    cer = sds_read(f, 'Cloud_Effective_Radius')
    
    # Column Water Path two-channel retrieval using band 7(2.1um) and either 
    # band 1(0.65um), 2(0.86um), or 5(1.2um)
    cwp = sds_read(f, 'Cloud_Water_Path')
    
    # Interpolate some attributes from 5 km to 1 km    
    grid_x_1km, grid_y_1km = np.meshgrid(np.linspace(0, latitude.shape[1]-1, cwp.shape[1]), 
                                 np.linspace(0, latitude.shape[0]-1, cwp.shape[0]))
    
    grid_x_5km, grid_y_5km = np.meshgrid(np.arange(0, latitude.shape[1], 1), 
                                 np.arange(0, latitude.shape[0], 1))
    
    latitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(latitude), (grid_x_1km, grid_y_1km), 
                       method='linear')
    longitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(longitude), (grid_x_1km, grid_y_1km), 
                       method='linear')
    sza_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                   np.ravel(sza), (grid_x_1km, grid_y_1km), 
                   method='linear')

    # Regrid ERA5 to MODIS grid
    era_idx = np.abs(era_time['datetime'] - modis_time).argmin()
    era_t2m = era_t.variables['t2m'][era_idx, :, :]
    era_ssrdc = era_fluxes.variables['ssrdc'][era_idx, :, :] / 3600
  
    era_grid = pyresample.geometry.GridDefinition(lons=era_xx, lats=era_yy)
    modis_grid = pyresample.geometry.GridDefinition(lons=longitude_1km, lats=latitude_1km)
   
    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    t2m = pyresample.kd_tree.resample_nearest(source_geo_def=era_grid, 
                                                     target_geo_def=modis_grid, 
                                                     data=era_t2m, 
                                                     radius_of_influence=50000)
    ssrdc = pyresample.kd_tree.resample_nearest(source_geo_def=era_grid, 
                                                     target_geo_def=modis_grid, 
                                                     data=era_ssrdc, 
                                                     radius_of_influence=50000)
                      
    # Put into DataFrame
    df = pd.DataFrame(list(zip(np.ravel(longitude_1km[50:-50, 50:-50]), np.ravel(latitude_1km[50:-50, 50:-50]),
                               np.ravel(sza_1km[50:-50, 50:-50]), np.ravel(cot[50:-50, 50:-50]),
                               np.ravel(ctp[50:-50, 50:-50]), np.ravel(phase[50:-50, 50:-50]), 
                               np.ravel(cer[50:-50, 50:-50]), np.ravel(ctt[50:-50, 50:-50]), 
                               np.ravel(cth[50:-50, 50:-50]), np.ravel(cwp[50:-50, 50:-50]), 
                               np.ravel(t2m[50:-50, 50:-50]), np.ravel(ssrdc[50:-50, 50:-50]))))
    df.columns = ['lon', 'lat', 'sza', 'modis_cot', 'modis_ctp', 'modis_phase', 
                  'modis_cer', 'modis_ctt', 'modis_cth', 'modis_cwp', 't2m', 'ssrdc']
    
    # Filter NaN values
    df = df[df['t2m'] != 0]
    
    # Remove solar zenith angles > 85
    df = df[df['sza'] < 85]
    
    # Get clear-sky values
    df_clearsky = df[df['modis_phase'] == 1]
       
    # Drop rows with NaNs
    df.dropna(inplace=True)
    
    return df, df_clearsky


def MYD06_L2_Read(myd, attribute):
    
    # Read MODIS file
    f = SD(myd, SDC.READ)
    
    # Get datasets
    sds_lat = f.select('Latitude')
    latitude = sds_lat.get()
    
    sds_lon = f.select('Longitude')
    longitude = sds_lon.get()
    
    att = sds_read(f, attribute)
    
    # Interpolate some attributes from 5 km to 1 km    
    grid_x_1km, grid_y_1km = np.meshgrid(np.linspace(0, latitude.shape[1]-1, att.shape[1]), 
                                 np.linspace(0, latitude.shape[0]-1, att.shape[0]))
    
    grid_x_5km, grid_y_5km = np.meshgrid(np.arange(0, latitude.shape[1], 1), 
                                 np.arange(0, latitude.shape[0], 1))
    
    latitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(latitude), (grid_x_1km, grid_y_1km), 
                       method='linear')
    longitude_1km = griddata((np.ravel(grid_x_5km), np.ravel(grid_y_5km)), 
                       np.ravel(longitude), (grid_x_1km, grid_y_1km), 
                       method='linear')
    
    data = np.dstack((latitude_1km[50:-50, 50:-50], longitude_1km[50:-50, 50:-50], 
                  att[50:-50, 50:-50]))

    return data
       
