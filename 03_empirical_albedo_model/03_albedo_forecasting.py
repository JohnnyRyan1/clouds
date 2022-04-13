#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Produce albedo grids into the future using empirical relationships with CMIP6.

"""

# Import modules
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters
from skimage import morphology
import scipy.ndimage as ndimage
from scipy import interpolate

# Define path
path = '/Users/jryan4/Dropbox (University of Oregon)/projects/clouds/data/'

# Define summer albedo and temp. data
ds = xr.open_dataset(path + '/temp_albedo_summer_climatologies.nc')

# Define some functions
def new_linregress(x, y):
    # Wrapper around scipy linregress to use in apply_ufunc
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return np.array([slope, intercept, r_value, p_value, std_err])


def remove_pepper(array, threshold):
    
    """ 
    Function to remove peppper effects
    
    """
    
    # Convert to binary
    array_bool = np.isfinite(array).astype(int)
    
    # Label
    id_regions = morphology.label(array_bool, background = 0, connectivity = 2)
    
    # Number of regions
    num_ids = id_regions.max()
    
    # Compute region sizes in pixels
    id_sizes = np.array(ndimage.sum(array_bool, id_regions, range(num_ids + 1)))
    
    # Filter out regions smaller than threshold
    area_mask = (id_sizes < threshold)
    array[area_mask[id_regions]] = np.nan
    
    return array

def remove_salt(array, threshold):
    
    """
    Function to remove salt effects
    
    """
    
    # Convert to binary
    array_bool = np.isnan(array).astype(int)
    
    # Label
    id_regions = morphology.label(array_bool, background = 0, connectivity = 2)
    
    # Number of regions
    num_ids = id_regions.max()
    
    # Compute region sizes in pixels
    id_sizes = np.array(ndimage.sum(array_bool, id_regions, range(num_ids + 1)))
    
    # Assign regions larger than the threshold a value of 1
    area_mask = (id_sizes > threshold)

    array[area_mask[id_regions]] = 1
    
    return array

def linear_interpolate(array):
    
    """
    Function to linearly intepolate NaNs in array
    
    """
    masked_array = np.ma.masked_invalid(array)

    x = np.arange(0, masked_array.shape[1])
    y = np.arange(0, masked_array.shape[0])
    xx, yy = np.meshgrid(x, y)

    #get only the valid values
    x1 = xx[~masked_array.mask]
    y1 = yy[~masked_array.mask]

    newarr = masked_array[~masked_array.mask]

    linear = interpolate.griddata((x1, y1), newarr.ravel(),
                              (xx, yy), method='nearest')
    
    return linear

###############################################################################
# Compute linear relationship between albedo and air temperature for every pixel
###############################################################################
xr_stats = xr.apply_ufunc(new_linregress, ds['t2m'], 
                          ds['albedo'],
                          input_core_dims=[['z'], ['z']],
                          output_core_dims=[["parameter"]],
                          vectorize=True,
                          dask="parallelized",
                          output_dtypes=['float64'],
                          output_sizes={"parameter": 5})

# Copy 
crop = np.copy(xr_stats)


# Tests
west1, west2, west3, west4 = 1700, 1900, 350, 650
north1, north2, north3, north4 = 2200, 2400, 250, 450

# Crop an area for testing
west = linear_slope[west1:west2, west3:west4]
north = linear_slope[north1:north2, north3:north4]

###############################################################################
# Do filtering
###############################################################################

# Convert non-significant relationships to NaN
crop[crop[:,:,3] > 0.05] = np.nan

# Remove unrealistic slopes > 0
crop[crop[:,:,0] > 0] = np.nan

# Remove pepper effects
crop_slope = remove_pepper(crop[:,:,0], 5000)
crop_inter = remove_pepper(crop[:,:,1], 5000)

# Median filter
selem = morphology.square(20)
median_slope = filters.median(crop_slope, selem=selem)
median_intercept = filters.median(crop_inter, selem=selem)

# Morphological closing 
close_slope = morphology.closing(median_slope)
close_intercept = morphology.closing(median_intercept)

# Remove salt effects
salt_slope = remove_salt(close_slope, 5)
salt_inter = remove_salt(close_intercept, 5)

# Linearly interpolate NaNs
linear_slope = linear_interpolate(salt_slope)
linear_inter = linear_interpolate(salt_inter)

# Convert 1s back to NaNs
linear_slope[linear_slope == 1] = np.nan 
linear_inter[linear_inter == 1] = np.nan 

###############################################################################
# Save 1 km dataset to NetCDF
###############################################################################
dataset = netCDF4.Dataset(path + 'empirical_albedo_model.nc', 
                          'w', format='NETCDF4_CLASSIC')
print('Creating... %s' % path + 'empirical_albedo_model.nc')
dataset.Title = "Slopes and intercepts for temperature vs. albedo relationship"
import time
dataset.History = "Created " + time.ctime(time.time())
dataset.Projection = "WGS 84"
dataset.Reference = "Ryan, J. C., Smith. L. C., Cooley, S. W., and Pearson, B. (in review), Emerging importance of clouds for Greenland Ice Sheet energy balance and meltwater production."
dataset.Contact = "jryan4@uoregon.edu"
    
# Create new dimensions
lat_dim = dataset.createDimension('y', linear_inter.shape[0])
lon_dim = dataset.createDimension('x', linear_inter.shape[1])

# Define variable types
Y = dataset.createVariable('latitude', np.float32, ('y','x'))
X = dataset.createVariable('longitude', np.float32, ('y','x'))
    
# Define units
Y.units = "degrees"
X.units = "degrees"
   
# Create the actual 3D variable
slope_nc = dataset.createVariable('slope', np.float32, ('y','x'))
inter_nc = dataset.createVariable('intercept', np.float32, ('y','x'))
temp_mean_nc = dataset.createVariable('mean_temp', np.float32, ('y','x'))

# Write data to layers
Y[:] = ds['latitude'].values
X[:] = ds['longitude'].values
slope_nc[:] = linear_slope
inter_nc[:] = linear_inter
temp_mean_nc[:] = np.mean(ds['t2m'], axis=2)

print('Writing data to %s' % path + 'empirical_albedo_model.nc')
    
# Close dataset
dataset.close()




















