#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DESCRIPTION

Draft figures for paper

"""

# Import modules
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import data
mod = netCDF4.Dataset('/home/johnny/Documents/Clouds/Data/Model_Evaluation/Final_Climatologies.nc')

def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')

###############################################################################
# Figure 1a. Map showing mean cloudiness
###############################################################################
lons = mod.variables['longitude'][:]
lats = mod.variables['latitude'][:]
mod_net_sw = np.nanmean(mod.variables['cloudiness'][:], axis=2) * 100

# Resize for more convenient plotting
lons = lons[::5,::5]
lats = lats[::5,::5]
mod_net_sw = mod_net_sw[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(0, 25, 1)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, mod_net_sw, v, transform=ccrs.PlateCarree(), vmin=0, vmax=25,
             cmap='Purples')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[0, 5, 10, 15, 20, 25, 30])
cbar.ax.set_yticklabels([0, 5, 10, 15, 20, 25, 30]) 
cbar.set_label('Cloudiness (%)', rotation=270, labelpad=12)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_1a_Mean_Cldy_SW.png', dpi=200)

###############################################################################
# Figure 1b. Map showing mean albedo
###############################################################################

mod_albedo = np.nanmean(mod.variables['albedo'][:], axis=2)

# Resize for more convenient plotting
mod_albedo = mod_albedo[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(0.3, 0.9, 0.05)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, mod_albedo, v, transform=ccrs.PlateCarree(), vmin=0.3, vmax=0.9,
             cmap='Purples_r')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
cbar.ax.set_yticklabels([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) 
cbar.set_label('Albedo', rotation=270, labelpad=12)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_1b_Mean_Albedo.png', dpi=200)

###############################################################################
# Figure 1c. Map showing mean CRE SW
###############################################################################
mod_cre_sw = np.nanmean(mod.variables['cre_sw'][:], axis=2)

# Resize for more convenient plotting
mod_cre_sw = mod_cre_sw[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(-40, 1, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, mod_cre_sw, v, transform=ccrs.PlateCarree(), vmin=-40, vmax=1,
             cmap='Blues_r')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-40, -30, -20, -10, 0, 10])
cbar.ax.set_yticklabels([-40, -30, -20, -10, 0, 10]) 
cbar.set_label('Mean CRE$_{SW}$ (W m$^{-2}$)', rotation=270, labelpad=12)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_1c_Mean_CRE_SW.png', dpi=200)

###############################################################################
# Figure 1d. Map showing mean CRE LW
###############################################################################
mod_cre_lw = np.nanmean(mod.variables['cre_lw'][:], axis=2)

# Resize for more convenient plotting
mod_cre_lw = mod_cre_lw[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(0, 50, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, mod_cre_lw, v, transform=ccrs.PlateCarree(), vmin=0, vmax=50,
             cmap='Reds')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[0, 10, 20, 30, 40])
cbar.ax.set_yticklabels([0, 10, 20, 30, 40]) 
cbar.set_label('Mean CRE$_{LW}$ (W m$^{-2}$)', rotation=270, labelpad=12)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_1d_Mean_CRE_LW.png', dpi=200)


###############################################################################
# Figure 1e. Map showing mean CRE NET
###############################################################################

mod_cre = np.nanmean(mod.variables['cre'][:], axis=2)

# Resize for more convenient plotting
mod_cre = mod_cre[::5,::5]

fig = plt.figure(figsize=(4, 4))
v = np.arange(-41, 41, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
plt.contourf(lons, lats, mod_cre, v, transform=ccrs.PlateCarree(), vmin=-21, vmax=41,
             cmap='coolwarm')
ax.coastlines(resolution='50m', color='black', linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-30, -20, -10, 0, 10, 20, 30])
cbar.ax.set_yticklabels([-30, -20, -10, 0, 10, 20, 30]) 
cbar.set_label('Mean CRE (W m$^{-2}$)', rotation=270, labelpad=12)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Figures/Fig_1e_Mean_CRE_NET.png', dpi=200)

###############################################################################
# Supp Figure 1i. Zoomed map showing mean CRE NET in SW Greenland
###############################################################################
lons = mod.variables['longitude'][:]
lats = mod.variables['latitude'][:]
mod_cre = np.nanmean(mod.variables['cre'][:], axis=2)

fig = plt.figure(figsize=(4, 4))
v = np.arange(-41, 41, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
ax.set_extent([-53, -47, 65, 69], crs=ccrs.PlateCarree()) # (x0, x1, y0, y1)
plt.contourf(lons, lats, mod_cre, v, transform=ccrs.PlateCarree(), vmin=-21, vmax=41,
             cmap='coolwarm')
ax.coastlines(resolution='10m', color='black', linewidth=0.3)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-30, -20, -10, 0, 10, 20, 30])
cbar.ax.set_yticklabels([-30, -20, -10, 0, 10, 20, 30])
cbar.set_label('Mean CRE (W m$^{-2}$)', rotation=270, labelpad=12)
#scale_bar(ax, length=50, location=(0.5, 0.05), linewidth=3)
#plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_1i_Mean_CRE_SW_Greenland.png', dpi=200)

###############################################################################
# Supp Figure 1j. Zoomed map showing mean CRE NET in SE Greenland
###############################################################################

fig = plt.figure(figsize=(4, 4))
v = np.arange(-41, 41, 2)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
ax.set_extent([-46, -40, 60, 64], crs=ccrs.PlateCarree()) # (x0, x1, y0, y1)
plt.contourf(lons, lats, mod_cre, v, transform=ccrs.PlateCarree(), vmin=-21, vmax=41,
             cmap='coolwarm')
ax.coastlines(resolution='10m', color='black', linewidth=0.3)
ax.add_feature(cfeature.LAND, facecolor='antiquewhite')
ax.outline_patch.set_edgecolor('white')
cbar = plt.colorbar(ticks=[-40, -30, -20, -10, 0, 10, 20, 30, 40])
cbar.ax.set_yticklabels([-40, -30, -20, -10, 0, 10, 20, 30, 40]) 
cbar.set_label('Mean CRE (W m$^{-2}$)', rotation=270, labelpad=12)
#scale_bar(ax, length=50, location=(0.15, 0.05), linewidth=3)
plt.tight_layout()
plt.savefig('/home/johnny/Documents/Clouds/Manuscript/Supp_Figures/Fig_1j_Mean_CRE_SE_Greenland.png', dpi=200)







