#!/usr/bin/env python


import netCDF4 as nc
import numpy as np
import mathex
import gnc

#Readme:
#The `land` variable in cruncep files records the serial No. of the land
#in a row-major approach


#This function is only for wind speed retrieval, will be dropped in the future.
def retrieve_wind(filename):
    grp = nc.Dataset(filename)
    #the land is 1-based indexed
    land = grp.variables['land'][:]
    arr = np.zeros((360,720))
    for ind,num in enumerate(land):
        #This is the last column
        if num%720 == 0:
            j = 719
            i = num/720 - 1
        #normal columns
        else:
            j = num%720 - 1
            i = num/720
        arr[i,j] = 1
    arrmask = np.ma.masked_equal(arr,0)
    landmask = arrmask.mask


    def retrieve_varvalue(varname):
        var = grp.variables[varname][:]
        varnew = np.zeros((var.shape[0],360,720))
        for ind,num in enumerate(land):
            #This is the last column
            if num%720 == 0:
                j = 719
                i = num/720 - 1
            #normal columns
            else:
                j = num%720 - 1
                i = num/720
            varnew[:,i,j] = var[:,ind]
        varnew=mathex.ndarray_mask_smart_apply(varnew,landmask)
        return varnew

    wind_east = retrieve_varvalue('Wind_E')
    wind_north = retrieve_varvalue('Wind_N')
    windspeed = np.ma.sqrt(wind_east**2 + wind_north**2)
    grp.close()
    return windspeed


def get_var_asgrid(grp,varname,npindex=np.s_[:],forcedata=None):
    """
    Retrieve a varname from the grp object (the land-only file) as a grided data.

    Parameters:
    -----------
    grp: grp = nc.Dataset(fname), the "land" must be a variable in grp.
    varname:
    """
    #the land is 1-based indexed
    if 'land' not in grp.variables:
        raise ValueError("land not in the variables.")
    else:
        land = grp.variables['land'][:]
        arr = np.zeros((360,720))
        for ind,num in enumerate(land):
            #This is the last column
            if num%720 == 0:
                j = 719
                i = num/720 - 1
            #normal columns
            else:
                j = num%720 - 1
                i = num/720
            arr[i,j] = 1
        arrmask = np.ma.masked_equal(arr,0)
        landmask = arrmask.mask

    #Retrieve the varname needed.
    var = grp.variables[varname][:][npindex]
    if forcedata is not None:
        var = forcedata
    varnew = np.zeros((var.shape[0],360,720))
    for ind,num in enumerate(land):
        #This is the last column
        if num%720 == 0:
            j = 719
            i = num/720 - 1
        #normal columns
        else:
            j = num%720 - 1
            i = num/720
        varnew[:,i,j] = var[:,ind]
    varnew=mathex.ndarray_mask_smart_apply(varnew,landmask)
    return varnew

def land_to_index(num):
    """
    This function returns the (ilat,ilon) from land sequential number.
    """
    #This is the last column
    if num%720 == 0:
        j = 719
        i = num/720 - 1
    #normal columns
    else:
        j = num%720 - 1
        i = num/720
    return (i,j)

def latlon_to_land_index(vlat,vlon):
    """
    This function returns the land index for the point of (lat,lon) for
    0.5-degree CRUNCEP data.

    Notes:
    ------
    it return the land value of the vlat,vlon, to get the index needed, one
    needs to use np.nonzero(grp.variables['land']==land_value)
    """
    globlat = np.arange(89.75,-90,-0.5)
    globlon = np.arange(-179.75,180,0.5)
    (index_lat,index_lon) = gnc.find_index_by_point(globlat,globlon,(vlat,vlon))
    return index_lat*720+index_lon+1


def landmask_to_land(land_mask):
    """
    Get the field 'land' used in the cruncep forcing by the land_mask. Land
        points should have the zero or False value in the land_mask array.
    """
    nlat,nlon = land_mask.shape
    return np.ravel_multi_index(np.nonzero(~land_mask),(nlat,nlon))+1

