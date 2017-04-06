#!/usr/bin/env python


import matplotlib.pyplot as plt
import matplotlib as mat
import g
import numpy as np
import netCDF4 as nc
import pdb as pdb
import mathex
import copy
import pb
import bmap
from inspect import isfunction
import Pdata
import datetime
import pandas as pa
import os
import sys
import Gdata
from collections import OrderedDict

home_dir = os.path.expanduser('~')
pylab_dir = home_dir+'/'+'python'
basedata_dir = pylab_dir + '/basedata'


def grp_find_unlimited_dim(grp):
    for dimname in grp.dimensions.keys():
        dimvar_full = grp.dimensions[dimname]
        if dimvar_full.isunlimited():
            return dimname

def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f
    return decorator

def find_index_by_region(globHDlon, globHDlat, region_lon, region_lat):
    """
    Find the four indices for the subregion against the whole region that will be formed by merging.

    Parameters:
    -----------
    globHDlon and globHDlat are vectors of lon/lat for the connected file;
    region_lon and region_lat are vectors of lon/lat for the individuel files.

    Warnings:
    ---------
    As this function is designed in case to merge some regional files to a global one (or connected one), in this case the globHDlon/globHDlat could be the lon/lat
        of the whole region that's to be formed by merging, and region_lon/region_lat could be the lon/lat of each subregion that participate in merging. So a strict
        rule has been applied that requires the globHDlon and region_lon (as well as lat) to be in the SAME ascending/descending order.

    Returns:
    --------
    lon_index_min, lon_index_max, lat_index_min, lat_index_max: the four indices indicating the position of the subregion in the bigger grid. Now to have region_lon
        by using lon_index_min and lon_index_max, one should use globHDlon[lon_index_min:lon_index_max+1] due to python's indexing method.
    """
    if not pb.indexable_check_same_order(globHDlon,region_lon):
        raise ValueError("the longtitudes are not in same order!")
    if not pb.indexable_check_same_order(globHDlat,region_lat):
        raise ValueError("the latitudes are not in same order!")

    lon_index = np.nonzero((globHDlon >= region_lon.min()) & (globHDlon <= region_lon.max()))[0]
    lon_index_min = np.min(lon_index)
    lon_index_max = np.max(lon_index)

    lat_index = np.nonzero((globHDlat >= region_lat.min()) & (globHDlat <= region_lat.max()))[0]
    lat_index_min = np.min(lat_index)
    lat_index_max = np.max(lat_index)
    return (lon_index_min, lon_index_max, lat_index_min, lat_index_max)

def find_index_by_vertex(globHDlon, globHDlat, (vlon1,vlon2), (vlat1,vlat2)):
    """
    globHDlon and globHDlat are vectors of lon/lat for the connected file;
    (vlon1,vlon2), (vlat1,vlat2) are (min, max) of lon/lat for the individuel files.
    Return:
        (lon_index_min, lon_index_max, lat_index_min, lat_index_max)

    Notes:
    ------
    1. To retrieve the data from derived indexes, one should use:
        np.s_[lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]
    """
    lon_index = np.nonzero((globHDlon >= vlon1) & (globHDlon <= vlon2))[0]
    lon_index_min = np.min(lon_index)
    lon_index_max = np.max(lon_index)

    lat_index = np.nonzero((globHDlat >= vlat1) & (globHDlat <= vlat2))[0]
    lat_index_min = np.min(lat_index)
    lat_index_max = np.max(lat_index)
    return (lon_index_min, lon_index_max, lat_index_min, lat_index_max)

def find_index_by_point(lat,lon,(vlat,vlon)):
    """
    find the index for the point(vlon,vlat) in the grid constructed by
        lon/lat.

    Returns:
    --------
    (index_lat,index_lon) which could be used directly in slicing data.

    Notes:
    ------
    The border of lat/lon should also be given as the central position rather
        than the abosolute border value, because the step of lat/lon and the
        central value for the border pixel are used to calculate the real(abosolute)
        border value inside the code.
    """
    latstep = lat[0] - lat[1]
    if latstep <= 0:
        raise TypeError("lat input is increasing!")
    else:
        lat_big = lat[0] + latstep/2.
        lat_small = lat[-1] - latstep/2.

    lonstep = lon[1] - lon[0]
    if lonstep <= 0:
        raise TypeError("lon input is decreasing!")
    else:
        lon_big = lon[-1] + lonstep/2.
        lon_small = lon[0] - lonstep/2.

    #vlat/vlon could not fall outside range
    if vlat > lat_big or vlat < lat_small:
        raise ValueError("vlat outside lat range")
    if vlon > lon_big or vlon < lon_small:
        raise ValueError("vlon outside lon range")


    #construct final index for lat
    try:
        index_more=np.where(lat>=vlat)[0][-1]
    except IndexError:
        index_lat = np.where(lat<=vlat)[0][0]
        index_more = None

    try:
        index_less=np.where(lat<=vlat)[0][0]
    except IndexError:
        index_lat = np.where(lat>=vlat)[0][-1]
        index_less = None

    if None not in (index_more,index_less):
        if abs(lat[index_more]-vlat) >= abs(lat[index_less]-vlat):
            index_lat=index_less
        else:
            index_lat=index_more

    #construct final index for lon
    try:
        index_more=np.where(lon>=vlon)[0][0]
    except IndexError:
        index_lon = np.where(lon<=vlon)[0][-1]
        index_more = None

    try:
        index_less=np.where(lon<=vlon)[0][-1]
    except IndexError:
        index_lon = np.where(lon>=vlon)[0][0]
        index_less = None

    if None not in (index_more,index_less):
        if abs(lon[index_more]-vlon) >= abs(lon[index_less]-vlon):
            index_lon=index_less
        else:
            index_lon=index_more

    return (index_lat,index_lon)


def test_find_index_by_vertex():
    d = pb.pfload(basedata_dir+'/landmask_et_latlon.pf')
    (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = find_index_by_vertex(d.globHDlon, d.globHDlat, (-50,20),(-35,25))
    lon = d.globHDlon[lon_index_min: lon_index_max+1]
    assert lon[0]==-49.75 and lon[-1]==19.75
    lat = d.globHDlat[lat_index_min: lat_index_max+1]
    assert lat[0]==24.75 and lat[-1]==-34.75

def test_find_index_by_region():
    d = pb.pfload('basedata/landmask_et_latlon.pf')
    region_lon = np.arange(-49.75,20,0.5)
    region_lat = np.arange(24.75,-35,-0.5)
    (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = find_index_by_region(d.globHDlon, d.globHDlat, region_lon, region_lat)
    reglon_new = d.globHDlon[lon_index_min: lon_index_max+1]
    np.testing.assert_array_equal(region_lon, reglon_new)
    reglat_new = d.globHDlat[lat_index_min: lat_index_max+1]
    np.testing.assert_array_equal(region_lat, reglat_new)

def _construct_slice_by_dim(numdim,(lon_index_min, lon_index_max, lat_index_min, lat_index_max)):
    """
    construct the slicing which we used to slice the global data in order to fill in regional data.
    """
    if numdim == 4:
        subslice=np.s_[:,:,lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]
    elif numdim == 3:
        subslice=np.s_[:,lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]
    elif numdim == 2:
        subslice=np.s_[lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]
    else:
        raise ValueError("Strange that numdim is 1")
    return subslice


def dic2ncHalfDegree(filename,timevar=('year',1),vardic='none',histtxt='none'):
    """
    Definition:
        dic2nc(filename,timevar=('year',1),vardic='none',histtxt='none')
    Note:
        1. Only use for 0.5dX0.5d resolution
        2. the default dimensions are lat, lon, time_counter
        3. timevar=('year',1), the first element of the tuple is the longname of time variable, the second element is the time length.
        4. vardic=[('ba','burned area','ha',ba),('fc','fire counts','1/day',fc)], the first two in the variable tuple are shortname and longname for variable, 
           the third one is unit, and the last one is data. Note the data should be a numpy masked array with dimension of "timelength * 360 * 720".
        5. The function can handle the missing value automatically from the fill_value attribute of the masked array.
    Example:
        >>> a=np.arange(360*720.).reshape(360,720)
        >>> b=np.ma.masked_less(a,100000)
        >>> ff=np.ma.array([b,b])
        >>> ff.shape
          (2, 360, 720)
        >>> dic2nc('test.nc',timevar=('year',2),vardic=[('tv','test variable','no unit',ff)],histtxt='only for test purpose')
    """
    print 'Begin to write to netcdf file',filename
    rootgrp=nc.Dataset(filename,'w',format='NETCDF3_CLASSIC')
    
    #creat dimension
    lat=rootgrp.createDimension('lat',360)
    lon=rootgrp.createDimension('lon',720)
    rootgrp.createDimension('time_counter',None)
    
    #creat dimension variable
    lat=rootgrp.createVariable('lat','f4',('lat',))
    lon=rootgrp.createVariable('lon','f4',('lon',))
    lat.long_name = "latitude"
    lon.long_name = "longitude"
    lon.units="degrees_east"
    lat.units="degrees_north"
    y=np.arange(-179.75,180,0.5)
    x=np.arange(89.75,-90,-0.5)
    lat[:]=x
    lon[:]=y
    
    time=rootgrp.createVariable('time','i4',('time_counter',))
    time[:]=np.arange(1,timevar[1]+1)
    time.long_name = timevar[0]
    
    
    #creat ordinary variable
    if vardic=='none':
        print 'no data is provided!'
    else:
        for vardata in vardic: 
            ba=rootgrp.createVariable(vardata[0],'f4',('time_counter','lat','lon',))
            ba.long_name=vardata[1]
            ba.units=vardata[2]
            if vardata[3].ndim==2:
                ba[0,:,:]=vardata[3]
            else:
                ba[:]=vardata[3]
            if np.ma.isMA(vardata[3]):
                ba.missing_value=vardata[3].fill_value
            else:
                ba.missing_value=float(-9999.)

            print 'Variable  --',vardata[0],'--  is fed into the file'
    rootgrp.history=histtxt
    rootgrp.close()


def dic_ndarray_to_nc_HalfDegree(filename,dic_of_ndarray,unit='unitless',histtxt='No other history provided'):
    """
    write a dictionary of ndarrays of HalfDegree to a netcdf file.

    Parameters:
    ----------
    dic_of_ndarray: a dictionary of ndarray, the dim of ndarray could only be 2-dim or 3-dim.

    Notes:
    ------
    1. The keys will be used as variable names in the nc file.

    """
    if mathex.ndarray_arraylist_equal_shape(dic_of_ndarray.values()):
        array_shape = np.shape(dic_of_ndarray.values()[0])
    else:
        raise ValueError("Please check: not all the ndarray in the dictionary share the same dimension")

    if len(array_shape) <2 or len(array_shape) >3:
        raise ValueError("This function currently only accepts the ndarray with 2 or 3 dimensions, the given dimension is {0}".format(array_shape))
    else:
        if len(array_shape) == 2:
            timevar = ('year',1)
        else:
            timevar = ('year',array_shape[0])
        vardic=[]
        for key,array in dic_of_ndarray.items():
            vardic.append((key,key,unit,array))
    histtxt = 'file created at ' + str(datetime.datetime.today()) + '\n' + histtxt
    dic2ncHalfDegree(filename,timevar=timevar,vardic=vardic,histtxt=histtxt)

def txt2nc_HalfDegree(filename,name_list=None,varname_list=None,name_keyword=False,common_prefix_in_name='',name_surfix='.txt',unit='unitless',histtxt='No other history provided',land_mask=True):
    """
    Write one or a group of txt files directly to a netcdf file.

    Parameters:
    -----------
    name_list: the name_list allows a flexible way to write txt to netcdf files.
        1. when name_list is a string, write only from one txt file, varname_list should also be a string specifying the variable name that's in the new nc file.
        2. when name_list is a list of string:
            2.1 when name_keyword is True, the members in the name_list will be used as nameid together with common_prefix_in_name and name_surfix to construct the complete
                file names, the nameid will be automacatilly used as variable names. In this case varname_list will be overwritten if it's not None
            2.2 when name_keyword is False, the members in name_list should indicate full path of txt files, and strings in varname_list will be used as the variable names
                for nc files.
    name_keyword: see above.
    land_mask: boolean variable. if True the input arrays will be applied with land mask.

    Notes
    -----
    1. the txt files are read by np.genfromtxt, so literally all file types that could be treated with np.genfromtxt could be used as input files.

    See also
    --------
    dic_ndarray_to_nc_HalfDegree
    dic2ncHalfDegree
    """
    land = pb.pfload(basedata_dir+'/landmask_et_latlon.pf')

    #prepare name_list when name_keyword is True
    if name_keyword and isinstance(name_list,list):
        varname_list = name_list[:]
        name_list = [common_prefix_in_name + nameid + name_surfix for nameid in name_list]

    dic_of_ndarray={}
    #we want to write from a list of files
    if isinstance(name_list,list):
        for nameid,varname in zip(name_list,varname_list):
            array = np.genfromtxt(nameid)
            if land_mask:
                array = np.ma.masked_array(array, mask=land.globlandmaskHD)
            dic_of_ndarray[varname] = array
    #we want to write from only one file
    elif isinstance(name_list,str):
        array = np.genfromtxt(name_list)
        if land_mask:
            array = np.ma.masked_array(array, mask=land.globlandmaskHD)
        dic_of_ndarray[varname_list] = array
    else:
        raise ValueError("name_list could only be string or list!")

    dic_ndarray_to_nc_HalfDegree(filename,dic_of_ndarray,unit=unit,histtxt=histtxt)

def dic2nc(filename,latvar=1,lonvar=1,timevar=('year',1),vardic='none',histtxt='none'):
    """
    Definition:
       dic2nc(filename,latvar=1,lonvar=1,timevar=('year',1),vardic='none',histtxt='none') 
    Note:
        1. latvar, lonvar receive lat and lon variables, eg. latvar=np.arange(89.75,-90,-0.5), lonvar=np.arange(-179.75,180,0.5)
        2. the default dimensions are (time_counter, lat, lon)
        3. timevar=('year',1), the first element of the tuple is the longname of time variable, the second element is the time length.
        4. vardic=[('ba','burned area','ha',ba),('fc','fire counts','1/day',fc)], the first two in the variable tuple are shortname and longname for variable, 
           the third one is unit, and the last one is data. Note the data should be a numpy masked array with dimension of "timelength * 360 * 720".
        5. The function can handle the missing value automatically from the fill_value attriabute of the masked array.
    Example:
        >>> a=np.arange(360*720.).reshape(360,720)
        >>> b=np.ma.masked_less(a,100000)
        >>> ff=np.ma.array([b,b])
        >>> ff.shape
          (2, 360, 720)
        >>> dic2nc('test.nc',latvar=np.arange(89.75,-90,-0.5),lonvar=np.arange(-179.75,180,0.5),timevar=('year',2),vardic=[('tv','test variable','no unit',ff)],histtxt='only for test purpose')
    """
    print 'Begin to write to netcdf file',filename
    rootgrp=nc.Dataset(filename,'w',format='NETCDF3_CLASSIC')
    
    #creat dimension
    lat=rootgrp.createDimension('lat',len(latvar))
    lon=rootgrp.createDimension('lon',len(lonvar))
    rootgrp.createDimension('time_counter',None)
    
    #creat dimension variable
    lat=rootgrp.createVariable('lat','f4',('lat',))
    lon=rootgrp.createVariable('lon','f4',('lon',))
    lat.long_name = "latitude"
    lon.long_name = "longitude"
    lon.units="degrees_east"
    lat.units="degrees_north"
    lat[:]=latvar
    lon[:]=lonvar
    
    time=rootgrp.createVariable('time','i4',('time_counter',))
    time[:]=np.arange(1,timevar[1]+1)
    time.long_name = timevar[0]
    
    
    #creat ordinary variable
    if vardic=='none':
        print 'no data is provided!'
    else:
        for vardata in vardic: 
            ba=rootgrp.createVariable(vardata[0],'f4',('time_counter','lat','lon',))
            ba.long_name=vardata[1]
            ba.units=vardata[2]
            if vardata[3].ndim==2:
                ba[0,:,:]=vardata[3]
            else:
                ba[:]=vardata[3]
            if np.ma.isMA(vardata[3]):
                ba.missing_value=vardata[3].fill_value
            else:
                ba.missing_value=1.e+20

            print 'Variable  --',vardata[0],'--  is fed into the file'
    rootgrp.history=histtxt
    rootgrp.close()


class NcWrite(object):
    """
    NcWrite object allows for flexible writing to NetCDF file.
    """

    def __init__(self,filename):
        print 'Begin to write to netcdf file',filename
        self.filename=filename
        self.rootgrp=nc.Dataset(filename,'w',format='NETCDF3_CLASSIC')
        self._diminfo_keys = ['dim_name', 'dimvar_name', 'dimvar_longname', 'dimvar_dtype', 'dimvar_value', 'dimvar_unit', 'unlimited']
        self.default_record_dim_name = 'time_counter'
        self.default_record_dim_var_name = 'time'
        self.dimensions = {}
        self.diminfo_lat = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname'],['lat', 'lat', 'latitude']))
        self.latdim_name = self.diminfo_lat['dim_name']
        self.latvar_name = self.diminfo_lat['dimvar_name']
        self.diminfo_lon = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname'],['lon', 'lon', 'longitude']))
        self.londim_name = self.diminfo_lon['dim_name']
        self.lonvar_name = self.diminfo_lon['dimvar_name']
        self.diminfo_time_year = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname', 'dimvar_dtype', 'dimvar_unit'],['time_counter','time', 'year', 'i4', 'year']))
        self.diminfo_time_month = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname', 'dimvar_dtype', 'dimvar_unit'],['time_counter','time', 'month', 'i4', 'month'])                                      )
        self.diminfo_time_day = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname', 'dimvar_dtype', 'dimvar_unit'],['time_counter','time', 'day', 'i4', 'day'])                                      )
        self.timedim_name = self.diminfo_time_year['dim_name']
        self.timevar_name = self.diminfo_time_year['dimvar_name']
        self.rootgrp.history=''

    @staticmethod
    def _replace_none_by_given(orinput,default):
        if orinput is None:
            return default
        else:
            return orinput

    def add_diminfo_lat(self,dim_name=None,dimvar_name=None,dimvar_longname=None):
        self.diminfo_lat = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname'],[dim_name,dimvar_name,dimvar_longname]))
        self.latdim_name = self.diminfo_lat['dim_name']
        self.latvar_name = self.diminfo_lat['dimvar_name']

    def add_diminfo_lon(self,dim_name=None,dimvar_name=None,dimvar_longname=None):
        self.diminfo_lon = dict(zip(['dim_name', 'dimvar_name', 'dimvar_longname'],[dim_name,dimvar_name,dimvar_longname]))
        self.londim_name = self.diminfo_lon['dim_name']
        self.lonvar_name = self.diminfo_lon['dimvar_name']

    def add_dim(self, diminfo_value = None,**attr_kwargs):
        """
        Write dimensions and dimensional variables to NetCDF file.
        diminfo could be supplied with a list: [dim_name, dimvar_name, dimvar_longname, dimvar_dtype, dimvar_value, dimvar_unit, unlimited], with unlimited as True or False.

        Returns:
        -------
        the dimension that's created.
        """

        assert len(diminfo_value) == len(self._diminfo_keys)
        diminfo = dict(zip(self._diminfo_keys, diminfo_value))

        #creat dimension and dimension variable
        if diminfo['unlimited'] == True:
            self.rootgrp.createDimension(diminfo['dim_name'], None)
        else:
            self.rootgrp.createDimension(diminfo['dim_name'], len(diminfo['dimvar_value']))

        ncdimvar = self.rootgrp.createVariable(diminfo['dimvar_name'], diminfo['dimvar_dtype'], (diminfo['dim_name'],))
        ncdimvar[:] = diminfo['dimvar_value']
        ncdimvar.long_name = diminfo['dimvar_longname']
        ncdimvar.units = diminfo['dimvar_unit']
        #set variable attributes by hand
        for key,value in attr_kwargs.items():
            ncdimvar.setncattr(key, value)
        self.dimensions[diminfo['dim_name']] = self.rootgrp.dimensions[diminfo['dim_name']]

    def _copy_dimvar_from_grp(self,grp,copy_timevar=False,dims_exclude=[]):
        """
        Copy the dimensional variables (only single-dim var) from an opened
            Dataset object expcet the variables with a single unlimited
            dimension (i.e., unlimited variables).

        Parameters:
        -----------
        grp: the opened nc file.
        """
        for varname in grp.variables.keys():
            varfull = grp.variables[varname]
            if len(varfull.dimensions) > 1:
                pass
            else:
                if varfull.dimensions[0] == grp_find_unlimited_dim(grp):
                    if copy_timevar:
                        varinfo = [varname,varfull.dimensions,varfull.dtype,varfull[:]]
                        self.add_var(varinfo_value=varinfo,attr_copy_from=varfull)
                else:
                    if varfull.dimensions[0] not in dims_exclude:
                        varinfo = [varname,varfull.dimensions,varfull.dtype,varfull[:]]
                        self.add_var(varinfo_value=varinfo,attr_copy_from=varfull)

    def copy_dim_from_grp(self,grp,create_unlimited=True,copy_dimvar=True,
                          copy_timevar=False,
                          dims_exclude=[]):
        """
        Add dimensions by copying from the grp object.

        Parameters:
        -----------
        grp: the Dataset object.
        create_unlimited: boolean value, if True the unlimited dimension of grp
            will also be created.
        copy_dimvar: boolean value, if True the corresponding single-dimensional
            varibles (except the unlimited variable) in grp will also be copied.
        dims_exclude: dimensions that are excluded when making copy.
        """
        for dimname in grp.dimensions.keys():
            dimvar_full = grp.dimensions[dimname]
            if not dimvar_full.isunlimited():
                if dimname not in dims_exclude:
                    self.rootgrp.createDimension(dimname,len(dimvar_full))
            else:
                if create_unlimited:
                    self.rootgrp.createDimension(dimname,None)
                    self.timedim_name=dimname
        if copy_dimvar:
            self._copy_dimvar_from_grp(grp,copy_timevar=copy_timevar,
                                       dims_exclude=dims_exclude)

        for dimname in self.rootgrp.dimensions.keys():
            self.dimensions[dimname] = self.rootgrp.dimensions[dimname]


    def add_dim_lat(self, latvar = None, **attr_kwargs):
        """
        A shortcut function to write latitude dimension, cf. add_dim for
            more information. default latvar is global half degree.
        """
        latvar = self._replace_none_by_given(latvar,np.arange(89.75,-90,-0.5))
        lat_diminfo_value = [self.diminfo_lat['dim_name'],
                             self.diminfo_lat['dimvar_name'],
                             self.diminfo_lat['dimvar_longname'],
                             'f4', latvar, 'degrees_north', False]
        self.add_dim(lat_diminfo_value, **attr_kwargs)
        self.latvar = latvar


    def add_dim_lon(self, lonvar = None, **attr_kwargs):
        """
        A shortcut function to write longitude dimension, cf. add_dim
            for more information. default lonvar is global half degree.
        """
        lonvar = self._replace_none_by_given(lonvar,np.arange(-179.75,180,0.5))
        lon_diminfo_value = [self.diminfo_lon['dim_name'],
                             self.diminfo_lon['dimvar_name'],
                             self.diminfo_lon['dimvar_longname'],
                             'f4', lonvar, 'degrees_east', False]
        self.add_dim(lon_diminfo_value, **attr_kwargs)
        self.lonvar = lonvar

    def add_dim_pft(self,pftnum=13,**attr_kwargs):
        """
        A shortcut function to write PFT dimension, cf. add_dim
            for more information.
        """
        pft_diminfo_value = ['PFT', 'PFT', 'Plant functional type',
                              'i4', np.arange(1,pftnum+1), '1', False]
        self.add_dim(pft_diminfo_value, **attr_kwargs)

    def add_dim_time(self, timevar = None, timestep='year', **attr_kwargs):
        """
        A shortcut fuction to write time dimension
            [default dimension name: time_counter;
             default dimension variable name: time],
            cf. add_dim for more information.
        """
        timevar = self._replace_none_by_given(timevar,np.arange(1,2))
        if timestep == 'year':
            diminfo_time = self.diminfo_time_year
        elif timestep == 'month':
            diminfo_time = self.diminfo_time_month
        elif timestep == 'day':
            diminfo_time = self.diminfo_time_day
        else:
            raise ValueError("timestep {0} is not recognized!".format(timestep))

        time_diminfo_value = [diminfo_time['dim_name'],
                              diminfo_time['dimvar_name'],
                              diminfo_time['dimvar_longname'],
                              diminfo_time['dimvar_dtype'],
                              timevar, diminfo_time['dimvar_unit'], True]
        self.add_dim(time_diminfo_value,**attr_kwargs)
        self.timedim_name = diminfo_time['dim_name']
        self.timevar_name = diminfo_time['dimvar_name']
        self.timevar = timevar


    #creat ordinary variable
    def add_var(self, varinfo_value = None, attr_copy_from = None, **attr_kwargs):
        """
        Purpose: Add variables to NetCDF file.
        Arguments:
            1. varinfo_value = [varname, dim_tuple, dtype, varvalue], eg.['ba', ('time_counter', 'lat', 'lon', ), 'f4', ba_data]
            2. use attr_copy_from if the variable attributes are copied from another netCDF4.Variable object.
            3. set variable attributes by using attr_kwargs.
        Notes:
        ------
        1. The function can handle the missing value automatically from the fill_value attriabute of the masked array.
        2. It can handle the case of time axis lenght ==1
        """
        varinfo_keys=['varname', 'dim_tuple', 'dtype', 'varvalue']
        varinfo = dict(zip(varinfo_keys, varinfo_value))
        vardata = varinfo['varvalue']

        #set missing_value,_FillValue
        if np.ma.isMA(vardata):
            missing_value = vardata.fill_value
        else:
            if isinstance(varinfo['dtype'],str):
                missing_value = nc.default_fillvals[varinfo['dtype']]
            elif isinstance(varinfo['dtype'],np.dtype):
                dtype_str = varinfo['dtype'].char+str(varinfo['dtype'].itemsize)
                missing_value = nc.default_fillvals[dtype_str]
            else:
                raise AttributeError("Unknown type in retrieving the default missing value.")

        var = self.rootgrp.createVariable(varinfo['varname'],
                                          varinfo['dtype'],
                                          varinfo['dim_tuple'],
                                          fill_value=missing_value)

        #handle the case when len(time) == 1
        if var.ndim - vardata.ndim == 1 and var.shape[0] == 1:
            var[:] = vardata[np.newaxis,...]
        else:
            if np.ma.isMA(vardata):
                var[:] = vardata.filled()
            else:
                var[:] = vardata

        #set also the missing_value attribute for the variable
        var.setncattr('missing_value',missing_value)

        #copy the variable attributes from another netCDF4.Variable object.
        if isinstance(attr_copy_from, nc.Variable):
            for attr_name in attr_copy_from.ncattrs():
                if attr_name not in [u'_FillValue', u'missing_value']:
                    var.setncattr(attr_name, attr_copy_from.getncattr(attr_name))

        #set variable attributes by hand
        for key,value in attr_kwargs.items():
            var.setncattr(key, value)
        print 'Variable  --',varinfo['varname'],'--  is fed into the file'

    def add_2dim_lat_lon(self, lat_value=None, lon_value=None):
        """
        This is shortcut function for adding 3dim_time_lat_lon; all 3 dimensions will use default dim name.
        """
        self.add_dim_lat(latvar=lat_value)
        self.add_dim_lon(lonvar=lon_value)

    def add_3dim_time_lat_lon(self, time_length=None, lat_value=None, lon_value=None):
        """
        This is shortcut function for adding 3dim_time_lat_lon; all 3 dimensions will use default dim name.
        """
        self.add_dim_lat(latvar=lat_value)
        self.add_dim_lon(lonvar=lon_value)
        time_length = self._replace_none_by_given(time_length,1)
        self.add_dim_time(np.arange(time_length)+1)

    def add_4dim_time_pft_lat_lon(self, time_length=None, lat_value=None, lon_value=None):
        """
        This is shortcut function for adding 3dim_time_lat_lon; all 3 dimensions will use default dim name.
        """
        self.add_dim_lat(latvar=lat_value)
        self.add_dim_lon(lonvar=lon_value)
        time_length = self._replace_none_by_given(time_length,1)
        self.add_dim_time(np.arange(time_length)+1)
        self.add_dim_pft()

    def add_var_3dim_time_lat_lon(self, varname, data, attr_copy_from=None,
                                  **attr_kwargs):
        self.add_var((varname,
                     (self.timedim_name, self.latdim_name,self.londim_name, ),
                     'f4', data), attr_copy_from=attr_copy_from, **attr_kwargs)

    def add_var_2dim_lat_lon(self, varname, data, attr_copy_from=None, **attr_kwargs):
        self.add_var((varname, (self.latdim_name, self.londim_name, ), 'f4', data), attr_copy_from=attr_copy_from, **attr_kwargs)

    def add_var_4dim_time_pft_lat_lon(self, varname, data, attr_copy_from=None, **attr_kwargs):
        self.add_var((varname, (self.timedim_name, 'PFT', self.latdim_name, self.londim_name, ), 'f4', data), attr_copy_from=attr_copy_from, **attr_kwargs)

    def add_vars_from_Ncdata(self,varlist,Ncdata):
        """
        Add vars from existing Ncdata. The number of dimension
        of the variable in the Ncdata.d1 is used to guess which
        function has be used to put the data into the newly
        created netcdf file:
            - add_var_2dim_lat_lon, for ndim = 2
            - add_var_3dim_time_lat_lon, for ndim = 3
            - add_var_4dim_time_pft_lat_lon, for ndim = 4
        """
        for varname in varlist:
            Ncdata.retrieve_variables([varname])
            ndim = Ncdata.d1.__dict__[varname].ndim
            if ndim == 2:
                self.add_var_2dim_lat_lon(varname,Ncdata.d1.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            elif ndim == 3:
                self.add_var_3dim_time_lat_lon(varname,Ncdata.d1.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            elif ndim == 4:
                self.add_var_4dim_time_pft_lat_lon(varname,Ncdata.d1.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            else:
                ValueError("Variable -{0}- has {1} dimension!".format(varname,ndim))
            Ncdata.remove_variables([varname])

    def add_vars_from_Ncdata_pftsum(self,varlist,Ncdata):
        """
        Add vars from existing Ncdata. The number of dimension
        of the variable in the Ncdata.d1 is used to guess which
        function has be used to put the data into the newly
        created netcdf file:
            - add_var_2dim_lat_lon, for ndim = 2
            - add_var_3dim_time_lat_lon, for ndim = 3
            - add_var_4dim_time_pft_lat_lon, for ndim = 4
        """
        for varname in varlist:
            Ncdata.retrieve_variables([varname])
            Ncdata.get_pftsum(varlist=[varname])
            ndim = Ncdata.pftsum.__dict__[varname].ndim
            if ndim == 2:
                self.add_var_2dim_lat_lon(varname,Ncdata.pftsum.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            elif ndim == 3:
                self.add_var_3dim_time_lat_lon(varname,Ncdata.pftsum.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            elif ndim == 4:
                self.add_var_4dim_time_pft_lat_lon(varname,Ncdata.pftsum.__dict__[varname],
                    attr_copy_from=Ncdata.d0.__dict__[varname])
            else:
                ValueError("Variable -{0}- has {1} dimension!".format(varname,ndim))
            Ncdata.remove_variables([varname],pftsum=True)

    def add_history_attr(self,histtxt='No history provided'):
        self.rootgrp.history=histtxt

    def add_global_attributes(self,attr_dic):
        self.rootgrp.setncatts(attr_dic)

    def close(self):
        self.rootgrp.history = self.rootgrp.history + \
                    '\nfile created at ' + str(datetime.datetime.today())
        self.rootgrp.close()

    def _construct_data_by_dim(self,numdim):
        """
        construct the global ndarray that we need and fill in with
            the data from each region.

        Notes:
        ------
        1. The data is iniated as all being masked.
        """
        dimlen_dic = pb.Dic_Apply_Func(len,self.dimensions)
        if numdim == 4:
            glob_data = np.ma.zeros((dimlen_dic[self.timedim_name],dimlen_dic['PFT'],dimlen_dic[self.latdim_name],dimlen_dic[self.londim_name]))
            glob_data.mask = True
        elif numdim == 3:
            glob_data = np.ma.zeros((dimlen_dic[self.timedim_name],dimlen_dic[self.latdim_name],dimlen_dic[self.londim_name]))
            glob_data.mask = True
        elif numdim == 2:
            glob_data = np.ma.zeros((dimlen_dic[self.latdim_name],dimlen_dic[self.londim_name]))
            glob_data.mask = True
        else:
            raise ValueError("Strange that numdim is 1")
        return glob_data


    def add_var_smart_ndim(self,varname,numdim,data,pftdim=False,**attr_kwargs):
        '''
        Select the add_var_* method in a smart way by knowing the ndim.

        Parameters:
        -----------
        ndim: the number of dimensions for the varname
        '''
        if numdim == 4:
            if data.ndim == 4:
                self.add_var_4dim_time_pft_lat_lon(varname, data, **attr_kwargs)
            elif data.ndim == 3:
                if pftdim == False:
                    self.add_var_3dim_time_lat_lon(varname, data, **attr_kwargs)
                else:
                    self.add_var_4dim_time_pft_lat_lon(varname, data, **attr_kwargs)
            elif data.ndim == 2:
                self.add_var_3dim_time_lat_lon(varname, data, **attr_kwargs)
            else:
                raise ValueError("data ndim < 2!")
        elif numdim == 3:
            self.add_var_3dim_time_lat_lon(varname, data, **attr_kwargs)
        elif numdim == 2:
            self.add_var_2dim_lat_lon(varname, data, **attr_kwargs)
        else:
            raise ValueError("Strange that numdim is 1")



    def _add_var_by_dim(self,varname,numdim,data,pyfunc=None):
        """
        Used only by add_var_from_file_list
        """
        glob_data = self._construct_data_by_dim(numdim)
        for subdata in data:
            lon_index_min, lon_index_max, lat_index_min, lat_index_max = find_index_by_region(self.lonvar, self.latvar, subdata.lon, subdata.lat)
            subslice = _construct_slice_by_dim(numdim,(lon_index_min, lon_index_max, lat_index_min, lat_index_max))
            glob_data[subslice] = subdata.d1.__dict__[varname] #note here the mask of glob_data will be changed automatically.
            print "data fed from file --{0}--".format(subdata.filename)

        if pyfunc is not None:
            if callable(pyfunc):
                glob_data = pyfunc(glob_data)
            else:
                raise TypeError("pyfunc not callable")


        if numdim == 4:
            self.add_var_4dim_time_pft_lat_lon(varname, glob_data, attr_copy_from=subdata.d0.__dict__[varname])
        elif numdim == 3:
            self.add_var_3dim_time_lat_lon(varname, glob_data, attr_copy_from=subdata.d0.__dict__[varname])
        elif numdim == 2:
            self.add_var_2dim_lat_lon(varname, glob_data, attr_copy_from=subdata.d0.__dict__[varname])
        else:
            raise ValueError("Strange that numdim is 1")

    def construct_data_from_file_list(self,dimlist,lonvarname,latvarname,
                                 input_file_list,varname,
                                 Ncdata_latlon_dim_name=None):
        """
        The original add_var_from_file_list could only handle
        the cases of having PFT as dim when there are 4 dims,
        this function allows to retrieve arbitrary dimension
        data and combine them spatially.

        Parameters:
        -----------
        input_file_list: Could be a filename list or Ncdata list.
        """
        #data = [Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name) for filename in input_file_list]
        data = []
        for filename in input_file_list:
            if isinstance(filename,Ncdata):
                gnc_dt = filename
            else:
                gnc_dt = Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name,largefile=True)
            gnc_dt.retrieve_variables([varname])
            data.append(gnc_dt)

        if not pb.object_list_check_unique_attribute([subdata.d0.__dict__[varname] for subdata in data],'dimensions'):
            raise ValueError("The variable {0} in all input files does not have the same dimension!".format(varname))
        else:
            dimtuple = tuple([len(self.dimensions[dimname]) for dimname in dimlist])
            glob_data = np.ma.zeros(dimtuple)
            glob_data.mask = True

            for subdata in data:
                lon_index_min, lon_index_max, lat_index_min, lat_index_max = \
                    find_index_by_region(self.rootgrp.variables[lonvarname][:], self.rootgrp.variables[latvarname][:], subdata.lon, subdata.lat)
                subslice = _construct_slice_by_dim(len(dimlist),(lon_index_min, lon_index_max, lat_index_min, lat_index_max))
                glob_data[subslice] = subdata.d1.__dict__[varname] #note here the mask of glob_data will be changed automatically.
                print "data fed from file --{0}--".format(subdata.filename)

            return glob_data
        for subdata in data:
            subdata.close()

    def add_var_from_file_list(self,input_file_list,varlist,
                               Ncdata_latlon_dim_name=None,
                               pyfunc=None):
        """
        Mainly used for merging nc files spatially

        Parameters:
        -----------
        input_file_list: the NetCDF input file list for merging.
        varlist: variable name list that will appear in merged nc file.
            Note each input file have have the specified variable
            and the number of dimension accros all input files
            must be the same.
        Ncdata_latlon_dim_name: the selective variable that's used
            in Ncdata when open a nc file.

        Notes:
        ------
        1. If there are intersections in the input file spatial coverage,
            files come later will overwrite the proceeding ones.
        """
        #data = [Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name) for filename in input_file_list]
        data = []
        for filename in input_file_list:
            gnc_dt = Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name,largefile=True)
            gnc_dt.retrieve_variables(varlist)
            data.append(gnc_dt)

        subdata_first = data[0]
        for varname in varlist:
            if not pb.object_list_check_unique_attribute([subdata.d0.__dict__[varname] for subdata in data],'dimensions'):
                raise ValueError("The variable {0} in all input files does not have the same dimension!".format(varname))
            else:
                numdim = len(subdata_first.d0.__dict__[varname].dimensions)
                self._add_var_by_dim(varname,numdim,data,pyfunc=pyfunc)

        glob_attr_dic = subdata_first.global_attributes
        #pdb.set_trace()
        #add time stamp for this operation
        if 'history' in glob_attr_dic:
            glob_attr_dic['history'] = 'file created at ' + str(datetime.datetime.today()) + '\n' + \
                                       'by merging files:' + '\n'+ \
                                       '--'+('--'+'\n'+'--').join(input_file_list)+'--'+'\n' + \
                                        str(glob_attr_dic['history'])
        else:
            glob_attr_dic['history'] = 'file created at ' + str(datetime.datetime.today()) + '\n' + \
                                       'by merging files:' + '\n'+ \
                                       '--'+('--'+'\n'+'--').join(input_file_list)+'--'

        #write the global attributes
        if nc.__version__ == '1.0.1':
            self.add_global_attributes(glob_attr_dic)
        elif nc.__version__ == '0.9.7':
            for key,value in glob_attr_dic.items():
                self.rootgrp.setncattr(key,value)

        #close data
        for gnc_dt in data:
            gnc_dt.close()

def _set_default_ncfile_for_write(ncfile,**kwargs):
    '''
    set default lat,lon,time,pft... ect for writing to nc file.

    kwargs:
    -------
    timestep: 'year' or 'month', default is 'year'
    time_length: np.arange(1,time_length+1) will be the time variable value,
        default value is 1.
    latvar,lonvar: the lat/lon for megered data. default is 0.5 degree
        resolution with global coverage.
    latinfo,loninfo: tuple containing ('lat/lon_dim_name','lat/lon_var_name',
        'lat/lon_var_longname'); default for lat is ('lat','lat','latitude')
        and for lon is ('lon','lon','longitude').
    pft: if pft==True, then PFT dimensions from ORCHIDEE will be added.
    '''

    loninfo = kwargs.get('loninfo',('lon','lon','longitude'))
    latinfo = kwargs.get('latinfo',('lat','lat','latitude'))
    latvar = kwargs.get('latvar',np.arange(89.75,-90,-0.5))
    lonvar = kwargs.get('lonvar',np.arange(-179.75,180,0.5))
    timestep = kwargs.get('timestep','year')
    time_length = kwargs.get('time_length',1)

    ncfile.add_diminfo_lon(*loninfo)
    ncfile.add_diminfo_lat(*latinfo)
    ncfile.add_2dim_lat_lon(latvar,lonvar)
    ncfile.add_dim_time(np.arange(1,1+time_length),timestep=timestep)
    if kwargs.get('pft',False):
        ncfile.add_dim_pft()

@append_doc_of(_set_default_ncfile_for_write)
def nc_spatial_concat_ncfiles(outfile,input_file_list,
                              varlist=None,
                              pyfunc=None,
                              Ncdata_latlon_dim_name=None,
                              **kwargs):
    """
    A shortcut for merging spatially a list of files. For detailed control
        of dimensions and variables, use NcWrite first, followed by
        add_var_from_file_list.

    Parameters:
    -----------
    varlist: the variable list that's to be retained in merged file.
        default inclules all variables except the dimension variable.
    Ncdata_latlon_dim_name: the specified lat/lon dimension names that are
        used when calling Ncdata.
    pyfunc: functions to be applied.

    see also
    --------
    gnc.Ncdata

    Test
    ----
    nc_subgrid_csv and nc_merge_files are tested against each other
        in the gnc_test.py.
    """
    subdata_first = Ncdata(input_file_list[0],
                           latlon_dim_name=Ncdata_latlon_dim_name,
                           largefile=True)
    if 'PFT' in subdata_first.dimensions or 'veget' in subdata_first.dimensions:
        kwargs['pft'] = True
    kwargs['time_length'] = subdata_first.unlimited_dimlen
    ncfile = NcWrite(outfile)
    _set_default_ncfile_for_write(ncfile,**kwargs)
    if varlist is None:
        varlist = pb.StringListAnotB(subdata_first.list_var(),
                                     subdata_first.dimvar_name_list)
    ncfile.add_var_from_file_list(input_file_list,varlist,
                         Ncdata_latlon_dim_name=Ncdata_latlon_dim_name,
                         pyfunc=pyfunc)
    ncfile.close()




def ncm2y(infile,outfile,mode=None,varlist='all'):
    """
    Purpose: Transform monthly data to yearly mean or sum. Note the input file format is very strict.
    Note:
        1. the input netCDF file must have only 3 (2or4 will fail) dimensions and only 3 dimension variables with the corresponding axis as the only dimension.
        2. the input netCDF file other variables must have only 3 dimesions with the first dimension as the unlimited time dimension.
        3. mode = 'sum' or 'mean', use mathex.m2ymean or mathex.m2ysum to do the transformation.
        4. the function copy the dimension name, variable name and data for the other 2 dimensions other than the time dimension direclty from the input file to output file.
        5. by default, if varlist is 'all', all the variables in input file will be calculated.

        6.This function has been tested against NCO ncra and is working correctly.
    Example:
    >>> import gnc
    >>> gnc.ncm2y('testdata/cru1999prm.nc','testdata/cru1999prm.yearsum.nc',mode='sum')
    """
    f1=nc.Dataset(infile,'r')
    if len(f1.dimensions) >3:
        print 'the input file has more than 3 dimensions!'

    fdimdic=f1.dimensions
    fdimname=[]
    for i in fdimdic.keys():
        fdimname.append(str(i))
    for i in fdimname:
        tempdimvar=f1.dimensions[i]
        if tempdimvar.isunlimited():
            globe_unlimdimname=i
    fdimname.remove(globe_unlimdimname)
    globe_limdimname1=fdimname[0]
    globe_limdimname2=fdimname[1]

    for name in f1.variables.keys():
        tempvar_full=f1.variables[str(name)]
        if len(tempvar_full.dimensions)==1:
            if str(tempvar_full.dimensions[0])==globe_unlimdimname:
                unlimdimvar_name=str(name)
            elif str(tempvar_full.dimensions[0])==globe_limdimname1:
                limdimvar1_name=str(name)
            elif str(tempvar_full.dimensions[0])==globe_limdimname2:
                limdimvar2_name=str(name)

    limdimvar1_full=f1.variables[limdimvar1_name]
    limdimvar2_full=f1.variables[limdimvar2_name]
    unlimdimvar_full=f1.variables[unlimdimvar_name]
    
    print 'Begin to write to netcdf file ',outfile
    rootgrp=nc.Dataset(outfile,'w',format='NETCDF3_CLASSIC')
    
    #creat dimension
    lat=rootgrp.createDimension(globe_limdimname1,len(limdimvar1_full))
    lon=rootgrp.createDimension(globe_limdimname2,len(limdimvar2_full))
    rootgrp.createDimension(globe_unlimdimname,None)
    print 'Dimensions ',globe_limdimname1,globe_limdimname2,globe_unlimdimname,' created'
    
    #creat dimension variable
    lat=rootgrp.createVariable(str(limdimvar1_full._name),limdimvar1_full.dtype,(globe_limdimname1,))
    lon=rootgrp.createVariable(str(limdimvar2_full._name),limdimvar2_full.dtype,(globe_limdimname2,))
    if hasattr(limdimvar1_full,'long_name') and hasattr(limdimvar2_full,'long_name'):
        lat.long_name = limdimvar1_full.long_name.encode()
        lon.long_name = limdimvar2_full.long_name.encode()
    lat.units=limdimvar1_full.units.encode()
    lon.units=limdimvar2_full.units.encode()
    lat[:]=limdimvar1_full[:].copy()
    lon[:]=limdimvar2_full[:].copy()
    time=rootgrp.createVariable(unlimdimvar_full._name,unlimdimvar_full.dtype,(globe_unlimdimname,))
    time[:]=np.arange(1,len(unlimdimvar_full[:])/12+1)
    time.long_name = 'time'
    print 'Dimension variables ','--'+str(limdimvar1_full._name)+'--','--'+str(limdimvar2_full._name)+'--','--time-- created'

    varlist_all=[name.encode() for name in f1.variables.keys()]
    varlist_all.remove(str(limdimvar1_full._name))
    varlist_all.remove(str(limdimvar2_full._name))
    varlist_all.remove(str(unlimdimvar_full._name))
    if varlist=='all':
        varlist2=varlist_all
    else:
        varlist2=varlist

    #creat ordinary variable
    for varname in varlist2:
        var_full=f1.variables[varname]
        if str(var_full.dimensions[0])!=globe_unlimdimname:
            print 'the time dimension is not the first dimension for variable --',varname,'--!'
        var_value=copy.deepcopy(var_full[:])
        if mode is None:
            raise ValueError('please specify one mode!')
        elif mode=='mean':
            vardata=mathex.m2ymean(var_value)
        elif mode=='sum':
            vardata=mathex.m2ysum(var_value)
        else:
            raise ValueError('only mean or sum can be provided!')
        ba=rootgrp.createVariable(str(var_full._name),var_full.dtype,(str(var_full.dimensions[0]),str(var_full.dimensions[1]),str(var_full.dimensions[2]),))
        if hasattr(var_full,'long_name'):
            ba.long_name=var_full.long_name
        try:
            ba.units=var_full.units
        except AttributeError:
            pass
        if vardata.ndim==2:
            ba[0,:,:]=vardata
        else:
            ba[:]=vardata
        if np.ma.isMA(vardata):
            ba.missing_value=vardata.fill_value
        else:
            ba.missing_value=float(-9999)
        print 'Variable  --',str(var_full._name),'--  is fed into the file ','--',outfile,'--'

    if hasattr(f1,'history'):
        rootgrp.history='year value by applying method month-to-year '+mode+' from file '+'--'+infile+'--\n'+f1.history.encode()
    else:
        rootgrp.history='year value by applying method month-to-year '+mode+' from file '+'--'+infile+'--'
    f1.close()
    rootgrp.close()



def ncmask(infile,outfile,llt=('lat_name','lon_name','rec_name'),mask=None):
    """
    This is only an easy application of pb.ncread and gnc.dic2nc function, can only handle variables with 3 dimension.
    Note:
         1. The variable name for latitude,longitude and time must be explicitly stated.
         2. latitude and longitude must be 1D array but NOT meshgrid
    """
    if llt==('lat_name','lon_name','rec_name'):
        raise ValueError('lat,lon,record variable name must be explicitly stated.')
    else:
        d0,d1=pb.ncreadg(infile)
        varlist=d1.__dict__.keys()
        for dimvar in llt:
            varlist.remove(dimvar)
        #creat output varlist
        outvarlist=[]
        for var in varlist:
            data=mathex.ndarray_mask_smart_apply(d1.__dict__[var],mask)
            varfull=d0.__dict__[var]
            varfull_name=varfull._name
            if hasattr(varfull,'long_name'):
                varfull_longname=varfull.long_name
            else:
                varfull_longname=varfull._name
            if hasattr(varfull,'units'):
                varfull_units=varfull.units
            else:
                varfull_units='no units'
            vartuple=(varfull_name,varfull_longname,varfull_units,data)
            outvarlist.append(vartuple)
        #get length of record variable
        try:
            reclen=len(d1.__dict__[llt[2]])
        except TypeError:
            reclen=1
        dic2nc(outfile,latvar=d1.__dict__[llt[0]],lonvar=d1.__dict__[llt[1]],timevar=(d0.__dict__[llt[2]]._name,reclen),\
        vardic=outvarlist,histtxt='created from file '+infile)

def ncfilemap(infile,latvarname=None,lonvarname=None,mapvarname=None,mapdim=None,agremode=None,pyfunc=None,mask=None,unit=None,title=None,\
              projection='cyl',mapbound='all',gridstep=(30,30),shift=False,cmap=None,map_threshold=None,colorbarlabel=None,levels=None,\
              data_transform=False,ax=None,\
              colorbardic=None):
    """
    OneLineInfo: This is a simple wrap of ncdatamap
    Purpose: plot map directly from a nc file.
    Arguments: cf. ncdatamap arguments
    Returns: 
        if ax is None:
            return d0,d1,fig,axt,m,cbar,mapvar
        else:
            return d0,d1,m,cbar,mapvar
        ---
        d0,d1 --> d0,d1=pb.ncreadg(infile)
        mapvar --> final data used for plotting the map.
    See also bmap.contourfmap2
    Example:
        see testdata/gnc_ncmap_eg.py for examples.
    """
    d0,d1=pb.ncreadg(infile)
    if ax is None:
        fig,axt,m,cbar,mapvar=ncdatamap(d0,d1,latvarname=latvarname,lonvarname=lonvarname,mapvarname=mapvarname,mapdim=mapdim,agremode=agremode,pyfunc=pyfunc,\
                                        mask=mask,unit=unit,title=title,\
                                        projection=projection,mapbound=mapbound,gridstep=gridstep,shift=shift,cmap=cmap,map_threshold=map_threshold,\
                                        colorbarlabel=colorbarlabel,levels=levels,data_transform=data_transform,ax=ax,colorbardic=colorbardic)
        return d0,d1,fig,axt,m,cbar,mapvar
    else:
        m,cbar,mapvar=ncdatamap(d0,d1,latvarname=latvarname,lonvarname=lonvarname,mapvarname=mapvarname,mapdim=mapdim,agremode=agremode,pyfunc=pyfunc,\
                                mask=mask,unit=unit,projection=projection,mapbound=mapbound,gridstep=gridstep,shift=shift,cmap=cmap,map_threshold=map_threshold,\
                                colorbarlabel=colorbarlabel,levels=levels,data_transform=data_transform,ax=ax,colorbardic=colorbardic)
        return d0,d1,m,cbar,mapvar


def ncdatamap(d0,d1,latvarname=None,lonvarname=None,mapvarname=None,forcedata=None,mapdim=None,agremode=None,pyfunc=None,mask=None,mask_value=None,unit=None,title=None,\
             projection='cyl',mapbound='all',gridstep=(30,30),shift=False,cmap=None,map_threshold=None,colorbarlabel=None,levels=None,data_transform=False,ax=None,\
             colorbardic={}):
    """
    Purpose: plot map directly from d0,d1=pb.ncreadg('file.nc') object.
    Arguments:
        d0,d1 --> generated by d0,d1=pb.ncreadg(infile)
        latvarname & lonvarname --> Dimension variables in the file; by default the function tries to use ('lat','lon'),or ('latitude','longtitude') pair.
        mapvarname --> variable name for mapping. variable dimension should be strictly as (timedim,lat,lon); 4dim variable are not handled currently.
        forcedata --> 
            1. when forcedata is ndarray, mapvarname will not be used to retrieve data from d1 but only for displaying variable name purpose. Instead,
               forcedata will be used as data for plotting. This can be used in case data is very big to increase speed. 
            2. In this case, mapvarname will still be used to retrieve variable units from d0.
        unit --> None for use default units inside file; 'unit' to overwrite default; False for no units display.
        title --> None for use default var lont_name inside file; 'I am title' to overwrite default; False for no title display.
        mapdim --> when variable has 3 dim,eg. set mapdim=2 to plot only for the 3rd (mapdim 0-based) data in the time dimension.
        agremode --> 'sum' or 'mean', m2y transformation apply to data prior to mapping. None for ploting original (monthly) data. When agremode is not None, 
            mapdim means the selected year. 'fsum' or 'fmean' apply to the first axis of data.
        pyfunc --> 
            1. a function that will be applied on data. if it's only a number, then data will be multiplied by given number; otherwise one can use pyfunc = 
               lambda x : x*2+3 to define a function. 
            2. pyfunc is used after applying the agremode operation.
            3. Note when using pyfunc, map_threshold will be applied to data after pyfunc transformation as aim of map_threshold is mainly for mapping prupose.
        mask --> a 2dim boolean array
        levels --> levels are used in a way after pyfunc application.
        ax --> axex for which to plot.
        other args --> the same as in function bmap.contourfmap2.
    Returns: 
        if ax is None:
            return fig,ax,m,cbar,mapvar
        else:
            return m,cbar,mapvar
        mapvar --> final data used for plotting the map.
    See also bmap.contourfmap2
    Example:
        see testdata/gnc_ncmap_eg.py for examples. (examples last tested 24/May/2012)
    """
    if latvarname is None and lonvarname is None:
        if 'lat' in d0.__dict__.keys() and 'lon' in d0.__dict__.keys():
            latvar=d0.__dict__['lat'][:]
            lonvar=d0.__dict__['lon'][:]
        elif 'latitude' in d0.__dict__.keys() and 'longitude' in d0.__dict__.keys():
            latvar=d0.__dict__['latitude']
            lonvar=d0.__dict__['longitude']
        else:
            raise ValueError('Default lat and lon names are not found in the data, please speicify them and add into default list')
    else:
        latvar=d0.__dict__[latvarname][:]
        lonvar=d0.__dict__[lonvarname][:]
    #extract 2dim lat,lon to 1dim.
    if latvar.ndim==2:
        latvar=latvar[:,0]
    if lonvar.ndim==2:
        lonvar=lonvar[0,:]

    #read mapvar data and prepare for mapping
    if mapvarname is None:
        if forcedata is None:
            raise ValueError("mapvarname and forcedata both as None!")
        else:
            mapvar = forcedata
        #raise ValueError('please provide a variable name or data')
    else:
        if isinstance(mapvarname,str):
            if forcedata is None:
                mapvar=d1.__dict__[mapvarname]
            else:
                mapvar=forcedata
        else:
            raise ValueError('mapvarname must be string type')
    if mapvar.ndim==2 and (mapdim is not None or agremode is not None):
        raise ValueError('{0} has only 2 valid dimension but mapdim or agremode is not None'.format(mapvarname))
    elif mapvar.ndim==3:
        if agremode is None and mapdim is None:
            raise ValueError('{0} has 3 valid dimension, cannot leave both mapdim and agremode as None'.format(mapvarname))
        elif agremode is not None:
            #make sum or mean of monthly data
            if agremode=='sum':
                mapvar=mathex.m2ysum(mapvar)
            elif agremode=='mean':
                mapvar=mathex.m2ymean(mapvar)
            elif agremode=='fsum':
                mapvar=np.ma.sum(mapvar,axis=0)
            elif agremode=='fmean':
                mapvar=np.ma.mean(mapvar,axis=0)

            #extract data for only specified dimension
            if mapdim is not None:
                if mapvar.ndim==2: #the original mapvar has 3 dim with first dim of 12.
                    raise ValueError ('{0} has 3 valid dimension with firest dim size as 12, cannot specify agremode and mapdim simultaneously'.format(mapvarname))
                else: #mapvar.ndim==3
                    mapvar=mapvar[mapdim,:,:]
            #error handling when mapdim is None
            else:
                if mapvar.ndim==2: #the original mapvar has 3 dim with first dim of 12.
                    pass
                else: #mapvar.ndim==3
                    raise ValueError('the dimension after data transformation is {0}, must specify mapdim to plot'.format(mapvar.shape))
        #plot specified dimension for data without transformation
        else:
            mapvar=mapvar[mapdim,:,:]
    elif mapvar.ndim==4:
        raise ValueError('cannot handle 4 dimensional data')
    #apply the defined function on data
    if pyfunc is not None:
        if isfunction(pyfunc):
            mapvar=pyfunc(mapvar)
        elif isinstance(pyfunc,list) and isfunction(pyfunc[0]):
            for subpyfunc in pyfunc:
                mapvar=subpyfunc(mapvar)
        else:
            mapvar=mapvar*pyfunc

    #apply mask
    if mask is not None:
        mapvar=mathex.ndarray_apply_mask(mapvar,mask)

    if mask_value is not None:
        mapvar=np.ma.masked_equal(mapvar,mask_value)

    #finally, if lat is provide in increasing sequence, flip over the data.
    if latvar[0]<latvar[-1]:
        latvar=latvar[::-1]
        mapvar=np.flipud(mapvar)
    #make the mapping
    if ax is None:
        fig,axt,m,cbar=bmap.contourfmap2(latvar,lonvar,mapvar,projection=projection,mapbound=mapbound,gridstep=gridstep,shift=shift,cmap=cmap,\
                                        map_threshold=map_threshold,\
                                        colorbarlabel=colorbarlabel,levels=levels,data_transform=data_transform,ax=None,colorbardic=colorbardic)
    else:
        axt=ax
        m,cbar=bmap.contourfmap2(latvar,lonvar,mapvar,projection=projection,mapbound=mapbound,gridstep=gridstep,shift=shift,cmap=cmap,\
                                        map_threshold=map_threshold,\
                                        colorbarlabel=colorbarlabel,levels=levels,data_transform=data_transform,ax=ax,colorbardic=colorbardic)

    mapvar_full=d0.__dict__[mapvarname]
    #fuction to retreive title or unit from either inside data or
    #by external forcing
    def retrieve_external_default(external_var,attribute):
        if external_var is not None:
            outvar=external_var
        elif hasattr(mapvar_full,attribute):
            if external_var==False:
                outvar=None
            else:
                outvar=mapvar_full.getncattr(attribute)
        else:
            outvar=None
        return outvar
    #retrieve title or unit
    map_unit=retrieve_external_default(unit,'units')
    map_title=retrieve_external_default(title,'long_name')
    try:
        agre_title_complement='[yearly '+agremode+']'
    except:
        agre_title_complement=None

    #function handling ax title
    def set_title_unit(ax,title=None,unit=None,agre_title_complement=None):
        try:
            title_unit=title+('\n'+unit)
        except TypeError:
            try:
                title_unit=title
                title_agre=agre_title_complement+' '+title
            except TypeError:
                pass
        finally:
            try:
                title_full=agre_title_complement+' '+title_unit
            except TypeError:
                title_full=title_unit
        if title_full is not None:
            ax.set_title(title_full)
        else:
            pass
    set_title_unit(axt,map_title,map_unit,agre_title_complement)
    if ax is None:
        return fig,axt,m,cbar,mapvar
    else:
        return m,cbar,mapvar

def ncreadg(filename):
    """
    Purpose: read a .nc file using netCDF4 package and read the data into netCDF4.Variable object
    Definition: ncread(filename)
    Arguments:
        file--> file name
    Return: return a list of ncdata object; the 1st one contains original nctCDF4 objects and the 2nd one contains data with duplicate dimensions removed.
    Note:
        1. This is for general purpose

    Example:
        >>> data=g.ncread('cru1901.nc')
        >>> data
          <g.ncdata object at 0x2b1e4d0>
        >>> data.t2m
          <netCDF4.Variable object at 0x2b20c50>

    """
    f1=nc.Dataset(filename,mode="r")
    datanew=ncdata()
    datanew2=ncdata()
    for var in f1.variables.keys():
        var=str(var)
        datanew.__dict__[var]=f1.variables[var]
        datanew2.__dict__[var]=Remove_dupdim(f1.variables[var][:])
    return [datanew,datanew2]
    f1.close()


def ncfile_dim_check(infile,outfile,varlist='all',reginterval=None):
    """
    """
    f1=nc.Dataset(infile,'r')
    if len(f1.dimensions) >3:
        raise ValueError('the input file has more than 3 dimensions!')

    #identity dimension variable and unlimited variable
    fdimdic=f1.dimensions
    fdimname=[]
    for i in fdimdic.keys():
        fdimname.append(str(i))
    for i in fdimname:
        tempdimvar=f1.dimensions[i]
        if tempdimvar.isunlimited():
            globe_unlimdimname=i
    fdimname.remove(globe_unlimdimname)
    globe_limdimname1=fdimname[0]
    globe_limdimname2=fdimname[1]

    for name in f1.variables.keys():
        tempvar_full=f1.variables[str(name)]
        if len(tempvar_full.dimensions)==1:
            if str(tempvar_full.dimensions[0])==globe_unlimdimname:
                unlimdimvar_name=str(name)
            elif str(tempvar_full.dimensions[0])==globe_limdimname1:
                limdimvar1_name=str(name)
            elif str(tempvar_full.dimensions[0])==globe_limdimname2:
                limdimvar2_name=str(name)

    limdimvar1_full=f1.variables[limdimvar1_name]
    limdimvar2_full=f1.variables[limdimvar2_name]
    unlimdimvar_full=f1.variables[unlimdimvar_name]

class Ncdata(object):
    """
    NCdata is object facilitating maping and exploring nc file.
    """
    _timedim_candidate_list=['time_counter','time','tstep']
    _timevar_name_can_list=['time_counter','time']
    _default_pftdim='PFT' #not used
    flags=OrderedDict()
    flags['NonbioAdjust'] = False
    flags_original = flags.copy()

    def __init__(self,filename,latlon_dim_name=None,largefile=False,multifile=False,
                 replace_nan=False,print_timecheck=True):
        """
        Parameters:
        -----------
        latlon_dim_name: a tuple giving the names of lat/lon dimension
            names of the file (eg. ('latdimname','londimname')), if it's
            None, they will be guessed from the default candidate list.
        largefile: boolen. When True, variable datat will not be loaded
            automatically but could be loaded with the method
            retrieve_variables and de-loaed using remove_variables.
        multifile: True when a file pattern is given.
        replace_nan: replace the Nan (detected using np.isnan) in
            the file and replace them uisng 0.
        print_timecheck: False to suppress warning information from time
            axis and time variable check.

        Notes:
        ------
        1. The lat/lon attributes of the Ncdata object are
            the only cooridates used in many downstream calculations,
            such as find_index_by_region,mapping,etc. In cases the
            lat/lon need to be adjusted in case of ill-defined input
            grid. The original lat/lon could be find in the attributes
            of latorg/lonorg
        """
        self.filename=filename
        self.fpro_replace_nan = replace_nan
        if not multifile:
            if os.path.isfile(filename):
                pass
            else:
                raise IOError("{0} does not exist".format(filename))

        if multifile:
            grp=nc.MFDataset(filename)
        else:
            grp=nc.Dataset(filename)

        self.grp = grp
        self.dimensions = grp.dimensions
        self.dim_name_list = self.dimensions.keys()
        self.varlist_all = grp.variables.keys()
        self.global_attributes = dict(grp.__dict__)

        #guess the unlimited dim name
        timedim_detect = False
        for timedim_candidate in Ncdata._timedim_candidate_list:
            if timedim_candidate in grp.dimensions.keys():
                tempdim=grp.dimensions[timedim_candidate]
                if tempdim.isunlimited():
                    self.unlimited_dimname=timedim_candidate
                    self.unlimited_dimlen=len(tempdim)
                    timedim_detect = True
                    break
                else:
                    if print_timecheck:
                        print "Warning! the dimension {0} exists but is not unlimited dimension".format(timedim_candidate)
                    break
            else:
                pass
        if not timedim_detect:
            if print_timecheck:
                print """unlimited_dimname not found, please make sure the file has no unlimited dimension,
                         otherwise please expand the time dimension name candidate list"""
            self.unlimited_dimname = None

        #guess time variable
        timevar_detect = False
        for timevar_name_can in Ncdata._timevar_name_can_list:
            if timevar_name_can in self.varlist_all:
                if grp.variables[timevar_name_can].dimensions[0] == self.unlimited_dimname:
                    self.timevar_name = timevar_name_can
                    timevar_detect = True
                    break
                else:
                    print "Warning! candidate timevar {0} detected but its dimensions is not {1}".format(timevar_name_can,self.unlimited_dimname)
                    break
            else:
                pass
        if timevar_detect:
            self.timevar = grp.variables[self.timevar_name][:]
        else:
            if print_timecheck:
                print "timevar not found. please make sure there is not time var in the file"
            self.timevar = None
            self.timevar_name = None

        #guess the lat/lon dimension names
        if latlon_dim_name is None:
            if 'lat' in self.dimensions and 'lon' in self.dimensions:
                self.latdim_name = 'lat'
                self.londim_name = 'lon'
            elif 'LAT' in self.dimensions and 'LON' in self.dimensions:
                self.latdim_name = 'LAT'
                self.londim_name = 'LON'
            elif 'latdim' in self.dimensions and 'londim' in self.dimensions:
                self.latdim_name = 'latdim'
                self.londim_name = 'londim'
            elif 'LATITUDE' in self.dimensions and 'LONGITUDE' in self.dimensions:
                self.latdim_name = 'LATITUDE'
                self.londim_name = 'LONGITUDE'
            elif 'latitude' in self.dimensions and 'longitude' in self.dimensions:
                self.latdim_name = 'latitude'
                self.londim_name = 'longitude'
            elif 'Latitude' in self.dimensions and 'Longitude' in self.dimensions:
                self.latdim_name = 'Latitude'
                self.londim_name = 'Longitude'
            elif 'x' in self.dimensions and 'y' in self.dimensions:
                self.latdim_name = 'y'
                self.londim_name = 'x'
            else:
                raise ValueError("lat/lon dimension names could not be guessed, please either provide the latlon_dim_name or expand default guess list")
        else:
            self.latdim_name = latlon_dim_name[0]
            self.londim_name = latlon_dim_name[1]


        #set default latvar,lonvar name and retrieve lonvar/latvar values.
        if 'lat' in self.varlist_all and 'lon' in self.varlist_all:
            self.latvar_name='lat'
            self.lonvar_name='lon'
        elif 'LAT' in self.varlist_all and 'LON' in self.varlist_all:
            self.latvar_name='LAT'
            self.lonvar_name='LON'
        elif 'LATITUDE' in self.varlist_all and 'LONGITUDE' in self.varlist_all:
            self.latvar_name='LATITUDE'
            self.lonvar_name='LONGITUDE'
        elif 'Latitude' in self.varlist_all and 'Longitude' in self.varlist_all:
            self.latvar_name='Latitude'
            self.lonvar_name='Longitude'
        elif 'latitude' in self.varlist_all and 'longitude' in self.varlist_all:
            self.latvar_name='latitude'
            self.lonvar_name='longitude'
        elif 'nav_lat' in self.varlist_all and 'nav_lon' in self.varlist_all:
            self.latvar_name='nav_lat'
            self.lonvar_name='nav_lon'
        elif 'latvar' in self.varlist_all and 'lonvar' in self.varlist_all:
            self.latvar_name='latvar'
            self.lonvar_name='lonvar'
        else:
            raise ValueError('Default lat and lon names are not found in the data, please specify by add_latvar_lonvar_name, or update default list')

        #find lat/lon values
        org_latvar = grp.variables[self.latvar_name][:]
        if np.ndim(org_latvar) in [0,1]:
            self.latorg = org_latvar
        elif np.ndim(org_latvar) == 2:
            self.latorg = org_latvar[:,0]
        else:
            raise ValueError("the lat variable ndim is {0}".format(np.ndim(org_latvar)))

        org_lonvar = grp.variables[self.lonvar_name][:]
        if np.ndim(org_lonvar) in [0,1]:
            self.lonorg = org_lonvar
        elif np.ndim(org_lonvar) == 2:
            self.lonorg = org_lonvar[0,:]
        else:
            raise ValueError("the lon variable ndim is {0}".format(np.ndim(org_lonvar)))

        self.geo_limit={'lat':(np.min(self.latorg),np.max(self.latorg)), 'lon':(np.min(self.lonorg),np.max(self.lonorg)),
                        'numlat':len(self.latorg),'numlon':len(self.lonorg)}
        self.rlat = self.geo_limit['lat']
        self.rlon = self.geo_limit['lon']

        #This is for compatiblity with previous version.
        self.lat_name = self.latvar_name
        self.lon_name = self.lonvar_name
        self.lat = self.latorg.copy()
        self.lon = self.lonorg.copy()

        #build the dimension variable list
        self.dimvar_name_list = [self.lonvar_name, self.latvar_name, self.timevar_name]
        #get PFT variable
        if 'PFT' in self.varlist_all:
            self.pftvar_name = 'PFT'
            self.pftvar = grp.variables[self.pftvar_name][:]
            self.dimvar_name_list.append(self.pftvar_name)


        #read the variables by form into d0
        self.d0 = g.ncdata()
        for var in grp.variables.keys():
            self.d0.__dict__[var]=grp.variables[var]

        #set the d1 to hold variable values
        self.d1 = g.ncdata()
        varlist_var = pb.StringListAnotB(self.varlist_all,self.dimvar_name_list)

        self.largefile = largefile
        if largefile == False:
            for var in varlist_var:
                if self.fpro_replace_nan == False:
                    self.d1.__dict__[var] = pb.Remove_dupdim(self.grp.variables[var][:])
                elif self.fpro_replace_nan == True:
                    dt = self.grp.variables[var][:]
                    dt[np.nonzero(np.isnan(dt))] = 0.
                    self.d1.__dict__[var] = pb.Remove_dupdim(dt)
                else:
                    raise ValueError("Unknown value for replace_nan")
            self.varlist = varlist_var[:]
        else:
            self.varlist = []
            for var in ['Areas','VEGET_MAX','CONTFRAC','NONBIOFRAC','vegetfrac','maxvegetfrac']:
                if var in varlist_var:
                    self.d1.__dict__[var] = pb.Remove_dupdim(grp.variables[var][:])
                    self.varlist.append(var)

        #Calculate Contiental Area
        if 'Areas' in self.varlist and 'CONTFRAC' in self.varlist:
            self.d1.__dict__['ContAreas'] = self.d1.__dict__['Areas'] * self.d1.__dict__['CONTFRAC']
            if Ncdata.flags['NonbioAdjust']:
                if 'NONBIOFRAC' in self.varlist:
                    self.d1.__dict__['ContAreas'] = self.d1.__dict__['ContAreas'] * (1-self.d1.__dict__['NONBIOFRAC'])
                else:
                    raise ValueError("NonbioAdjust is True but NONBIOFRAC is not found!")

        #append a proxy pftsum attribute
        self.pftsum = g.ncdata()

        #check whethere the file is a single-point file
        if len(self.lat) == 1 and len(self.lon) == 1:
            self._SinglePoint = True
        else:
            self._SinglePoint = False

    @classmethod
    def flags_restore(cls):
        cls.flags = cls.flags_original.copy()

    def add_latvar_lonvar_name(cls,latvar_name,lonvar_name):
        self.latvar_name=latvar_name
        self.lonvar_name=lonvar_name

    def __repr__(self):
        return '\n'.join([repr(self.__class__),self.filename])

    def __len__(self):
        return len(self.timevar)

    def close(self):
        self.grp.close()

    def show_latlon(self):
        print self.lat[[0,1,-2,-1]],len(self.lat),self.lat[1]-self.lat[0]
        print self.lon[[0,1,-2,-1]],len(self.lon),self.lon[1]-self.lon[0]

    def retrieve_variables(self,varlist,mask=None):
        """
        Retrieve varlist.
        """
        for var in varlist:
            if var in self.varlist:
                pass
            else:
                if self.fpro_replace_nan == False:
                    self.d1.__dict__[var] = pb.Remove_dupdim(self.grp.variables[var][:])
                elif self.fpro_replace_nan == True:
                    dt = self.grp.variables[var][:]
                    dt[np.nonzero(np.isnan(dt))] = 0.
                    self.d1.__dict__[var] = pb.Remove_dupdim(dt)
                else:
                    raise ValueError("Unknown value for replace_nan")
                self.varlist.append(var)

                if mask is not None:
                    self.d1.__dict__[var] = mathex.ndarray_mask_smart_apply(
                                            self.d1.__dict__[var],mask)

    def remove_variables(self,varlist,pftsum=False):
        """
        Remove varlist from gnc.d1. but not gnc.pftsum and gnc.spasum/spamean
            if there are any.
        """
        for var in varlist:
            del self.d1.__dict__[var]
            self.varlist.remove(var)
            if pftsum:
                del self.pftsum.__dict__[var]


    def set_timevar(self,taxis):
        self.timevar = taxis

    def get_pftsum(self,varlist=None,area_include=True,veget_npindex=np.s_[:],
                   print_info=False,name_vegetmax=None):
        """
        Get PFT (VEGET_MAX) weighted average of variables.
        Parameters:
        -----------
        varlist: limit varlist scope
        veget_npindex: 
            1. could be used to restrict for example the PFT
            weighted average only among natural PFTs by setting
            veget_npindex=np.s_[:,0:11,:,:]. It will be used to slice
            VEGET_MAX or vegetfrac or maxvegetfrac variable.
            2. could also be used to slice only for some subgrid
            of the whole grid, eg., veget_npindex=np.s_[...,140:300,140:290].

        Note:
        -----
        1. This function is very flexible, it can handle cases of spatial dataset
            with single or multiple time steps, or a single point file with
            multiple time steps. This is confirmed again on 24/02/2016
        """
        if varlist is None:
            varlist = self.varlist

        #retrieve vars if they're not retrieved yet.
        vars_retrieve = pb.StringListAnotB(varlist,['Areas','VEGET_MAX','CONTFRAC']+self.varlist)
        self.retrieve_variables(vars_retrieve)

        #set for name_vegetmax
        if name_vegetmax is None:
            if 'VEGET_MAX' in self.varlist:
                name_vegetmax = 'VEGET_MAX'
            elif 'vegetfrac' in self.varlist:
                name_vegetmax = 'vegetfrac'
            else:
                raise ValueError("""name_vegetmax is None but cannot guess its value""")

        d0=self.d0
        d1=self.d1
        pft_length = d1.__dict__[name_vegetmax].shape[0]

        print "*******PFT VEGET_MAX weighted sum begin******"
        for var in varlist:
            if var not in ['VEGET_MAX','vegetfrac','maxvegetfrac']:
                #4-dim variable before squeeze
                if d0.__dict__[var].ndim==4:
                    if 'PFT' not in d0.__dict__[var].dimensions and 'veget' not in d0.__dict__[var].dimensions:
                        if print_info: print """Warning! Original var '{0}' has 4 dimensions before squeezing,
                                 but PFT or veget is not one dimension.""".format(var)
                    #'PFT' is one dimension
                    else:
                        #if 'PFT' is the second dimension
                        if d0.__dict__[var].dimensions[1] in ['PFT','veget']:
                            #first/time dim lenght is not 1
                            if d1.__dict__[var].ndim == 4:
                                operation_axis = 1
                            #first/time dim length is 1
                            elif d1.__dict__[var].ndim == 3:
                                operation_axis = 0
                            #first/time dim length is not 1 but the opned file
                            #has only a single point.
                            elif d1.__dict__[var].ndim == 2:
                                #we verify if it's a single-point file
                                if self._SinglePoint:
                                    operation_axis = 1
                                else:
                                    if print_info: print """Warning! var '{0}' has only 2 dimensions after squeezing,
                                                        but the file is not a single-point file.""".format(var)
                            else:
                                raise ValueError("""Original 4-dim var '{0}' is a single-dimension
                                                    array after being squeezed""".format(var))

                            #if no errors occurs in above, we process the data
                            veget_max=d1.__dict__[name_vegetmax][veget_npindex]
                            vardata = d1.__dict__[var][veget_npindex]
                            temp=vardata*veget_max
                            temppftsum=np.ma.sum(temp,axis=operation_axis)
                            self.pftsum.__dict__[var]=temppftsum
                            if print_info: print '{0} treated'.format(var)

                        else:
                            if print_info: print """Warning! Original var '{0}' has 4 dimensions,
                                                but the second dimension is not PFT or veget""".format(var)
                #3-dim variable before squeeze
                elif d0.__dict__[var].ndim==3:
                    #if PFT is not one of the dimensions, probably
                    #this variable has already been PFT-weigthed within the model.
                    if 'PFT' not in d0.__dict__[var].dimensions and 'veget' not in d0.__dict__[var].dimensions:
                        self.pftsum.__dict__[var]=d1.__dict__[var][veget_npindex]
                    else:
                        # if there is a timevar and its length is
                        # one, and the length if its first dim is of
                        # pft_length, maybe this is still a variable with PFT as one
                        # of its dimensions, so we do PFT weighting
                        # here.
                        if self.timevar is not None and len(self.timevar) == 1\
                            and d1.__dict__[var].shape[0] == pft_length:

                            veget_max=d1.__dict__[name_vegetmax][veget_npindex]
                            vardata = d1.__dict__[var][veget_npindex]
                            temp=vardata*veget_max
                            temppftsum=np.ma.sum(temp,axis=0)
                            self.pftsum.__dict__[var]=temppftsum

                        else:
                            if print_info: print """Warning! Original var '{0}' has 3 dim but
                                                PFT or veget is still one dimension""".format(var)
                            self.pftsum.__dict__[var]=d1.__dict__[var][veget_npindex]

                #<=2 dim variable before squeeze
                else:
                    if print_info: print """Original var '{0}' has 2 dim before sequeezing""".format(var)


        #if the variable `Areas` is contained in original file, we copy it
        #into the `pftsum` attribute.
        # if area_include:
        #     try:
        #         self.pftsum.__dict__['Areas'] = d1.__dict__['Areas'][veget_npindex]
        #     except KeyError:
        #         print "Warning! Areas not in the history file!"

        print "*******END of PFT sum******"

    def _get_npindex_dimnum_tree_grass(self,dimnum):
        """
        Return the dict of tree/grass npindex depending on their dimnum.
        """
        if dimnum == 4:
            tree = np.s_[:,1:9,...]
            grass = np.s_[:,9:11,...]
            crop = np.s_[:,11:13,...]
        elif dimnum == 3:
            tree = np.s_[1:9,...]
            grass = np.s_[9:11,...]
            crop = np.s_[11:13,...]
        elif dimnum == 2:
            tree = np.s_[:,1:9]
            grass = np.s_[:,9:11]
            crop = np.s_[:,11:13]
        else:
            raise ValueError("Can only handle dimnum=2,3,4")

        dic = dict(zip(['tree','grass','crop'],[tree,grass,crop]))
        return dic


    def get_pftsum_tree_grass_crop(self,varlist,groups=None,dimnum=4):
        """
        This is only a shortcut of get_pftsum to allow easy retrieval of
        vegetmax-weighted variable values limited to three broad PFT groups.

        Parameters:
        -----------
        dimnum: the number of dimensions of the input variables after squeezing.
        groups: list, could be sublist of ['tree','grass','crop']

        Return:
        -------
        for len(varlist) == 1, a dictionary of groups will be returned; otherwise
            a nested dictionary will be returned, with first-tier keys as varlist,
            and second-tier keys as groups. This could be readily fed into pandas
            Panel constructor.

        Notes:
        ------
        This function is mainly designed to allow faster retrieval for the
        single-pointed file.

        Warnings:
        ------
        This function is simple wrapper of get_pftsum method, so beware that
        the `pftsum` will change after calling this method.
        """
        outdic = OrderedDict()
        if groups is None:
            groups = ['tree','grass','crop']

        dic_npindex = self._get_npindex_dimnum_tree_grass(dimnum)

        for name in groups:
            self.get_pftsum(varlist=varlist,veget_npindex=dic_npindex[name],
                            area_include=False)
            outdic[name] = self.pftsum.__dict__.copy()

        outdic = pb.Dic_Nested_Permuate_Key(outdic)
        if len(varlist) == 1:
            return outdic[varlist[0]]
        else:
            return outdic


    def get_spa(self,pftop=True,varind=np.s_[:],areaind=np.s_[:]):
        """
        Get the PFT weighed spatial mean and sum, only non-masked values
            will be considered.

        Notes:
        ------
        1. Only variables which are keys to Ncdata.pftsum will be treated.
        """
        if not hasattr(self,'pftsum'):
            if pftop==True:
                self.get_pftsum()
                d2=self.pftsum
            else:
                d2=self.d1
        else:
            d2=self.pftsum

        d3=g.ncdata()  #sum
        d4=g.ncdata()  #mean
        for var in d2.__dict__.keys():
            if var != 'Areas':
                temppftsum=d2.__dict__[var][varind]
                try:
                    area = self.d1.__dict__['ContAreas'][areaind]
                except KeyError:
                    try:
                        area = self.d1.__dict__['Areas'][areaind]
                    except KeyError:
                        raise KeyError("""Error! ContAreas or Areas not in the file
                                        and cannot carry out spatial sum!""")

                try:
                    temparea = temppftsum * area
                except ValueError:
                    pdb.set_trace()
                tempspa_sum=np.ma.sum(np.ma.sum(temparea,axis=-1),axis=-1)
                d3.__dict__[var]=tempspa_sum
                tempspa_mean=tempspa_sum/np.ma.sum(np.ma.sum(area,axis=-1), axis=-1)
                d4.__dict__[var]=tempspa_mean
            else:
                pass
        self.spasum=d3
        self.spamean=d4

    def combine_vars(self,varlist,pftsum=False):
        """
        combine variables into a ndarray by adding a new dimension.
        """
        out_array = []
        if pftsum == False:
            data = self.d1
        else:
            data = self.pftsum
        for varname in varlist:
            out_array.append(data.__dict__[varname])
        return np.ma.array(out_array)

    def list_var(self,keyword=None):
        """
        supply keyword for show var names matching the pattern.
        supply "excludedim" to keyword to exclude the dim varnames in the
        output list.
        """
        if keyword is None:
            return self.d0.__dict__.keys()
        else:
            if keyword == 'excludedim':
                return pb.StringListAnotB(self.d0.__dict__.keys(),
                                          self.dimvar_name_list+['Areas','VEGET_MAX'])
            else:
                return pb.FilterStringList(keyword,self.d0.__dict__.keys())

    def list_var_attr(self,varlist,attrname=None):
        """
        list var attributes.
        """
        if isinstance(varlist,(list,tuple)):
            if attrname is None:
                raise ValueError("must give attrname when a list of vars is given")
            else:
                for varname in varlist:
                    print "{0} : {1}".format(varname,self.d0.__dict__[varname].__dict__[attrname])

        elif isinstance(varlist,str):
            varname = varlist
            for attr_name in self.d0.__dict__[varname].__dict__.keys():
                print "{0} : {1}".format(attr_name,self.d0.__dict__[varname].__dict__[attr_name])
        else:
            raise TypeError("input must be of type string or iterable")

    @staticmethod
    def _apply_function_ndarray(mapvar,pyfunc=None):
        if pyfunc is not None:
            if callable(pyfunc):
                mapvar=pyfunc(mapvar)
            elif isinstance(pyfunc,list) and isfunction(pyfunc[0]):
                for subpyfunc in pyfunc:
                    mapvar=subpyfunc(mapvar)
            else:
                mapvar=mapvar*pyfunc
        return mapvar

    def _retrieve_data_for_map(self,mapvarname=None,
                               grid=None,pftsum=False,
                               forcedata=None,mapdim=None,
                               agremode=None,pyfunc=None,
                               mask=None,mask_value=None,
                               npindex=np.s_[:]):

        """
        Parameters:
        -----------
        mask: Boolean mask to be applied on the data.
        grid: A tuple of (lat1,lon1,lat2,lon2)
        """
        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)

        #read mapvar data and prepare for mapping
        if mapvarname is None:
            if forcedata is None:
                raise ValueError("mapvarname and forcedata both as None!")
            else:
                mapvar = forcedata
        else:
            if isinstance(mapvarname,str):
                if forcedata is None:
                    mapvar=final_ncdata.__dict__[mapvarname]
                else:
                    mapvar=forcedata
            else:
                raise ValueError('mapvarname must be string type')
        if mapvar.ndim==2 and (mapdim is not None or agremode is not None):
            raise ValueError("""{0} has only 2 valid dimension but
                             mapdim or agremode is not None""".format(mapvarname))
        elif mapvar.ndim==3:
            if agremode is None and mapdim is None:
                raise ValueError("""{0} has 3 valid dimension, cannot
                                 leave both mapdim and agremode as None"""
                                 .format(mapvarname))
            #we want to transform data directly for plotting
            elif agremode is not None:
                #make sum or mean of monthly data
                if agremode=='sum':
                    mapvar=mathex.m2ysum(mapvar)
                elif agremode=='mean':
                    mapvar=mathex.m2ymean(mapvar)
                elif agremode=='fsum':
                    mapvar=np.ma.sum(mapvar,axis=0)
                elif agremode=='fmean':
                    mapvar=np.ma.mean(mapvar,axis=0)

                #extract data for only specified dimension
                if mapdim is not None:
                    #the original mapvar has 3 dim with first dim of 12.
                    if mapvar.ndim==2:
                        raise ValueError("""{0} has 3 valid dimension with first
                                         dim size as 12, cannot specify agremode
                                         and mapdim simultaneously"""
                                         .format(mapvarname))
                    else: #mapvar.ndim==3
                        mapvar=mapvar[mapdim,:,:]
                #error handling when mapdim is None
                else:
                    if mapvar.ndim==2: #the original mapvar has 3 dim with first dim of 12.
                        pass
                    else: #mapvar.ndim==3
                        raise ValueError("""the dimension after data
                                         transformation is {0}, must specify
                                         mapdim to plot""".format(mapvar.shape))
            #plot specified dimension for data without transformation
            else:
                mapvar=mapvar[mapdim,:,:]
        elif mapvar.ndim >= 4:
            mapvar = mapvar[npindex]

        #apply the defined function on data
        mapvar = self._apply_function_ndarray(mapvar,pyfunc)

        #apply mask
        if mask is not None:
            mapvar=mathex.ndarray_mask_smart_apply(mapvar,mask)
        if mask_value is not None:
            mapvar=np.ma.masked_equal(mapvar,mask_value)

        #apply grid
        latvar,lonvar = self.Get_latlon_by_Grid(grid=grid)
        if grid is None:
            pass
        else:
            (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = \
                self.find_index_by_vertex((grid[0],grid[2]),(grid[1],grid[3]))
            mapvar = mapvar[lat_index_min:lat_index_max+1,
                            lon_index_min:lon_index_max+1]

        #finally, if lat is provide in increasing sequence, flip over the data.
        if latvar[0]<latvar[-1]:
            latvar=latvar[::-1]
            mapvar=np.flipud(mapvar)

        return latvar,lonvar,mapvar


    def _set_map_title_unit_by_mapvarname(self,mapvarname,axt,
                                          unit=None,
                                          title=None):
        mapvar_full=self.d0.__dict__[mapvarname]
        def retrieve_external_default(external_var,attribute):
            if external_var is not None:
                outvar=external_var
            elif hasattr(mapvar_full,attribute):
                if external_var==False:
                    outvar=''
                else:
                    outvar=mapvar_full.getncattr(attribute)
            else:
                outvar=None
            return outvar
        #retrieve title or unit
        map_unit=retrieve_external_default(unit,'units')
        map_title=retrieve_external_default(title,'long_name')
        try:
            agre_title_complement='[yearly '+agremode+']'
        except:
            agre_title_complement=None

        #function handling ax title
        def set_title_unit(ax,title=None,unit=None,agre_title_complement=None):
            try:
                title_unit=title+('\n'+unit)
            except TypeError:
                try:
                    title_unit=title
                    title_agre=agre_title_complement+' '+title
                except TypeError:
                    pass
            finally:
                try:
                    title_full=agre_title_complement+' '+title_unit
                except TypeError:
                    title_full=title_unit
            if title_full is not None:
                ax.set_title(title_full)
            else:
                pass
        set_title_unit(axt,map_title,map_unit,agre_title_complement)

    @append_doc_of(bmap.mapcontourf)
    def map(self,mapvarname=None,
            forcedata=None,mapdim=None,
            agremode=None,pyfunc=None,mask=None,
            unit=None,title=None,pftsum=False,mask_value=None,
            grid=None,npindex=np.s_[:],
            projection='cyl',mapbound='all',gridstep=None,
            shift=False,cmap=None,map_threshold=None,
            colorbarlabel=None,levels=None,show_colorbar=True,
            data_transform=False,ax=None,
            colorbardic={},
            maptype='con',
            cbarkw={},
            gmapkw={},
            **kwargs):
        """
        This is an implementation of bamp.mapcontourf

        Parameters:
        -----------
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        mask: Boolean mask to be applied on the data.

        Docs for bmap.mapcontourf:
        -------------------------
        """
        mlat,mlon,mdata = self._retrieve_data_for_map(
                               mapvarname=mapvarname,
                               grid=grid, pftsum=pftsum,
                               forcedata=forcedata, mapdim=mapdim,
                               agremode=agremode, pyfunc=pyfunc,
                               mask=mask, mask_value=mask_value,
                               npindex=npindex)

        if maptype == 'con':
            mapfunc = getattr(bmap,'mapcontourf')
        elif maptype == 'img':
            mapfunc = getattr(bmap,'mapimshow')
        else:
            raise ValueError("Unknow map type!")

        mcon = mapfunc(data=mdata,lat=mlat,lon=mlon,
                       projection=projection,
                       mapbound=mapbound, gridstep=gridstep,
                       shift=shift, cmap=cmap,
                       map_threshold=map_threshold,
                       show_colorbar=show_colorbar,
                       colorbarlabel=colorbarlabel,
                       levels=levels,
                       data_transform=data_transform,
                       ax=ax,colorbardic=colorbardic,
                       cbarkw=cbarkw,
                       gmapkw=gmapkw,
                       **kwargs)

        self.mcon = mcon
        self.m = mcon.m
        self.cbar = mcon.cbar
        self._set_map_title_unit_by_mapvarname(mapvarname,self.m.ax,
                                               unit=unit,title=title)

    def contourf(self,data,**kwargs):
        self.mcon.m.contourf(self.mcon.gmap.lonpro,self.mcon.gmap.latpro,data,**kwargs)

    def contour(self,data,**kwargs):
        self.mcon.m.contour(self.mcon.gmap.lonpro,self.mcon.gmap.latpro,data,**kwargs)


    def imshowmap(self,varname,forcedata=None,pftsum=False,ax=None,projection='cyl',mapbound='all',gridstep=(30,30),shift=False,colorbar=True,
                  colorbarlabel=None,*args,**kwargs):
        """
        A temporary wrapper of bamp.imshowmap
        """
        if pftsum==False:
            d1=self.d1
        else:
            d1=self.pftsum

        if forcedata is not None:
            data = forcedata
        else:
            data = d1.__dict__[varname]
        m,cs,cbar = bmap.imshowmap(self.lat,self.lon,data,ax=ax,projection=projection,mapbound=mapbound,gridstep=gridstep,shift=shift,colorbar=colorbar,
                                   colorbarlabel=colorbarlabel,*args,**kwargs)
        self.m=m
        self.cbar=cbar
        self.cs=cs

    def scatter(self,lon,lat,*args,**kwargs):
        """
        Draw scatter points on the map.

        Parameters:
        -----------
        lon/lat: single value for 1D ndarray.
        index keyword: lon/lat could be the indexes when index == True,
            when making scatter plots, index is removed from the keys
            before it's passed to scatter function.
        """
        index = kwargs.get('index',False)
        if index == True:
            lon = self.lon[lon]
            lat = self.lat[lat]
            del kwargs['index']
        elif index == False:
            pass
        else:
            raise TypeError("keyword index must boolean type!")

        x,y = self.m(lon,lat)
        self.m.scatter(x,y,*args,**kwargs)

    def add_Rectangle(self,(lat1,lon1),(lat2,lon2),index=False,
                      **kwargs):
        """
        Add a rectangle of by specifing (lat1,lon1) and (lat2,lon2).

        Parameters:
        -----------
        (lat1,lon1): lowerleft coordinate
        (lat2,lon2): upperright coordinate
        index: if index is True, the input (lat1,lon1),(lat2,lon2) will
            not be treated as lat/lon values but as the index to
            retrieve lat/lon values.
        """
        if index:
            lat1,lat2 = self.lat[lat1],self.lat[lat2]
            lon1,lon2 = self.lon[lon1],self.lon[lon2]
        else:
            pass
        (x1,x2),(y1,y2) = self.m([lon1,lon2],[lat1,lat2])
        rec = mat.patches.Rectangle((x1,y1),x2-x1,y2-y1,**kwargs)
        self.m.ax.add_patch(rec)
        return rec


    def add_Rectangle_grid(self,grid=None,index=False,
                      **kwargs):
        self.add_Rectangle((grid[0],grid[1]),(grid[2],grid[3]),index=index,**kwargs)

    def add_text(self,lat,lon,s,fontdict=None,index=False,**kwargs):
        """
        Add text s on the position (lat,lon)

        Parameters:
        -----------
        index: lon/lat could be the indexes when index == True.
        """
        if index == True:
            lon = self.lon[lon]
            lat = self.lat[lat]
        elif index == False:
            pass
        else:
            raise TypeError("keyword index must boolean type!")
        x,y = self.m(lon,lat)
        self.m.ax.text(x,y,s,fontdict=fontdict,**kwargs)

    def add_text_by_dataframe(self,dataframe,name='region',
                                        fontdict=None,**kwargs):
        """
        Add a series of text by using fields from a dataframe.

        Parameters:
        -----------
        1.name: the name field in the dataframe which denotes the column of text.

        Notes:
        ------
        1. the dataframe should have "lat,lon" to denote the text position.
        """
        for cor_name in ['lon','lat']:
            if cor_name not in dataframe.columns:
                raise ValueError("{0} not a column name of dataframe").format(cor_name)
        for index,row in dataframe.iterrows():
            lat = row['lat']
            lon = row['lon']
            s = row[name]
            self.add_text(lat,lon,s,fontdict=fontdict,**kwargs)



    def add_Rectangle_list_coordinates(self,coordlist,textlist=None,fontdict=None,textkw={},**kwargs):
        """
        Add a list of [(lat1,lon1),(lat2,lon2)] to add a series of rectangles at one time.

        Parameters:
        -----------
        coordlist: a nested list with [(lat1,lon1),(lat2,lon2)] as its members.
        """
        for i,coord in enumerate(coordlist):
            self.add_Rectangle(coord[0],coord[1],**kwargs)
            if textlist is not None:
                lat = (coord[0][0]+coord[1][0])/2.
                lon = (coord[0][1]+coord[1][1])/2.
                self.add_text(lat,lon,textlist[i],fontdict=fontdict,**textkw)

    def add_Rectangle_list_by_dataframe(self,dataframe,label=False,
                                        fontdict=None,textkw={},**kwargs):
        """
        Add rectangle list by using dataframe.

        Notes:
        ------
        1. the dataframe should have "lat1,lat2,lon1,lon2,region" as column
            names.
        """
        for name in ['lat1','lat2','lon1','lon2','region']:
            if name not in dataframe.columns:
                raise ValueError("{0} not a column name of dataframe").format(name)
        region_name_list = []
        coordlist = []
        for index,row in dataframe.iterrows():
            region_name_list.append(row['region'])
            coordlist.append([(row['lat1'],row['lon1']),(row['lat2'],row['lon2'])])
        if label == False:
            textlist = None
        else:
            textlist = region_name_list
        self.add_Rectangle_list_coordinates(coordlist,textlist=textlist,
                        fontdict=fontdict,textkw=textkw,**kwargs)


    def find_index_by_point(self,vlat,vlon):
        """
        find the index for the point(vlat,vlon).

        Returns:
        --------
        (index_lat,index_lon) which could be used directly in slicing data.
        """
        ind_lat,ind_lon = find_index_by_point(self.lat,self.lon,(vlat,vlon))
        return (ind_lat,ind_lon)

    def find_index_by_vertex(self,(vlat1,vlat2),(vlon1,vlon2)):
        """
        Purpose: find the index specified by (vla1,vlat2),(vlon1,vlon2)
        Return: (lon_index_min, lon_index_max, lat_index_min, lat_index_max)
        Note: This is a direct application of gnc.find_index_by_vertex.
        1. To retrieve the data from derived indexes, one should use:
            np.s_[lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]
        """
        return find_index_by_vertex(self.lon, self.lat, (vlon1,vlon2), (vlat1,vlat2))

    def find_index_by_vertex_grid(self,grid=None):
        """
        Find the indices by using the grid information, using find_index_by_vertex.
        Return: (lon_index_min, lon_index_max, lat_index_min, lat_index_max)

        Parameters:
        -----------
        grid: the grid information from Ncdata object.
        """
        return self.find_index_by_vertex((grid[0],grid[2]),(grid[1],grid[3]))

    def find_index_latlon_each_point_in_vertex(self,rlat=None,rlon=None):
        """
        Return a list of nested tuples, the nested tuple is like:
            ((index_lat,index_lon),(vlat,vlon)), suppose var is of shape
            (time,lat,lon), var[time,index_lat,index_lon] will allow
            to retrieve var value corresponding to (vlat,vlon)

        parameters:
        -----------
        rlat: range of lat, (valt1,vlat2)
        rlon: range of lon, (vlon1,vlon2)

        Notes:
        ------
        1. Suppose the rlat include m gridcells and rlon include n gridcells,
            the result will be mXn length list and it will allow to retrieve
            the variable values for each of the point which are within
            the rectangle given by rlat,rlon.

        See also,
        ---------
        Add_Single_Var_Pdata
        """
        rlat,rlon = self._return_defautl_latlon_range(rlat,rlon)
        (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = \
            self.find_index_by_vertex(rlat,rlon)
        index_latlon_tuple_list = []
        for inlat in range(lat_index_min, lat_index_max+1):
            for inlon in range(lon_index_min, lon_index_max+1):
                index_latlon_tuple_list.append((
                                    (inlat,inlon),
                                    (self.lat[inlat],self.lon[inlon])
                                    ))
        return index_latlon_tuple_list


    def _get_latlon_by_vertex(self,(vlat1,vlat2),(vlon1,vlon2)):
        (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = \
            self.find_index_by_vertex((vlat1,vlat2),(vlon1,vlon2))
        sublat = self.lat[lat_index_min:lat_index_max+1]
        sublon = self.lon[lon_index_min:lon_index_max+1]
        return (sublat,sublon)

    def _get_var_by_grid(self,varname,pftsum=False,grid=None):
        """
        Only for internal use.
        """
        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)
        if grid is None:
            data = final_ncdata.__dict__[varname]
        else:
            data = self.Get_GridValue(varname,(grid[0],grid[2]),(grid[1],grid[3]),
                                     pftsum=pftsum)
        return data

    def Get_latlon_by_Grid(self,grid=None):
        """
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        """
        if grid is not None:
            (sublat,sublon)=self._get_latlon_by_vertex((grid[0],grid[2]),
                                                      (grid[1],grid[3]))
        else:
            (sublat,sublon)=(self.lat,self.lon)
        return sublat,sublon

    def Add_Var_to_Gdata3D(self,varname,grid=None,pftsum=False,
                           forcedata=None,func=None,taxis=None):
        """
        Add var to Gdata3D.
        """

        sublat,sublon = self.Get_latlon_by_Grid(grid=grid)
        data = self.Get_GridValue_grid(varname,grid=grid,pftsum=pftsum,
                                       forcedata=forcedata)
        if func is not None:
            data = func(data)

        if taxis is None:
            taxis = self.timevar

        return Gdata.Gdata3D(data=data,taxis=taxis,lat=sublat,lon=sublon)

    def Add_Vars_to_NGdata3D(self,varlist,grid=None,pftsum=False,
                           forcedata=None,taxis=None):
        """
        Add var to NGdata3D.
        """

        sublat,sublon = self.Get_latlon_by_Grid(grid=grid)

        dic = OrderedDict()
        for varname in varlist:
            dic[varname] = self.Get_GridValue_grid(varname,grid=grid,pftsum=pftsum,
                                       forcedata=forcedata)

        if taxis is None:
            taxis = self.timevar

        ng3d = Gdata.NGdata3D.from_dict_of_array(dic,taxis=taxis,lat=sublat,lon=sublon)
        return ng3d

    def Get_PointValue(self,var,(vlat,vlon),pftsum=False,forcedata=None):
        """
        Return a point value.

        Parameters:
        ----------
        var: variable name
        (vlat,vlon): the values of (lat,lon)
        pftsum: boolean
        forcedata: using external data
        """
        ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)
        ind_lat,ind_lon = find_index_by_point(self.lat,self.lon,(vlat,vlon))
        if forcedata is None:
            data = ncdata.__dict__[var]
        else:
            data = forcedata
        return data[...,ind_lat,ind_lon]

    def Get_GridValue(self,var,(vlat1,vlat2),(vlon1,vlon2), pftsum=False,
                      forcedata=None):
        """
        Get_GridValue(self,var,(vlat1,vlat2),(vlon1,vlon2), pftsum=False,
                      forcedata=None):
        """
        (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = find_index_by_vertex(self.lon, self.lat, (vlon1,vlon2), (vlat1,vlat2))
        if pftsum:
            try:
                data = self.pftsum.__dict__[var]
            except AttributeError:
                raise ValueError("please do pftsum operation first!")
        else:
            data = self.d1.__dict__[var]
        if forcedata is not None:
            data = forcedata
        return data[..., lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1]

    def Get_GridValue_grid(self,var,grid=None,pftsum=False,forcedata=None):
        """
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        """
        if grid is None:
            grid = self.grid
        return self.Get_GridValue(var,(grid[0],grid[2]),(grid[1],grid[3]),pftsum=pftsum,forcedata=forcedata)

    def extract_values(self,var,lat,lon,pftsum=False,forcedata=None):
        """
        Extract the values from a series of lat,lon pairs.

        Parameters:
        -----------
        var,lat,lon: varname to be retrieved and lat/lon ndarray.
        pftsum, forcedata: as others
        """
        list_index = []
        list_value = []
        for vlat,vlon in zip(lat,lon):
            ind = self.find_index_by_point(vlat,vlon)
            list_index.append(ind)
            val = self.Get_PointValue(var,(vlat,vlon),pftsum=pftsum,forcedata=forcedata)
            list_value.append(val)
        dft = pa.DataFrame({'Location':list_index,var:list_value})
        return dft

    def Plot_PointValue(self,var,(vlat,vlon),ax=None,ylab=False,pyfunc=None,pftsum=False,**kwargs):
        if ax is None:
            ax=plt.gca()

        data=self.Get_PointValue(var,(vlat,vlon))
        if pftsum==True:
            veget=np.ma.masked_array(self.Get_PointValue('VEGET_MAX',(vlat,vlon)),0.)
            data=np.ma.sum(data*veget,axis=1)
        if pyfunc is not None:
            if isfunction(pyfunc):
                data=pyfunc(data)
            else:
                data=data*pyfunc

        ax.plot(data,label=var,**kwargs)
        if ylab==True:
            try:
                ax.set_ylabel(self.d0.__dict__[var].getncattr('units'))
            except AttributeError:
                pass
        elif isinstance(ylab,str):
            ax.set_ylabel(ylab)
        elif ylab==False:
            pass
        else:
            raise TypeError("Incorrect ylab type")

    def Plot_PointValue_PFTsum(self,var,(vlat,vlon),ax=None,ylab=False,pyfunc=None):
        pass

    def get_zonal(self,varlist,pftsum=False,
                  npindex=np.s_[:],pyfunc=None,
                  area=False,mode='sum',forcedata=None):
        """
        Get the zonal mean or sum. Apply sequence: pftsum, npindex,pyfunc.
            After this sequential treatment, the data must be of dim = (lat,lon).

        Parameters:
        -----------
        npindex: used to retrieve part of the varname data
        pyfunc: being applied just after the retrieval of the data
        area: True to the "Areas" variable in the data; ndarray for directly
            supplying an area array. Will be used to get the flux amount. 
        mode: 'sum' or 'mean'
        """
        if isinstance(varlist,(str,unicode)):
            varname = varlist
            ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)
            if varname is not None:
                if forcedata is not None:
                    data = forcedata
                else:
                    data = ncdata.__dict__[varname]
            else:
                raise ValueError("must give either varname or forcedata")

            data = data[npindex]
            data = mathex.apply_func(data,pyfunc=pyfunc)

            if len(data.shape) > 2:
                raise ValueError("data shape >2 before final operation")

            if area is True:
                try:
                    data = data * self.d1.Areas
                except AttributeError:
                    raise ValueError("Areas not present in the ncdata")
            elif isinstance(area,np.ndarray):
                data = data * area
            else:
                pass

            if mode == 'sum':
                fmdata = np.ma.sum(data,axis=-1)
            elif mode == 'mean':
                fmdata = np.ma.mean(data,axis=-1)
            else:
                raise ValueError("wrong operation mode")

            return pa.Series(fmdata,index=self.lat)
        else:
            dic = OrderedDict()
            for var in varlist:
                s = self.get_zonal(var,pftsum=pftsum,
                                   npindex=npindex,pyfunc=pyfunc,
                                   area=area,mode=mode,forcedata=forcedata)
                dic[var] = s
            return pa.DataFrame(dic)


    def _get_final_ncdata_by_flag(self,pftsum=False,spa=None):
        """
        pftsum: True/False
        spa:string type, 'sum' or 'mean', used to retrieve Ncdata attributes.
        """
        if spa=='sum':
            final_ncdata=self.spasum
        elif spa=='mean':
            final_ncdata=self.spamean
        elif spa is None:
            if pftsum==True:
                final_ncdata=self.pftsum
            else:
                final_ncdata=self.d1
        else:
            raise ValueError('''spatial operation '{0}' not expected!'''
                             .format(spa))
        return final_ncdata

    def _return_defautl_latlon_range(self,rlat=None,rlon=None):
        if rlat is not None:
            pass
        else:
            rlat = self.geo_limit['lat']
        if rlon is not None:
            pass
        else:
            rlon = self.geo_limit['lon']
        return (rlat,rlon)


    @property
    def grid(self):
        lat = self.geo_limit['lat']
        lon = self.geo_limit['lon']
        return (lat[0],lon[0],lat[1],lon[1])

    @property
    def mapbound(self):
        lat = self.geo_limit['lat']
        lon = self.geo_limit['lon']
        return (lat[0],lat[1],lon[0],lon[1])

    def add_varname(self,varname,data):
        """
        add varname.
        """
        self.d0.__dict__[varname] = data
        self.d1.__dict__[varname] = data
        if varname not in self.varlist:
            self.varlist.append(varname)

    def add_var_by_dic(self,dic):
        for varname,data in dic.items():
            self.add_varname(varname,data)


    def apply_func(self,func,varlist=None,inplace=False):
        """
        Apply function for varnames in self.varlist and attach them back to
            self.d1 if inplace==True, otherwise return a dict.
        """
        dic = OrderedDict()
        if varlist is None:
            varlist = self.varlist[:]
        for var in varlist:
            if not inplace:
                dic[var] = func(self.d1.__dict__[var])
            else:
                self.d1.__dict__[var] = func(self.d1.__dict__[var])
        if not inplace:
            return dic


    @append_doc_of(_get_final_ncdata_by_flag)
    def Add_Vars_to_Pdata(self,varlist,npindex=np.s_[:],unit=True,
                          pd=None,pftsum=False,spa=None):
        """
        This will add the varnames in varlist to a Pdata object
            specified by npindex, note the npindex must be numpy
            index trick object (np.s_).
        """
        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum,
                            spa=spa)
        if pd is None:
            pd=Pdata.Pdata()
        for varname in varlist:
            data=final_ncdata.__dict__[varname][npindex]
            x=np.arange(len(data))
            pd.add_entry_noerror(x,data,varname)
            if unit==True:
                try:
                    pd.add_attr_by_tag(unit=[(varname,
                            self.d0.__dict__[varname].getncattr('units'))])
                except AttributeError:
                    pass
        return pd

    def Add_Single_Var_Pdata(self,varname,
                             rlat=None,
                             rlon=None,
                             pftsum=False,
                             pd=None,npindex=np.s_[:],):
        '''
        This is mainly to plot for each spatial point within the rectangle
            defined by rlat/rlon.

        Parameters:
        -----------
        lat: (lat1,lat2), lat range.
        lon: (lon1,lon2), lon ragne.

        Notes:
        ------
        npindex: npindex will be applied before slicing by lat/lon index.
            The data after npindex slicing should have len(rlat),len(rlon)
            as the last two dimensions.

        See also,
        ---------
        find_index_latlon_each_point_in_vertex
        '''
        if pd is None:
            pd = Pdata.Pdata()
        ydic = OrderedDict()

        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)
        data = final_ncdata.__dict__[varname][npindex]
        index_latlon_tuple_list = \
            self.find_index_latlon_each_point_in_vertex(rlat,rlon)
        for (inlat,inlon),lat_lon in index_latlon_tuple_list:
            ydic[str(lat_lon)] = data[...,inlat,inlon]
        pd.add_entry_sharex_noerror_by_dic(ydic)
        return pd

    def Add_Varlist_NestedPdata(self,varlist,rlat=None,rlon=None,
                           pftsum=False,npindex=np.s_[:]):
        '''
        Plot a list of variables for each point within the rectangle defined by
            rlat/rlon. Return a dictionry of Pdata.Pdata instances, with
            viriable names as keys.
        '''
        outdic = OrderedDict()
        for varname in varlist:
            pd = self.Add_Single_Var_Pdata(varname,rlat=rlat,rlon=rlon,
                                           pftsum=pftsum,npindex=npindex)
            outdic[varname] = pd
        return Pdata.NestPdata(outdic)

    def Plot_Single_Var_Each_Point(self,varname,npindex=np.s_[:],
                                   pftsum=False,**legkw):
        '''
        Parameters:
        -----------
        legkw: as in plt.legend()
        '''
        pd = self.Add_Single_Var_Pdata(varname,npindex=npindex,pftsum=pftsum)
        pd.plot()
        pd.set_legend_all(taglab=True,**legkw)
        return pd

    def Plot_Varlist_Each_Point(self,varlist,npindex=np.s_[:],
                                pftsum=False,
                                legtag=None,
                                legtagseq=None,legkw={},
                                plotkw={},
                                **kwargs):
        '''
        Parameters:
        -----------
        kwargs: for Pdata.NestPdata.plot_split_parent_tag
        legkw: for plt.legend
        plotkw: for plt.plot
        '''
        npd = self.Add_Varlist_NestedPdata(varlist)
        npd.plot_split_parent_tag(plotkw=plotkw,
                                  legtag=legtag,
                                  legtagseq=legtagseq,
                                  legkw=legkw,
                                  **kwargs)
        pd0 = npd.child_pdata[varlist[0]]
        #pd0.set_legend_all(taglab=True,**legkw)

    def Add_Vars_to_Mdata(self,varlist,grid=None,
                          mask_by=None,npindex=np.s_[:],
                          md=None,pftsum=False,
                          transform_func=None,
                          prefix=None):
        '''
        Add several vars to Mdata for easy mapping, this is a simple
            warpper of Add_Vars_to_Dict_Grid.

        Parameters:
        -----------
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        mask_by:
            1. in case of a boolean numpy array, the mask will be directly
               apply on the retrieved array by using
               mathex.ndarray_mask_smart_apply;
            2. in case of a tuple like (varname,{'lb':2000,'ub':5000}), the mask
               will first be generated using mathex.ndarray_mask_by_threshold,
               followed by mathex.ndarray_mask_smart_apply.
            3. in case of a function, it could be like
               lambda x:np.ma.masked_invalid(x)
            4. it could a non-masked function, eg.
               mask_by=lambda x:np.ma.mean(x,axis=0), used to perform
               proper data transfrom.
        npindex: further index the data after using grid. Note the npindex
            is applied after applying the mask_by.
        transform_func: data transform_func after applying npindex. That's
            could be useful in case to select specific time range and
            then make time sum or mean.
        prefix: string used to prefix before the variable names used to make
            further distinction. In case of input varlist is a single-element
            list, the prefix will be used directly as the tag name in the
            output Mdata.
        '''
        if md is None:
            md=Pdata.Mdata()

        ydic = self.Add_Vars_to_Dict_Grid(varlist,grid=grid,mask_by=mask_by,
                                          pftsum=pftsum,npindex=npindex,
                                          transform_func=transform_func,
                                          prefix=prefix)
        if len(ydic) == 1 and prefix is not None:
            data = ydic[ydic.keys()[0]].copy()
            del ydic[ydic.keys()[0]]
            ydic[prefix] = data

        md.add_entry_array_bydic(ydic)
        (sublat,sublon) = self.Get_latlon_by_Grid(grid)
        md.add_attr_by_tag(lat=dict.fromkeys(ydic.keys(),sublat),
                           lon=dict.fromkeys(ydic.keys(),sublon))
        return md


    def Add_Single_Var_Mdata(self,varname=None,forcedata=None,
                             taglist=None,md=None,
                             pftsum=False,spa=None,npindex_list=None):
        '''
        Add single var to Mdata to explore the difference among
            different dimensions.

        TODO: allow default taglist and add a new key axis to allow select
        the dimension rather than only the first dimension.
        Parameters:
        -----------
        varname: the final data used for 'varname' should be of dim
            (tag_length,lat,lon),tag_length could be equal to dimension
            length.
        taglist: could be dimension name or other name.
        npindex_list: used to select only few dimensions for mapping.
            when npindex_list is None, the final array for varname must be
            of 3dimensions, the len(taglist) must be equal to the length
            of first dimension of the array.
        '''
        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum,spa=spa)
        if forcedata is None:
            maparray = final_ncdata.__dict__[varname]
        else:
            maparray = forcedata

        if md is None:
            md=Pdata.Mdata()
        if npindex_list is None:
            if taglist is None:
                taglist = ['Tag '+str(i) for i in range(1,maparray.shape[0]+1)]
            else:
                if len(taglist) != maparray.shape[0]:

                    raise ValueError('''the length of tgalist and first
                        dimension of maparray not equal!''')
                else:
                    pass
            for tag,array in zip(taglist,maparray):
                md.add_array_lat_lon(tag=tag,data=array,
                                     lat=self.lat,lon=self.lon)
        else:
            if taglist is None:
                taglist = ['Tag '+str(i) for i in range(1,len(npindex_list)+1)]
            else:
                if len(taglist) != len(npindex_list):
                    raise ValueError('''the length of tgalist not equal
                        to npindex_list''')
                else:
                    pass
            for tag,nindex in zip(taglist,npindex_list):
                md.add_array_lat_lon(tag=tag,data=maparray[nindex],
                                     lat=self.lat,lon=self.lon)
        return md


    def _find_mask(self):
        """
        Find the land/seak mask if there is any.
        """
        geoshape = (len(self.lat),len(self.lon))
        for varname in self.varlist:
            vardata = self.d1.__dict__[varname]
            if vardata.shape[-2:] == geoshape:
                if np.ma.isMA(vardata):
                    return vardata.mask
                else:
                    return None
            else:
                raise ValueError("""No variable found with corresponidng lat/lon
                                  dimension.""")


    def _generate_dic_region_mask_by_dataframe(self,dataframe):
        """
        Generate the dictionary of region mask to be used in break_region from
        a table of region nameas and their extents.

        Parameters:
        -----------
        1. the dataframe should have "lat1,lat2,lon1,lon2,region" as column
            names.
        """
        for name in ['lat1','lat2','lon1','lon2','region']:
            if name not in dataframe.columns:
                raise ValueError("{0} not a column name of dataframe").format(name)
        dic_region_mask = OrderedDict()
        for i,row in dataframe.iterrows():
            region_mask = np.ones((len(self.lat),len(self.lon)),dtype=bool)
            regname = row['region']
            vlat1,vlon1,vlat2,vlon2 = row['lat1'],row['lon1'],row['lat2'],row['lon2']
            (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = \
                self.find_index_by_vertex((vlat1,vlat2),(vlon1,vlon2))
            region_mask[lat_index_min:lat_index_max+1, lon_index_min:lon_index_max+1] = False
            dic_region_mask[regname] = region_mask
        return dic_region_mask

    def _generate_dic_region_mask_by_single_array(self,separation_array,
                                                  regdict=None):
        """
        Generate the dictionary of region mask to be used in break_region from
        a single array.

        Parameters:
        -----------
        separation_array: a 2D array containing differnet values representing
            different regions.
        regdict: a dictionay used to change the keys in the output dict with
            the values of `regdict`. Note the keys in regdict must be contained
            in the unique values of `separation_array`.
        """

        unique_array = np.unique(separation_array)
        if np.ma.isMA(unique_array):
            unique_array = unique_array.compressed()
        else:
            pass
        unique_array = unique_array.astype(int)
        #we need to avoid the duplicates after moldering the type into int.
        if len(np.unique(unique_array)) < len(unique_array):
            raise ValueError("""
            Input separation_array have duplicate values after being
            moldered into int type.
            """)
        else:
            dic_region_mask = OrderedDict()
            for reg_id in unique_array:
                reg_valid = np.ma.masked_not_equal(separation_array,reg_id)
                dic_region_mask[reg_id] = reg_valid.mask
        if regdict is not None:
            dic_region_mask_new = pb.Dic_replace_key_by_dict(dic_region_mask,regdict)
            return dic_region_mask_new
        else:
            return dic_region_mask

    def break_by_region(self,varname=None,separation=None,
                        forcedata=None,
                        pyfunc=None,dimred_func=None,
                        pftsum=False,regdict=None):
        """
        Break the concerned variables into regional sum or avg or extracted array by specifying the separation_array.

        Parameters:
        -----------
        varname: string type.
        separation:
            1. In case of an array:
                The array that's used to separate different regions.
                the np.unique(separation) will be used as the keys for
                the dictionary which will be returned by the function. If
                separation_array is a masked array, the final masked value
                in the np.unique result will be dropped. If regdict is provided,
                then the new keys in regdict will be used, rather than the unique
                values in separation.
            2. In case of a dataframe:
                The dataframe should have "lat1,lat2,lon1,lon2,region" as column
                names, and the method _generate_dic_region_mask_by_dataframe is used
                to produce the dictionary of region mask.
            3. In case of a dictionary of region mask, it will be directly used.
        forcedata: used to force input data.
        pyfunc:
            1. In case of function, used to change the regional array data;
            2. In case of a scalar, will be directly multiplied.
        dimred_func: functions that used to reduce the dimensions of regional
            array data, as long as it could be applied on numpy ndarray.
            eg., np.sum will get the sum of the regional array. If None,
            regional array will be returned.
            == Note == to be depreciated in the future.
        pftsum: if True, the varname data from the pftsum will be used.
        regdict: a dictionay used to change the keys in the output dict with
            the values of `regdict`. Note the keys in regdict must be contained
            in the unique values of `separation` if it's provided as a single
            2D array.

        Notes:
        ------
        1. varname could be any dimension as long as the last two dimensions
            are the same as separation_array.
        2. pyfunc and dimred_func work for np.ma functions
        3. dimred_func is not a general disign will be removed later.
            A general use of pyfunc is rather preferred. For example,
            to have a region sum with unit change, use:
            pyfunc = lambda x: np.ma.sum(np.ma.sum(x,axis=-1),axis=-1)*30.
        4. within the code, the array for each region is eaxtracted by masking
            the values outside the region so that it has the same dimension
            before applying any function operation. This allow a wide range of
            operation on the regional data by proper definition of pyfunc, as
            long as the dimensions accord.
            for instance:
            - to have the array first multiplied by the area array:
            pyfunc = lambda x: np.ma.sum(x*area,axis=(1,2))*30

        Returns:
        --------
        region_dic:
            A dictionary with the region ID/names as keys and the region
            arrays or mean or sum as key values.

        """
        print "dimred_func will be removed in the future!"

        #Get data: treat varname and forcedata
        if forcedata is not None:
            vardata = forcedata
        else:
            if pftsum == True:
                vardata = self.pftsum.__dict__[varname]
            else:
                vardata = self.d1.__dict__[varname]


        #treat separation object
        if isinstance(separation,np.ndarray):
            dic_region_mask = \
                self._generate_dic_region_mask_by_single_array(separation,
                                                               regdict=regdict)
        elif isinstance(separation,pa.DataFrame):
            dic_region_mask = \
                self._generate_dic_region_mask_by_dataframe(separation)
        elif isinstance(separation,dict):
            dic_region_mask = separation
        else:
            raise TypeError("separation could only be np.ndarray, dataframe or dict")

        #Get the final output dic and apply fuctions
        regdic=OrderedDict()
        for name,region_mask in dic_region_mask.items():
            annual_reg = mathex.ndarray_apply_mask(vardata,mask=region_mask)
            if np.any(np.isnan(annual_reg)) or np.any(np.isinf(annual_reg)):
                print "Warning! nan or inf values have been masked for variable {0} reg_id {1}".format(varname,reg_id)
                annual_reg = np.ma.masked_invalid(annual_reg)
            if pyfunc is not None:
                if callable(pyfunc):
                    data=pyfunc(annual_reg)
                else:
                    data=annual_reg*pyfunc
            else:
                data = annual_reg

            if dimred_func is None:
                regdic[name] = data
            elif callable(dimred_func):
                if data.ndim >= 2:
                    regdic[name] = dimred_func(dimred_func(data,axis=-1),axis=-1)
                else:
                    raise ValueError("strange the dimension of data is less than 2!")
            else:
                raise TypeError("dimred_func must be callable")


        return regdic

    def Plot_Vars(self,ax=None,varlist=None,npindex=np.s_[:],
                  unit=True,pftsum=False,spa=None,
                  split=False,splitkw={},
                  legkw={},
                  **plotkw):
        '''
        Plot for varlist as pd.
        '''

        pd=self.Add_Vars_to_Pdata(varlist=varlist,npindex=npindex,unit=unit,
                                  pftsum=pftsum,spa=spa)
        if not split:
            pd.plot(axes=ax,**plotkw)
            pd.set_legend_all(taglab=True,**legkw)
        else:
            pd.plot_split_axes(self,plotkw=plotkw,**splitkw)
        return pd


    def Add_Vars_to_Dict_Grid(self,varlist,grid=None,
                              pftsum=False,mask_by=None,
                              npindex=np.s_[:],
                              transform_func=None,prefix=None):
        """
        Add vars to dictionary.

        Parameters:
        -----------
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        mask_by:
            1. in case of a boolean numpy array, the mask will be directly
               apply on the retrieved array by using
               mathex.ndarray_mask_smart_apply;
            2. in case of a tuple like (varname,{'lb':2000,'ub':5000}), the mask
               will first be generated using mathex.ndarray_mask_by_threshold,
               followed by mathex.ndarray_mask_smart_apply.
            3. in case of a function, it will be directly applied on the
               retrieved ndarray, eg., lambda x:np.ma.masked_invalid(x).
            4. Following the point above, the `mask_by` could be extended to
               any function for a general purpose, eg.,
               mask_by=lambda x:np.ma.mean(x,axis=0), used to perform
               proper operation on data.

        npindex: further index the data after using grid. Note the npindex
            is applied after applying the mask_by.
        transform_func: data transform_func after applying npindex. That's
            could be useful in case to select specific time range and
            then make time sum or mean. This repeats some function of point4
            for the general purpose of mask_by but it's only applied after
            npindex.
        prefix: string used to prefix before the variable names in the output
            dictionary.

        Notes:
        ------
        The applying sequence for different methods:
        grid, mask_by,npindex,tansform_func
        """
        #The final_ncdata is used only to retrieve variable values used
        #in the mask_by function.
        final_ncdata = self._get_final_ncdata_by_flag(pftsum=pftsum)
        final_dict = OrderedDict()


        #define some special mask_by functions
        if mask_by == 'invalid':
            mask_by = lambda arr:np.ma.masked_invalid(arr)

        def treat_data_by_mask(data,mask_by):
            if mask_by is None:
                return data
            elif isinstance(mask_by,np.ndarray):
                return mathex.ndarray_mask_smart_apply(data,mask_by)

            elif isinstance(mask_by,tuple):
                varname = mask_by[0]
                map_threshold = mask_by[1]
                mask_vardata = \
                    mathex.ndarray_mask_by_threshold(
                        final_ncdata.__dict__[varname],map_threshold)
                return mathex.ndarray_mask_smart_apply(data,mask_vardata.mask)

            elif callable(mask_by):
                return mask_by(data)
            else:
                raise TypeError("wrong mask_by type")

        def apply_transform_func(data,transform_func):
            if transform_func is None:
                return data
            elif callable(transform_func):
                return transform_func(data)
            else:
                raise TypeError("transform_func must be function")

        for var in varlist:
            data = self._get_var_by_grid(var,grid=grid,
                                         pftsum=pftsum)
            data = treat_data_by_mask(data,mask_by)

            if prefix is None:
                final_dict[var] = apply_transform_func(data[npindex],transform_func)
            else:
                final_dict[prefix+var] = apply_transform_func(data[npindex],transform_func)

        return final_dict


    @append_doc_of(Add_Vars_to_Dict_Grid)
    def Add_Vars_to_Dict_by_RegSum(self,varlist,mode='sum',pftsum=False,
                                   mask_by=None,
                                   transform_func=None,
                                   grid=None,
                                   npindex=np.s_[:],
                                   area_weight=False,
                                   unit_func=None):
        """
        Add vars to dictionary, but with some spatial operation.
        This is a further wrapper of gnc.Ncdata.Add_Vars_to_Dict_Grid

        Parameters:
        -----------
        == Extra ones for this function:
        mode: 'sum' for regional sum;'mean' for regional mean. When area_weight 
            is False, simple algebra sum or mean is done; otherwise area will be
            included when doing sum or mean.

        area_weight: boolean or string or ndarray, default to boolean False.
            False: simple sum or mean operation is done, assuming equal weight
                of each pixel.
            True: default ORCHIDEE 'Areas' variable used as `area` variable name.
            string: used to denote the variable name in the nc file which will
                be used as the weighting factor. Thus it does not necessarily
                have to be area.
            ndarray: will be directly used in the calculation.

        unit_func: functions used to scale the data due to unit reason.

        Notes:
        ------
        Applying sequence:
            grid, mask_by,npindex,tansform_funcm,mode, area_weight, unit_func

            first apply the sequential methods as in
            gnc.Ncdata.Add_Vars_to_Dict_Grid:
            grid, mask_by,npindex,tansform_func
            then by sequence of
            mode, area_weight, unit_func

        See also
        --------
        gnc.Ncdata.Add_Vars_to_Dict_Grid

        Appended doc of Add_Vars_to_Dict_Grid:
        --------------------------------------
        """
        dic = self.Add_Vars_to_Dict_Grid(varlist,grid=grid,pftsum=pftsum,
                                         npindex=npindex,
                                         mask_by=mask_by,
                                         transform_func=transform_func)

        def get_area_array(area_weight,shape):
            """
            """
            if isinstance(area_weight,str):
                area = self.Add_Vars_to_Dict_Grid([area_weight],grid=grid).values()[0]
            elif isinstance(area_weight,bool):
                if area_weight == True:
                    area = self.Add_Vars_to_Dict_Grid(['Areas'],grid=grid).values()[0]
                else:
                    area = np.ones(shape)  #we use one to apply simple avearge or sum
            elif isinstance(area_weight,np.ndarray):
                area = area_weight.copy()
            else:
                raise ValueError("area_weight could only be boolean or str type.")

            return area



        def get_func_by_mode(mode,area):
            func_sum = lambda x:np.ma.sum(np.ma.sum(x,axis=-1),axis=-1)
            if mode == 'sum':
                func = func_sum

            elif mode == 'mean':
                if area_weight is False:
                    func = lambda x:np.ma.mean(np.ma.mean(x,axis=-1),axis=-1)
                # this case does not apply in above because if variables
                # are masked data, func_sum(np.ones(shape)) will lose
                # the effect of mask. But this issue does not happen for
                # the case of simple sum, because the operation
                # data_area = data*area will include the effect of mask.
                else:
                    func = lambda x:func_sum(x)/func_sum(area)
            else:
                raise ValueError("mode could only be sum/mean")
            return func

        for key,data in dic.items():
            shape = (data.shape[-2],data.shape[-1])
            area = get_area_array(area_weight,shape)

            # here we calculate data already considering the 'area'
            data_area = data*area
            mode_func = get_func_by_mode(mode,area)
            data_new = mode_func(data_area)

            if unit_func is None:
                final_data = data_new
            else:
                final_data = unit_func(data_new)
            dic[key] = final_data

        return dic

    @append_doc_of(Add_Vars_to_Dict_by_RegSum)
    def Add_Vars_to_dataframe(self,varlist,mode='sum',pftsum=False,
                              mask_by=None,
                              transform_func=None,
                              grid=None,
                              npindex=np.s_[:],
                              area_weight=False,
                              unit_func=None,
                              index=None):
        """
        This is a simple wrapper of Add_Vars_to_Dict_by_RegSum

        Parameters:
        -----------
        index: used as index in the dataframe.

        See also
        --------
        gnc.Ncdata.Add_Vars_to_Dict_Grid
        gnc.Ncdata.Add_Vars_to_Dict_by_RegSum

        The doc of Ncdata.Add_Vars_to_Dict_by_RegSum:
        --------------------------------------------
        """
        dic = self.Add_Vars_to_Dict_by_RegSum(varlist,mode=mode,
                                              pftsum=pftsum,
                                              mask_by=mask_by,
                                              area_weight=area_weight,
                                              transform_func=transform_func,
                                              unit_func=unit_func,
                                              npindex=npindex,
                                              grid=grid)
        return pa.DataFrame(dic,index=index)


    def Add_Vars_to_ndarray(self,varlist):
        """
        Extract the values in varlist as a concatenated ndarray.
        """
        datalist = [self.d1.__dict__[var] for var in varlist]
        dt = np.concatenate([data[...,np.newaxis] for data in datalist],axis=-1)
        return dt

    def OrcBio_get_woodmass(self):
        varlist = ['SAP_M_AB','SAP_M_BE','HEART_M_AB','HEART_M_BE']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['WOODMASS'] = self.d1.SAP_M_AB + self.d1.SAP_M_BE + self.d1.HEART_M_AB + self.d1.HEART_M_BE
            self.d1.__dict__['WOODMASS_AB'] = self.d1.SAP_M_AB + self.d1.HEART_M_AB
            self.d1.__dict__['WOODMASS_BE'] = self.d1.SAP_M_BE + self.d1.HEART_M_BE
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['WOODMASS'] = self.d1.SAP_M_AB + self.d1.SAP_M_BE + self.d1.HEART_M_AB + self.d1.HEART_M_BE
            self.d1.__dict__['WOODMASS_AB'] = self.d1.SAP_M_AB + self.d1.HEART_M_AB
            self.d1.__dict__['WOODMASS_BE'] = self.d1.SAP_M_BE + self.d1.HEART_M_BE

        for varname in ['WOODMASS_BE','WOODMASS_AB','WOODMASS']:
            self.d0.__dict__[varname] = self.d1.__dict__[varname]
            self.d0.__dict__[varname].__dict__['dimensions'] = self.d0.__dict__['SAP_M_AB'].dimensions
            self.varlist.append(varname)


    def OrcBio_get_bmab(self):
        """
        Calculate above- and belowground biomass carbon. This follows the litter allocation in ORCHIDEE.
        """
        varlist = ['SAP_M_AB','SAP_M_BE','HEART_M_AB','HEART_M_BE','LEAF_M','ROOT_M','FRUIT_M','RESERVE_M']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['BIOMASS_AB'] = self.d1.SAP_M_AB + self.d1.HEART_M_AB + self.d1.LEAF_M + self.d1.FRUIT_M + self.d1.RESERVE_M
            self.d1.__dict__['BIOMASS_BE'] = self.d1.SAP_M_BE + self.d1.HEART_M_BE + self.d1.ROOT_M
            self.d1.__dict__['BIOMASS'] = self.d1.BIOMASS_AB + self.d1.BIOMASS_BE
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['BIOMASS_AB'] = self.d1.SAP_M_AB + self.d1.HEART_M_AB + self.d1.LEAF_M + self.d1.FRUIT_M + self.d1.RESERVE_M
            self.d1.__dict__['BIOMASS_BE'] = self.d1.SAP_M_BE + self.d1.HEART_M_BE + self.d1.ROOT_M
            self.d1.__dict__['BIOMASS'] = self.d1.BIOMASS_AB + self.d1.BIOMASS_BE

        for varname in ['BIOMASS_BE','BIOMASS_AB','BIOMASS']:
            self.d0.__dict__[varname] = self.d1.__dict__[varname]
            self.d0.__dict__[varname].__dict__['dimensions'] = self.d0.__dict__['LEAF_M'].dimensions
            self.varlist.append(varname)

    def OrcBio_get_litterab(self):
        varlist = ['LITTER_STR_AB','LITTER_STR_BE','LITTER_MET_AB','LITTER_MET_BE']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['LITTER_AB'] = self.d1.LITTER_STR_AB + self.d1.LITTER_MET_AB
            self.d1.__dict__['LITTER_BE'] = self.d1.LITTER_STR_BE + self.d1.LITTER_MET_BE
            self.d1.__dict__['LITTER'] = self.d1.__dict__['LITTER_AB'] + self.d1.__dict__['LITTER_BE']
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['LITTER_AB'] = self.d1.LITTER_STR_AB + self.d1.LITTER_MET_AB
            self.d1.__dict__['LITTER_BE'] = self.d1.LITTER_STR_BE + self.d1.LITTER_MET_BE
            self.d1.__dict__['LITTER'] = self.d1.__dict__['LITTER_AB'] + self.d1.__dict__['LITTER_BE']

        for varname in ['LITTER_BE','LITTER_AB','LITTER']:
            self.d0.__dict__[varname] = self.d1.__dict__[varname]
            self.d0.__dict__[varname].__dict__['dimensions'] = self.d0.__dict__['LITTER_STR_AB'].dimensions
            self.varlist.append(varname)

    def OrcBio_get_autoresp(self):
        varlist = ['MAINT_RESP','GROWTH_RESP']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['AUTO_RESP'] = self.d1.MAINT_RESP + self.d1.GROWTH_RESP
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['AUTO_RESP'] = self.d1.MAINT_RESP + self.d1.GROWTH_RESP

    def OrcBio_get_NBP(self):
        varlist = ['NPP','HET_RESP','CO2_FIRE']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['NBP'] = self.d1.NPP - self.d1.HET_RESP - self.d1.CO2_FIRE
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['NBP'] = self.d1.NPP - self.d1.HET_RESP - self.d1.CO2_FIRE

    def OrcBio_get_NEP(self):
        varlist = ['NPP','HET_RESP']
        if self.largefile:
            self.retrieve_variables(varlist)
            self.d1.__dict__['NEP'] = self.d1.NPP - self.d1.HET_RESP
            self.remove_variables(varlist)
        else:
            self.d1.__dict__['NEP'] = self.d1.NPP - self.d1.HET_RESP

    def OrcDebug_get_latlon_index(self,(vlat,vlon),forcedata=None):
        """
        Get the index of (vlat,vlon) used for model debugging.

        Notes:
        ------
        1. In ORCHIDEE, when spaital data are re-arranged into 1-dimensional
            array using the 'C' (row-major) convention, which is the same as
            numpy np.nonzero.
        """
        if forcedata is None:
            try:
                forcedata = self.d1.Areas
            except KeyError:
                raise ValueError("Areas not found, forcedata must be provided.")
        else:
            if np.ndim(forcedata) != 2:
                raise ValueError("Make sure forcedata has only 2 dims.")

        latpos,lonpos = self.find_index_by_point(vlat,vlon)
        indlat,indlon = np.nonzero(~forcedata.mask)
        list_pos = [(indlat[num],indlon[num]) for num in range(len(indlat))]
        return list_pos.index((latpos,lonpos))+1 #note the index in ORC
                                                 #fortran code is 1-based.

class NestNcdata(object):
    """
    A wrap of dictionary of Ncdata
    """
    def __init__(self,nddic):
        self.labels = nddic.keys()
        self.data = nddic
        for lab in self.labels:
            self.__dict__[lab] = nddic[lab]

    def __repr__(self):
        filenames = [nd.filename for nd in self.data.values()]
        frp = ['{0}: {1}'.format(lab,fname) for (lab,fname) in zip(self.labels,filenames)]
        return '\n'.join(frp)

    @classmethod
    def from_ncfiles(cls,labels=None,ncfiles=None,
                     latlon_dim_name=None,
                     largefile=False,multifile=False,
                     replace_nan=False,print_timecheck=True):
        """
        Create NestNcdata from ncfiles.
        """
        dic = OrderedDict()
        for lab,f in zip(labels,ncfiles):
            dic[lab] = Ncdata(f,latlon_dim_name=latlon_dim_name,
                              largefile=largefile,multifile=multifile,
                              replace_nan=replace_nan,
                              print_timecheck=print_timecheck)

        return NestNcdata(dic)


    def items(self):
        return self.data.items()

    def call(self,method,*args,**kwargs):
        dic = OrderedDict()
        for lab,nd in self.items():
            dic[lab] = nd.__dict__[method](*args,**kwargs)

    def get_pftsum(self,varlist=None,area_include=True,veget_npindex=np.s_[:],
                   print_info=False,name_vegetmax=None):
        """
        Get PFT (VEGET_MAX) weighted average of variables.
        Parameters:
        -----------
        varlist: limit varlist scope
        veget_npindex: 
            1. could be used to restrict for example the PFT
            weighted average only among natural PFTs by setting
            veget_npindex=np.s_[:,0:11,:,:]. It will be used to slice
            VEGET_MAX or vegetfrac or maxvegetfrac variable.
            2. could also be used to slice only for some subgrid
            of the whole grid, eg., veget_npindex=np.s_[...,140:300,140:290].

        Note:
        -----
        1. This function is very flexible, it can handle cases of spatial dataset
            with single or multiple time steps, or a single point file with
            multiple time steps. This is confirmed again on 24/02/2016
        """
        for lab,nd in self.items():
            nd.get_pftsum(varlist=varlist,area_include=area_include,
                         veget_npindex=veget_npindex,
                         print_info=print_info,name_vegetmax=name_vegetmax)

    def apply_func(self,varlist,func):
        """
        Apply function over varlist, return a dictionary when varlist is of
            type string, otherwise a nested dictionary will be returned.
        """
        if isinstance(varlist,str):
            dic = OrderedDict()
            for tag,nd in self.items():
                dic[tag] = func(nd.d1.__dict__[varlist])
            return dic
        elif isinstance(varlist,list):
            dic = OrderedDict()
            for varname in varlist:
                dic[varname] = self.apply_func(varname,func)
            return dic


def nc_get_var_value_grid(ncfile,varname,(vlat1,vlat2),(vlon1,vlon2),
                          pftsum=False):
    '''
    simple way to quickly retrieve grid value
    '''
    d = Ncdata(ncfile)
    return d.Get_GridValue(varname,(vlat1,vlat2),(vlon1,vlon2),
                    pftsum=pftsum)

def nc_get_var_value_point(ncfile,varname,(vlat,vlon)):
    '''
    simple way to quickly retrieve grid value
    '''
    d = Ncdata(ncfile)
    return d.Get_PointValue(self,varname,(vlat,vlon))



def nc_pftsum_trans(infile, outfile, length_one_time_dim=False):
    """
    Apply the pftsum transformation for infile and write to outfile. set length_one_time_dim to True to keep the time dimension whose lenght is 1 in the new nc file.
    """
    d = Ncdata(infile)
    d.get_pftsum()
    outfile_ob = NcWrite(outfile)
    if d.unlimited_dimlen == 1:
        if length_one_time_dim:
            outfile_ob.add_3dim_time_lat_lon(1, d.lat, d.lon)
            dim_flag = 2
        else:
            outfile_ob.add_2dim_lat_lon(d.lat, d.lon)
            dim_flag = 1
    else:
        outfile_ob.add_3dim_time_lat_lon(d.unlimited_dimlen, d.lat, d.lon)
        dim_flag = 3

    for varname in d.pftsum.__dict__.keys():
        if varname != 'Areas':
            if dim_flag == 1:
                outfile_ob.add_var_2dim_lat_lon('PFTSUM_'+varname, d.pftsum.__dict__[varname], attr_copy_from=d.d0.__dict__[varname])
            else:
                outfile_ob.add_var_3dim_time_lat_lon('PFTSUM_'+varname, d.pftsum.__dict__[varname], attr_copy_from=d.d0.__dict__[varname])

    if 'Areas' in d.pftsum.__dict__.keys():
        outfile_ob.add_var_2dim_lat_lon('Areas', d.pftsum.__dict__['Areas'], attr_copy_from=d.d0.__dict__['Areas'])

    outfile_ob.close()


def nc_subgrid(infile, outfile, subgrid=[(None,None),(None,None)]):
    """
    Purpose : subgrid infile by vertex. Currently can only handle as much as 4 dimensions (time,PFT,lat,lon)
    Definition : nc_subgrid(infile, outfile, subgrid=[(vlat1,vlat2),(vlon1,vlon2)]):
    Arguments:
        subgrid : [(vlat1,vlat2),(vlon1,vlon2)]; latitude before latitude
    Test : Test has been done for 2dim/3dim/4dim variables. see test_nc_subgrid
    """
    print "subgrid for file --{0}-- begins".format(infile)
    d = Ncdata(infile)
    (lon_index_min, lon_index_max, lat_index_min, lat_index_max) = find_index_by_vertex(d.lonvar, d.latvar, subgrid[1], subgrid[0])
    outfile_ob = NcWrite(outfile)
    outfile_ob.add_dim([d.latdim_name, d.latvar_name, d.latvar_name, 'f4', d.latvar[lat_index_min:lat_index_max+1], 'None', False])
    outfile_ob.add_dim([d.londim_name, d.lonvar_name, d.lonvar_name, 'f4', d.lonvar[lon_index_min:lon_index_max+1], 'None', False])
    if hasattr(d,'unlimited_dimname'):
        outfile_ob.add_dim([d.unlimited_dimname, d.timevar_name, d.timevar_name, 'i4', d.timevar, 'None', True])
    if 'PFT' in d.dimensions:
        outfile_ob.add_dim_pft()

    for varname in pb.StringListAnotB(d.d1.__dict__.keys(), d.dimvar_name_list):
        ordata = d.d0.__dict__[varname]
        assert ordata.ndim>=2,"variable '{0}' has ndim <2".format(varname)
        outfile_ob.add_var(varinfo_value=[varname, ordata.dimensions, ordata.dtype, ordata[:][...,lat_index_min:lat_index_max+1,lon_index_min:lon_index_max+1]], attr_copy_from=ordata)
        print "subgrid for --{0}-- done".format(varname)

    glob_attr_dic = d.global_attributes
    #add time stamp for this operation
    if 'history' in glob_attr_dic:
        glob_attr_dic['history'] = 'file created at ' + str(datetime.datetime.today()) + 'by subgriding file --' + infile + '--\n' + glob_attr_dic['history']
    else:
        glob_attr_dic['history'] = 'file created at ' + str(datetime.datetime.today()) + 'by subgriding file --' + infile

    #write the global attributes
    outfile_ob.add_global_attributes(glob_attr_dic)
    outfile_ob.close()
    print "subgrid for file --{0}-- done".format(outfile)

def nc_subgrid_csv(infile,csv_file=None,bound_name = ['south_bound','north_bound','west_bound','east_bound']):
    """
    Separate a (global) nc file by regions as specified in the csv_file.

    Parameters:
    -----------
    csv_file: the title should be region names, the first column should be the four bound names. see example in base_data/region.csv
    bound_name: the names that are used to distinguish the four bounds: (vlat1,vlat2),(vlon1,vlon2), must be in sequence of south,north,west,east

    Outputs:
    --------
    Several regional nc files, with file names indicating each region.

    Test
    ----
    nc_subgrid_csv and nc_merge_files are tested against each other in the gnc_test.py.
    """
    region=pa.read_csv(csv_file,index_col=0)
    region_dic=region.to_dict()
    for name in region_dic.keys():
        data=region_dic[name]
        (vlat1,vlat2) = (data[bound_name[0]], data[bound_name[1]])
        (vlon1,vlon2) = (data[bound_name[2]], data[bound_name[3]])
        nc_subgrid(infile,infile[:-3]+'_'+name+'.nc',[(vlat1,vlat2),(vlon1,vlon2)])

def compare_nc_file(file1,file2,varlist,npindex=np.s_[:]):
    """
    compare if the input netcdf files have the same value for the same variable.

    Parameters:
    -----------
    npindex: it could be a 2-lenght tuple (which will be broadcast to number of variables) or a n-length list with each element as a 2-length tuple
    """
    d1 = Ncdata(file1)
    d2 = Ncdata(file2)
    reslist = []
    if isinstance(npindex,list):
        if len(npindex) != len(varlist):
            raise ValueError("npindex is provided as list but length not equal to varlist!")
        else:
            npindex_final = npindex
    elif isinstance(npindex,tuple):
        if len(npindex) != 2:
            raise ValueError("npindex is provided as tuple and lenght is not 2!")
        else:
            npindex_final = [npindex] * len(varlist)
    elif npindex == np.s_[:]:
        npindex_final = [(npindex,npindex)] * len(varlist)
    else:
        raise ValueError("unknown npindex value!")

    for varname,comindex in zip(varlist,npindex_final):
        if np.ma.allclose(d1.d1.__dict__[varname][comindex[0]],d2.d1.__dict__[varname][comindex[1]]):
            print "variable --{0}-- equal".format(varname)
            reslist.append(True)
        else:
            print "variable --{0}-- NOT equal".format(varname)
            reslist.append(False)

def arithmetic_ncfiles_var(filelist,varlist,func,npindex=np.s_[:]):
    """
    Get arithetic operation form variables in nc files by a quick way. This is to avoid the tedious way to try to get some simple arithmetic operation on
        seleted variables from some nc files.

    Parameters:
    -----------
    filelist: nc file list.
    varlist: variable name list, when the variable names for all the nc files are the same, it will be broadcast.
    func: function that used the corresponding ndarrays as arguments. np.ma functions are tested to work well.
    npindex: it could be given by a list to specify the indexing of ndarrays before applying the function.
    """

    if isinstance(varlist,str):
        varlist = [varlist]*len(filelist)

    if isinstance(npindex,list):
        if len(npindex) != len(varlist):
            raise ValueError("npindex is provided as list but length not equal to varlist!")
        else:
            npindex_final = npindex
    elif npindex == np.s_[:]:
        npindex_final = [npindex] * len(varlist)
    else:
        raise ValueError("unknown npindex value!")

    data_list = [Ncdata(filename) for filename in filelist]
    ndarray_list = [d.d1.__dict__[varname][comindex] for d,varname,comindex in zip(data_list,varlist,npindex_final)]
    return func(*ndarray_list)

def nc_Add_Mdata_by_DictFilenameVarlist(dict_filename_varlist,npindex=np.s_[:]):
    """
    Simple wrapper of Pdata.Mdata

    Notes:
    ------
    1. npindex applies to all filename and varname, thus the usage is
        very limited.

    2. dict_filename_varlist: a dictionary of (filename,varlist) pairs.
    """
    if pb.List_Duplicate_Check(pb.iteflat(dict_filename_varlist.values())):
        raise ValueError("Duplicate varnames found")
        sys.exit()
    md = Pdata.Mdata()
    for filename in dict_filename_varlist.keys():
        d = Ncdata(filename)
        for varname in dict_filename_varlist[filename]:
            tag = varname
            md.add_tag(tag)
            md.add_lat_lon(tag=tag,lat=d.latvar,lon=d.lonvar)
            md.add_array(d.d1.__dict__[varname][npindex],tag)
            if hasattr(d.d0.__dict__[varname],'units'):
                unit = d.d0.__dict__[varname].getncattr('units')
            elif hasattr(d.d0.__dict__[varname],'unit'):
                unit = d.d0.__dict__[varname].getncattr('unit')
            else:
                unit = None
            md.add_attr_by_tag(unit={tag:unit})
    return md

def nc_add_Mdata_mulitfile(filelist,taglist,varlist,npindex=np.s_[:]):
    '''
    Simple wrapper of Pdata.Mdata.add_entry_share_latlon_bydic.

    Parameters:
    -----------
    varlist: varlist length should be equal to filelist. Broadcast of list
        when single string value is provided.
    Notes:
    ------
    All the input nc files should have the same geo_limit
    '''
    if isinstance(varlist,list):
        pass
    elif isinstance(varlist,str):
        varlist = [varlist]*len(filelist)

    ydic = OrderedDict()
    d0 = Ncdata(filelist[0])
    for filename,tag,varname in zip(filelist,taglist,varlist):
        dt = Ncdata(filename)
        ydic[tag] = dt.d1.__dict__[varname][npindex]
    md = Pdata.Mdata()
    md.add_entry_share_latlon_bydic(ydic, lat=d0.lat, lon=d0.lon)
    return md

def nc_add_Mdata_list_ncdata(ncdatalist,taglist,varlist,
                             pftsum=False,npindex=np.s_[:],
                             pyfunc=None):
    '''
    Simple wrapper of Pdata.Mdata.add_entry_share_latlon_bydic.

    Parameters:
    -----------
    varlist: varlist length should be equal to filelist. Broadcast of list
        when single string value is provided.
    npindex: used to slice data
    Notes:
    ------
    All the input nc files should have the same geo_limit
    '''
    if isinstance(varlist,list):
        pass
    elif isinstance(varlist,str):
        varlist = [varlist]*len(ncdatalist)

    ydic = OrderedDict()
    for ncdata,tag,varname in zip(ncdatalist,taglist,varlist):
        if pftsum == True:
            dt = ncdata.pftsum
        else:
            dt = ncdata.d1
        ydic[tag] = mathex.apply_func(dt.__dict__[varname][npindex],pyfunc=pyfunc)
    md = Pdata.Mdata()
    md.add_entry_share_latlon_bydic(ydic, lat=ncdata.lat, lon=ncdata.lon)
    return md

def ncfile_get_varlist(filename):
    """
    This is a loose application of gnc.Ncdata.list_var()

    Arguments:
    ----------
    filename: when it's a single string, return varlist of the file;
        when it's a list of file names, return a dictionary of
        (filename,filelist).
    """
    if isinstance(filename,str):
        d = Ncdata(filename)
        return d.list_var(keyword='excludedim')
    elif isinstance(filename,list):
        return dict([(f,ncfile_get_varlist(f)) for f in filename])
        #tlist = [(f,ncfile_get_varlist(f)) for f in filename]
        #return dict(tlist)



def nc_add_Pdata_mulitfile(filelist,taglist,varlist,npindex=np.s_[:],
                           pyfunc=None):
    '''
    Simple wrapper of Pdata.Pdata.add_entry_sharex_noerror_by_dic

    Parameters:
    -----------
    varlist: varlist length should be equal to filelist. Broadcast of list
        when single string value is provided.
    Notes:
    ------
    1. all the entry use default indexed x
    '''
    if isinstance(varlist,list):
        pass
    elif isinstance(varlist,str):
        varlist = [varlist]*len(filelist)

    ydic = OrderedDict()
    d0 = Ncdata(filelist[0])
    for filename,tag,varname in zip(filelist,taglist,varlist):
        dt = Ncdata(filename)
        data = dt.d1.__dict__[varname][npindex]
        ydic[tag] = mathex.apply_func(data,pyfunc=pyfunc)
    pd = Pdata.Pdata()
    pd.add_entry_sharex_noerror_by_dic(ydic)
    return pd

def nc_add_Pdata_List_Ncdata(Ncdata_list,taglist,varname,spa=None,
                             npindex=np.s_[:],pyfunc=None):
    """
    Add a common varname from a list of Ncdata object to the Pdata.

    Parameters:
    ----------
    1. spa: 'spasum' or 'spamean', used to extract data after making
        the "get_spa()" operation of the Ncdata objects.
    2. npindex: used to directly slice the gnc.Ncdata.d1 attributes, which
        is the default when spa is None
    3. pyfunc: applied at the last stage on the data that are retrieved.
    """
    pd = Pdata.Pdata()
    for tag,ncdata in zip(taglist,Ncdata_list):
        if spa is None:
            data = ncdata.d1.__dict__[varname][npindex]

        else:
            data = ncdata.__dict__[spa].__dict__[varname]
        pd.add_entry_noerror(x=None,y=mathex.apply_func(data,pyfunc=pyfunc),
                                 tag=tag)

    return pd




@append_doc_of(_set_default_ncfile_for_write)
def nc_creat_ncfile_by_ncfiles(outfile,varname,input_file_list,input_varlist,
                               npindex=np.s_[:],
                               pyfunc=None,
                               Ncdata_latlon_dim_name=None,
                               attr_kwargs={},
                               **kwargs):
    '''
    Write ncfile by calculating from variables of other nc file.

    Parameters:
    -----------
    varname: the varname for writting into nc file.
    input_file_list: input file list.
    input_varlist: varname list from input files, broadcast when necessary.
    npindex: could be a list otherwise broadcast.
    Ncdata_latlon_dim_name: used in Ncdata function
    attr_kwargs: used in Ncwrite.add_var

    Notes:
    ------
    This is intended to write only ONE variable for the nc file.

    '''
    ncfile = NcWrite(outfile)
    _set_default_ncfile_for_write(ncfile,**kwargs)
    if isinstance(input_varlist,str):
        input_varlist = [input_varlist]*len(input_file_list)
    if not isinstance(npindex,list):
        npindex = [npindex]*len(input_file_list)

    data_list = [Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name) for
               filename in input_file_list]
    ndarray_list = [d.d1.__dict__[var][comindex] for d,var,comindex
                    in zip(data_list,input_varlist,npindex)]
    ndim = len(ncfile.dimensions)
    data = pyfunc(ndarray_list)
    ncfile.add_var_smart_ndim(varname,ndim,data,**attr_kwargs)
    ncfile.close()

@append_doc_of(_set_default_ncfile_for_write)
def nc_merge_ncfiles(outfile,input_file_list,
                               Ncdata_latlon_dim_name=None,
                               attr_kwargs={},
                               **kwargs):
    '''
    Merge all the variables in the input_file_list as a single file. The
        duplicate vriables that come later will be discast.

    Parameters:
    -----------
    input_file_list: input file list.
    Ncdata_latlon_dim_name: used in Ncdata function
    attr_kwargs: used in Ncwrite.add_var

    Notes:
    ------
    This is intended to write only ONE variable for the nc file.

    '''
    #get the default lat,lon,time info from the first input file
    subdata_first = Ncdata(input_file_list[0],
                           latlon_dim_name=Ncdata_latlon_dim_name)
    if 'PFT' in subdata_first.dimensions:
        kwargs['pft'] = True
    kwargs['time_length'] = subdata_first.unlimited_dimlen
    kwargs['latvar'] = subdata_first.lat
    kwargs['lonvar'] = subdata_first.lon

    ncfile = NcWrite(outfile)
    _set_default_ncfile_for_write(ncfile,**kwargs)
    ndim = len(ncfile.dimensions)
    varlist_inside = []

    for filename in input_file_list:
        d = Ncdata(filename,latlon_dim_name=Ncdata_latlon_dim_name)
        varlist = pb.StringListAnotB(d.d1.__dict__.keys(),
                                     d.dimvar_name_list)
        for varname in varlist:
            if varname in varlist_inside:
            #if varname in varlist_inside or varname == 'VEGET_MAX':
            #if varname in varlist_inside or varname == 'Areas' or varname == 'VEGET_MAX':
                print """var --{0}-- in file --{1}-- is discast due to
                    duplication""".format(varname,filename)
            else:
                if varname == 'VEGET_MAX':
                    pftdim = True
                else:
                    pftdim = False
                data = d.d1.__dict__[varname]
                ncfile.add_var_smart_ndim(varname,ndim,data,pftdim=pftdim,
                    attr_copy_from=d.d0.__dict__[varname],**attr_kwargs)
                varlist_inside.append(varname)
    ncfile.add_history_attr(histtxt='Created from files:'+'\n'.join(input_file_list))
    ncfile.close()


def test_nc_subgrid():
    #test 3dim nc data
    nc_subgrid('testdata/grid_test_3dim.nc','testdata/grid_test_3dim_sub.nc',subgrid=[(-35,25),(-50,20)])
    d0 = Ncdata('testdata/grid_test_3dim.nc')
    d1 = Ncdata('testdata/grid_test_3dim_sub.nc')
    lon = np.arange(-49.75,19.8,0.5)
    lat = np.arange(-34.75,24.8,0.5)
    #test for 2dim variable
    assert np.array_equal(d1.d1.xx2dim,np.tile(lon,(len(lat),1)))
    assert np.array_equal(d1.d1.yy2dim,np.tile(lat[::-1][:,np.newaxis],(1,len(lon))))
    #test for 3dim variable
    assert np.array_equal(d1.d1.xx3dim,np.tile(lon,(d0.unlimited_dimlen, len(lat), 1)))
    assert np.array_equal(d1.d1.yy3dim,np.tile(lat[::-1][:,np.newaxis],(d0.unlimited_dimlen, 1, len(lon))))

    nc_subgrid('testdata/grid_test_4dim.nc','testdata/grid_test_4dim_sub.nc',subgrid=[(-35,25),(-50,20)])
    d0 = Ncdata('testdata/grid_test_4dim.nc')
    d1 = Ncdata('testdata/grid_test_4dim_sub.nc')
    lon = np.arange(-49.75,19.8,0.5)
    lat = np.arange(-34.75,24.8,0.5)
    #test for 4dim variable
    assert np.array_equal(d1.d1.xx4dim,np.tile(lon,(d0.unlimited_dimlen, 13, len(lat), 1)))
    assert np.array_equal(d1.d1.yy4dim,np.tile(lat[::-1][:,np.newaxis],(d0.unlimited_dimlen, 13, 1, len(lon))))

def test_Ncdata():
    def test_3dim():
        #0 prepare test data
        dt = Ncdata('testdata/stomate_history_AS_1240_TOTAL_M_LITTER_CONSUMP_3dim.nc')
        #0.1 test PFT sum operation
        nonzero_veget_max = np.ma.masked_equal(dt.d1.VEGET_MAX, 0)
        pft_total_m = pb.MaskArrayByNan(dt.d1.TOTAL_M) * nonzero_veget_max
        pftsum_total_m = np.ma.sum(pft_total_m, axis=0)
        #0.2 test spatial operation
        pftsum_total_m_area = pftsum_total_m * dt.d1.Areas
        spasum_total_m = np.ma.sum(np.ma.sum(pftsum_total_m_area,axis=-1), axis=-1)
        spamean_total_m = spasum/np.ma.sum(dt.d1.Areas)

        #1 do the test
        dnew = Ncdata('testdata/stomate_history_AS_1240_TOTAL_M_LITTER_CONSUMP_3dim.nc')
        dnew.get_spa()
        assert np.array_equal(dnew.pftsum.TOTAL_M, pftsum_total_m)
        assert np.array_equal(dnew.pftsum.LITTER_CONSUMP, dt.d1.LITTER_CONSUMP)
        pdb.set_trace()
        assert dnew.spasum.TOTAL_M == spasum_total_m
        assert dnew.spamean.TOTAL_M == spamean_total_m

    def test_4dim():
        #0 prepare test data
        dt = Ncdata('testdata/stomate_history_TOTAL_M_LITTER_CONSUMP_4dim.nc')
        #0.1 test PFT sum operation
        nonzero_veget_max = np.ma.masked_equal(dt.d1.VEGET_MAX, 0)
        pft_total_m = pb.MaskArrayByNan(dt.d1.TOTAL_M) * nonzero_veget_max
        pftsum_total_m = np.ma.sum(pft_total_m, axis=1)
        #0.2 test spatial operation
        pftsum_total_m_area = pftsum_total_m * dt.d1.Areas
        spasum_total_m = np.ma.sum(np.ma.sum(pftsum_total_m_area,axis=-1), axis=-1)
        spamean_total_m = spasum/np.ma.sum(dt.d1.Areas)

        #1 do the test
        dnew = Ncdata('testdata/stomate_history_TOTAL_M_LITTER_CONSUMP_4dim.nc')
        dnew.get_spa()
        assert np.array_equal(dnew.pftsum.TOTAL_M, pftsum_total_m)
        assert np.array_equal(dnew.pftsum.LITTER_CONSUMP, dt.d1.LITTER_CONSUMP)
        pdb.set_trace()
        assert dnew.spasum.TOTAL_M == spasum_total_m
        assert dnew.spamean.TOTAL_M == spamean_total_m




