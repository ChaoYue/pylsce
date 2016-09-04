#!/usr/bin/env python

import gnc
import numpy as np
import mathex
import pandas as pa


#############################################################################
# 2016-04-13
# this file should not be updated anymore, go to Gdata.py for everything
############################################################################

class Geodata3D(object):
    """
    Geodata3D is used to store 3D numpy array, with the first dimension as
    the time dimension, the second as the latitude, and the third as the
    longitude.
    """
    def __init__(self,data=None,taxis=None,lat=None,lon=None,):
        self.data = data
        if isinstance(taxis,(np.ndarray,pa.DatetimeIndex)):
            self.taxis = taxis
        else:
            raise TypeError("taxis type error.")
        self.lat = lat
        self.lon = lon
        if self.data.shape != (len(taxis),len(lat),len(lon)):
            raise ValueError("""Shape of input array not equal to
                                the time,lat,lon axis length""")

    def subset(self,time=None,rlat=None,rlon=None):
        """
        Subset a Geodata.

        Parameters:
        -----------
        time,rlat,rlon must be tuples.
        """
        time_slice = self._get_time_slice(time)
        lon_slice = self._get_lon_slice(rlon)
        lat_slice = self._get_lat_slice(rlat)
        print time_slice,lon_slice,lat_slice
        subdata = self.data[time_slice,lat_slice,lon_slice]
        new_taxis = self.taxis[time_slice]
        new_lat = self.lat[lat_slice]
        new_lon = self.lon[lon_slice]
        sub_gdata = Geodata3D(data=subdata,taxis=new_taxis,
                              lat=new_lat,lon=new_lon)
        return sub_gdata

    def __repr__(self):
        return "shape: {0}".format(self.data.shape) + '\n'+ \
        "timeaxis: {0}".format(repr(self.taxis))+ '\n' + \
        """lat: {0} ... {1}""".format(self.lat[[0,1,2]],self.lat[[-3,-2,-1]]) + '\n' +\
        "lon: {0} ... {1}".format(self.lon[[0,1,2]],self.lon[[-3,-2,-1]])

    def _get_time_slice(self,time=None):
        if time is None:
            time_slice = slice(None)
        else:
            if not isinstance(time,tuple):
                raise ValueError("time slice must be a tuple")
            else:
                if isinstance(self.taxis,np.ndarray):
                    time_index = mathex.ndarray_get_index_by_interval(
                                      self.taxis,(time[0],time[1]))[0]
                    begin = np.min(time_index)
                    end = np.max(time_index)
                    time_slice = slice(begin,end+1)

                elif isinstance(self.taxis,pa.DatetimeIndex):
                    begin = self.taxis.get_loc(time[0])
                    end = self.taxis.get_loc(time[1])
                    time_slice = slice(begin,end+1)
                else:
                    raise TypeError("time axis type error.")
        return time_slice

    def _get_lon_slice(self,rlon=None):
        if rlon is None:
            lon_slice = slice(None)
        else:
            if not isinstance(rlon,tuple):
                raise ValueError("rlon must be a tuple")
            else:
                vlon1,vlon2 = rlon
                lon_index = mathex.ndarray_get_index_by_interval(
                                  self.lon,(vlon1,vlon2))[0]
                begin = np.min(lon_index)
                end = np.max(lon_index)
                lon_slice = slice(begin,end+1)
        return lon_slice

    def _get_lat_slice(self,rlat=None):
        if rlat is None:
            lat_slice = slice(None)
        else:
            if not isinstance(rlat,tuple):
                raise ValueError("rlat must be a tuple")
            else:
                vlat1,vlat2 = rlat
                lat_index = mathex.ndarray_get_index_by_interval(
                                  self.lat,(vlat1,vlat2))[0]
                begin = np.min(lat_index)
                end = np.max(lat_index)
                lat_slice = slice(begin,end+1)
        return lat_slice

    def to_nc(self,filename,varname,**var_attr_kwargs):
        """
        This is a VERY loose link to write out the data into netcdf file,
        in order to allow the data to be opened by gnc.Ncdata.

        In future, the Geodata will be directly linked to gnc.Ncdata
        """
        ncfile = gnc.NcWrite(filename)
        ncfile.add_3dim_time_lat_lon(time_length=len(self.taxis),
                                     lat_value=self.lat,lon_value=self.lon)
        ncfile.add_var_3dim_time_lat_lon(varname,self.data,**var_attr_kwargs)
        ncfile.close()


