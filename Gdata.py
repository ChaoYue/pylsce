#!/usr/bin/env python

import gnc
import numpy as np
import mathex
import pandas as pa
from collections import OrderedDict
import pb
import copy
import Pdata

class Gdata3D(object):
    """
    Gdata3D is used to store 3D numpy array, with the first dimension as
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
            print "data shape: ",self.data.shape
            print "axis shape: ",(len(taxis),len(lat),len(lon))
            raise ValueError("""Shape of input array not equal to
                                the time,lat,lon axis length""")
        self.comment=''

    def subset(self,rtime=None,rlat=None,rlon=None):
        """
        Subset a Geodata.

        Parameters:
        -----------
        time,rlat,rlon must be tuples.
        """
        time_slice = self._get_time_slice(rtime)
        lon_slice = self._get_lon_slice(rlon)
        lat_slice = self._get_lat_slice(rlat)
        print time_slice,lon_slice,lat_slice
        subdata = self.data[time_slice,lat_slice,lon_slice].copy()
        new_taxis = self.taxis[time_slice]
        new_lat = self.lat[lat_slice]
        new_lon = self.lon[lon_slice]
        sub_gdata = Gdata3D(data=subdata,taxis=new_taxis,
                              lat=new_lat,lon=new_lon)
        return sub_gdata

    def __repr__(self):
        return "shape: {0}".format(self.data.shape) + '\n'+ \
        "timeaxis: {0}".format(repr(self.taxis))+ '\n' + \
        """lat: {0} ... {1}""".format(self.lat[[0,1,2]],self.lat[[-3,-2,-1]]) + '\n' +\
        "lon: {0} ... {1}".format(self.lon[[0,1,2]],self.lon[[-3,-2,-1]]) + '\n' + \
        self.comment

    def _get_time_slice(self,rtime=None):
        if rtime is None:
            time_slice = slice(None)
        else:
            if not isinstance(rtime,tuple):
                raise ValueError("rtime slice must be a tuple")
            else:
                if isinstance(self.taxis,np.ndarray):
                    time_index = mathex.ndarray_get_index_by_interval(
                                      self.taxis,(rtime[0],rtime[1]))[0]
                    begin = np.min(time_index)
                    end = np.max(time_index)
                    time_slice = slice(begin,end+1)

                elif isinstance(self.taxis,pa.DatetimeIndex):
                    begin = self.taxis.get_loc(rtime[0])
                    end = self.taxis.get_loc(rtime[1])
                    time_slice = slice(begin,end+1)
                else:
                    raise TypeError("rtime axis type error.")
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

    def copy(self):
        data=copy.deepcopy(self.data)
        return Gdata3D(data=data,taxis=self.taxis.copy(),
                       lat=self.lat.copy(),
                       lon=self.lon.copy())

    def apply(self,func=None,taxis=None,lat=None,lon=None):
        '''
        Apply a function on the array, if any axis needs to be changed,
        use the keyword argument.
        '''
        if taxis is None:
            taxis = self.taxis.copy()
        if lat is None:
            lat = self.lat.copy()
        if lon is None:
            lon = self.lon.copy()
        g3d = Gdata3D(data=func(self.data),taxis=taxis,lat=lat,lon=lon)
        g3d.comment = self.comment
        return g3d

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


class NGdata3D(object):
    """
    This is a wrapper of Gdata3D. It provides convenient methods to handle
    a dictionary of Gdata3D objects.

    When NGdata3D is initialized in any way, the data is not explicitly copied,
    otherwise it's copied.
    """
    def __init__(self,data=None,copy=False):
        """
        Receive a dictionary of Gdata3D objects, ideally an ordered dictionary.
        """
        if data is None:
            self.data = {}
        else:
            if copy == True:
                dic = OrderedDict()
                for key,val in data.iteritems():
                    dic[key] = val.copy()
                self.data = dic
            else:
                self.data = data

        if self.data == {}:
            self._taglist = []
        else:
            self._taglist = self.data.keys()

        self.comment=''

    def add_entry(self,tag,gdata):
        self.data[tag]=gdata
        self._taglist.append(tag)

    @classmethod
    def from_dict_of_array(cls,ydic,taxis=None,lat=None,lon=None,even=True):
        """
        Creat NGdata3D from a dictionary of 3-D arrays. The arrays must have
        the same latter two dimensions.

        Parameters:
        -----------
        even: boolean type, True for all arrays share the same shape.
        ydic:
            In case even == True, a dict of even-shaped 3-D arrays.
            In case even == False, a dict of 2-len tuples, with the first
                element giving the array, the second element giving
                the taxis. In this case the taxis keyword in function definition
                will be dropped.
        """
        if even:
            if mathex.ndarray_arraylist_equal_shape(ydic.values()):
                array_shape = np.shape(ydic.values()[0])
            else:
                raise ValueError("Please check: not all the ndarray in the dictionary share the same dimension")

            dic = OrderedDict()
            for key,val in ydic.iteritems():
                dic[key] = Gdata3D(data=val,taxis=taxis,lat=lat,lon=lon)
        else:
            dic = OrderedDict()
            for key in ydic.keys():
                data = ydic[key][0]
                taxis = ydic[key][1]
                dic[key] = Gdata3D(data=data,taxis=taxis,lat=lat,lon=lon)
        return NGdata3D(dic)

    def __getitem__(self,key):
        if isinstance(key,str):
            return self.data[key].copy()
        elif isinstance(key,list):
            return self.regroup_data_by_tag(key)

    def __setitem__(self,key,value):
        self.add_entry(key,value)

    @property
    def taglist(self):
        return self._taglist

    def __repr__(self):
        return '\n'.join([repr(self.__class__),"tags:",','.join(self._taglist)])+'\n'+\
                self.comment

    def info(self):
        infolist = [tag+':\n'+self.data[tag].__repr__() for tag in self._taglist]
        print '\n\n'.join(infolist)
        print self.comment

    def regroup_data_by_tag(self,taglist,exclude=False):
        """
        Subset data by "taglist" and return as a new NGdata3D instance.
        With all features for tag reserved.

        Parameters:
        -----------
        1. exclude: use exclude for reverse selection.

        """
        if exclude == False:
            taglist = taglist
        else:
            taglist = pb.StringListAnotB(self._taglist,taglist)

        if len([tag for tag in taglist if tag not in self._taglist]) > 0:
            raise KeyError("extract tag not present in the taglist!")
        else:
            dic = OrderedDict()
            for tag in taglist:
                dic[tag] = self.data[tag].copy()
            ng3d = NGdata3D(dic)
            return ng3d

    def apply(self,func=None,taglist=None,
                       taxis=None,lat=None,lon=None):
        '''
        Apply a function on the array for the tags as specified in taglist
        '''
        if taglist is None:
            taglist = self._taglist

        dic = OrderedDict()
        for tag in taglist:
            dic[tag] = self.data[tag].apply(func=func,taxis=taxis,
                                               lat=lat,lon=lon)
        ng3d = NGdata3D(dic)
        ng3d.comment = self.comment
        return ng3d


    def to_dataframe(self,func=None):
        """
        Convert to dataframe allowing calling functions. After calling functions,
        the data should have only one time axis.
        """
        dic = OrderedDict()
        for tag in self._taglist:
            g3d = self.data[tag]
            dic[tag] = pa.Series(func(g3d.data),index=g3d.taxis)
        return pa.DataFrame(dic)

    def to_mdata(self,func=None):
        """
        Convert to Mdata allowing calling function. After calling the function,
        the time axis should shrink to 1-len.
        """
        md = Pdata.Mdata()
        for tag in self._taglist:
            g3d = self.data[tag]
            md.add_array_lat_lon(tag=tag,data=func(g3d.data),
                                 lat=g3d.lat,lon=g3d.lon)
        return md

    def copy(self):
        dic = OrderedDict()
        for tag in self._taglist:
            dic[tag] = self.data[tag].copy()
        return NGdata3D(dic)

    def subset(self,rtime=None,rlat=None,rlon=None):
        dic = OrderedDict()
        for tag in self._taglist:
            dic[tag] = self.data[tag].subset(rtime=rtime,rlat=rlat,rlon=rlon)
        return NGdata3D(dic)

    @classmethod
    def merge_ng3d(cls,*mdlist):
        """
        Merge the mdata.
        """
        newdata = {}
        for md in mdlist:
            newdata.update(md.data)
        md = NGdata3D(data=newdata)
        return md

    #@append_doc_of(gnc._set_default_ncfile_for_write)
    def to_ncfile(self,filename,**kwargs):
        """
        An dirty and quick way to write the underlying array data
            to NetCDF file.
        Notes:
        ------
        1. "long_name" and "unit" attributes, when they're present,
            will be used as the variable attributes of the output nc file.

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
        """
        ncfile = gnc.NcWrite(filename)

        latdic = self.get_data_as_dic('lat')
        lat = latdic.values()[0]
        londic = self.get_data_as_dic('lon')
        lon = londic.values()[0]

        gnc._set_default_ncfile_for_write(ncfile,latvar=lat,lonvar=lon,**kwargs)
        ndim = len(ncfile.dimensions)
        for tag in self._taglist:
            tagdic = self.data[tag]
            attr_kwargs = {}
            if 'unit' in tagdic:
                attr_kwargs['units'] = tagdic['unit']
            if 'long_name' in tagdic:
                attr_kwargs['long_name'] = tagdic['long_name']
            if 'varname' in tagdic:
                varname = tagdic['varname']
            else:
                varname = tag
            data = self.data[tag]['array']
            ncfile.add_var_smart_ndim(varname,ndim,data,**attr_kwargs)
        ncfile.close()

    def get_zonal_as_dataframe(self,mode='mean'):
        dic = {}
        for tag in self._taglist:
            arr = self.data[tag]['array']
            if mode == 'mean':
                data = arr.mean(axis=1)
            elif mode == 'sum':
                data = arr.sum(axis=1)
            dic[tag] = data

    def set_new_tags(self, old_new_tag_tuple_list):
        """
        Change the old tag to new tag according to old_new_tag_tuple_list

        Parameters:
        -----------
        old_new_tag_tuple_list: [(oldtag1,newtag1),(oldtag2,newtag2)]

        Notes:
        ------
        1. In-place operation.
        """
        for (oldtag,newtag) in old_new_tag_tuple_list:
            if oldtag != newtag:
                self.data[newtag] = self.data[oldtag]
                del self.data[oldtag]
            self._taglist[self._taglist.index(oldtag)] = newtag




