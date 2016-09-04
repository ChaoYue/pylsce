#!/usr/bin/env python


"""
The module name is pb (which means python basic), it contains:
1. basic extra functions for python basic objects operation (like on file, list, dictionary etc.)
2. some functions on operation on netcdf files which are mainly through netCDF4 library.
"""

import numpy as np
import pickle as pk
import os as os
import re as re
import pdb
import netCDF4 as nc
from collections import Iterable
from collections import OrderedDict
import csv

def FilterStringList(keyword,input_list):
    return [x for x in input_list if re.search(keyword,x)]

#note two ways for list comprehension containing if clause:
#a=[d for d in list if d>5]
#a=[d if d<5 else 0 for d in list]
def FilterListClass(objclass,inlist,exclude=False):
    """
    select list elements that [NOT when exclude=True] belong to (a group of) classes. 
    Arguments:
        objclass --> class name or tuple of class name
    """
    if exclude==True:
        return [lm for lm in inlist if not isinstance(lm,objclass)]
    else:
        return [lm for lm in inlist if isinstance(lm,objclass)]

def FilterAttributeValue(ObjectInstanceList,*condition_tuples):
    """
    Purpose: Filter a list of object instances by their attribute value; supply with (attribute_name,attribute_value) tuples
    Use:
        FilterAttributeValue([ins1,ins2,ins3],(attr1,value1),(attr2,value2))
    Note:
        1. Instances in ObjectInstanceList must all have attribute as listed in attr/attr_value pairs.
        2. logical AND is applied among multiple tuples.
    """
    select_bool=[True]*len(ObjectInstanceList)
    for i,artist in enumerate(ObjectInstanceList):
        for condition_tuple in condition_tuples:
            attribute_name,attribute_value=condition_tuple
            try:
                if artist.__dict__[attribute_name]==attribute_value:
                    pass
                else:
                    select_bool[i]=False
                    break
            except:
                pass
    return [obinstance for boolflag,obinstance in zip(select_bool,ObjectInstanceList) if boolflag==True]

def Dic_Filter_by_Key(inputdic,*keys):
    """
    Filter a dictionary for by checking if key contains (one or more) strings
    """
    outdic={}
    for inkey in inputdic.keys():
        state=True
        for key in keys:
            if key not in inkey:
                state=False
                break
            else:
                pass
        if state==True:
            outdic[inkey]=inputdic[inkey]
    return outdic

def ListSingleValue(d):
    flag=True
    for i in d:
        if i!=d[0]:
            flag=False
            return flag
    return d[0]


def iteflat(inlist):
    """
    Purpose: flat all things that are member elements in a iteralbe object (eg. tuple,list) to a list.
            inlist can also be itself a iteralbe with non-iterable elements (eg. np array). 
    Note:
        1. Check the source code when confused by this function and the definition is very clear.
        2. 2nd level nested lists will not be flatted. eg. 
           ['a','b','c'] --> no nested list
           [['a','b','c'],['e','f']]  --> 1st nested list.
           [['a','b','c'],['e','f',['h','l']]]  --> 2nd nested list.
    Example:
        >>> inlist1=[(np.NINF),(3),(np.PINF)]
        >>> inlist2=[(-10,-3),np.arange(-2,2.1,0.5),(3,10)]
        >>> inlist3=[(-10,-3),(0),(3,10)]

        In [28]: pb.iteflat(inlist1)
        Out[28]: [-inf, 3, inf]

        In [29]: pb.iteflat(inlist2)
        Out[29]: [-10, -3, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3, 10]

        In [30]: pb.iteflat(inlist3)
        Out[30]: [-10, -3, 0, 3, 10]

        In [55]: pb.iteflat([(-10,-3),0,(3,10)])
        Out[55]: [-10, -3, 0, 3, 10]

        In [56]: pb.iteflat(np.arange(10))
        Out[56]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        In [58]: pb.iteflat(((0,2,3),(1,2,3)))
        Out[58]: [0, 2, 3, 1, 2, 3]

        In [122]: pb.iteflat([['a','b','c'],['e','f',['h','l']]])
        Out[122]: ['a', 'b', 'c', 'e', 'f', ['h', 'l']]

        In [123]: pb.iteflat([['a','b','xyz'],['e','f',['h','l']]])
        Out[123]: ['a', 'b', 'xyz', 'e', 'f', ['h', 'l']]

    """
    outlist=[]
    for member in inlist:
        if not isinstance(member,Iterable) or isinstance(member,str):
            outlist.append(member)
        else:
            for submember in member:
                #return iteflat(member)
                outlist.append(submember)
    return outlist

def iteflat2(inlist):
    outlist=[]
    for member in inlist:
        if not isinstance(member,Iterable) or isinstance(member,str):
            outlist.append(member)
        else:
            outlist.append(iteflat2(member))
    return outlist

def StringListAnotB(listA,listB):
    return [i for i in listA if i not in listB]

def StringListAandB(listA,listB):
    return [i for i in listA if i in listB]

def PdfFile(filename):
    """
    Creat and return a pdf backend for ploting.
    """
    pp = PdfPages(filename)
    return pp

def Remove_dupTL(input_list_or_tuple,remove_element):
    if isinstance(input_list_or_tuple,tuple):
        templist=list(input_list_or_tuple)
        for i in range(templist.count(remove_element)):
            templist.remove(remove_element)
        return tuple(templist)
    elif isinstance(input_list_or_tuple,list):
        templist=input_list_or_tuple
        for i in range(templist.count(remove_element)):
            templist.remove(remove_element)
        return templist

def Remove_dupdim(input_numpy_array):
    or_shape=input_numpy_array.shape
    new_reshape=Remove_dupTL(or_shape,1)
    return np.reshape(input_numpy_array,new_reshape)
class ncdata(object):
    pass

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
    datanew.__dict__['filename']=filename
    for var in f1.variables.keys():
        var=str(var)
        datanew.__dict__[var]=f1.variables[var]
        datanew2.__dict__[var]=Remove_dupdim(f1.variables[var][:])
    return [datanew,datanew2]
    f1.close()

def ncreadg_pftspa(filename,pftsum=None,spaop=None):
    """
    Purpose: read regional nc file and make PFT weighted sum and sptial sum as required. 
    Definition: This is an easy application of ncreadg
    Arguments:
        file--> file name;  spaop (spatial operation) --> set 'sum' if spatial sum is desired and 'mean' if spatial mean is wanted.
    Return: return a list of ncdata object; the 1st one contains original nctCDF4 objects and the 2nd one contains data with duplicate dimensions removed.
            the 3rd one contains spatial PFT weighted sum; the 4th (spaop='sum' or 'mean') contains the PFT weighted spatial sum or mean.
    Note:
        1. can only hand the case of 4 dimensions as (time,PFT,lat,lon)
        2. can only handlf ORCHIDEE output and use "VEGET" as PFT fraction variable to make PFT weighted sum.
        
    Example:
        >>> d0,d1,d2,None=ncreadg_pftspa(filename)
        >>> d0,d1,d2,d3=ncreadg_pftspa(filename,True)
    """

    d0,d1=ncreadg(filename)
    if 'VEGET' not in d0.__dict__.keys():
        raise AttributeError('Variable VEGET is not in the file and cannot make PFT weighted sum')
    else:
        d2=ncdata()
        d3=ncdata()
        if pftsum==True:
            for var in d1.__dict__.keys():
                #only treat the 4-dimensional variable
                if d1.__dict__[var].ndim==4:
                    #assume PFT is always the 2nd dimension
                    if d1.__dict__[var].shape[1]!=13:
                        raise ValueError('the PFT is not the 2nd dimension')
                        print 'PFT is not the 2 dimension of variable ',var
                    else:
                        #do not treat VEGET
                        if var!='VEGET':
                            temp=d1.__dict__[var]*d1.__dict__['VEGET']
                            #make PFT weighted sum (this is required whatever the final spatial operation)
                            temppftsum=np.ma.sum(temp,axis=1)
                            d2.__dict__[var]=temppftsum
                            if spaop=='sum':
                                #make spatial sum, always assume sptial as the last 2 dimension
                                tempspasum=np.ma.sum(np.ma.sum(temppftsum,axis=1),axis=1)
                                d3.__dict__[var]=tempspasum
                            elif spaop=='mean':
                                #make spatial mean, always assume sptial as the last 2 dimension
                                tempspasum=np.ma.mean(np.ma.mean(temppftsum,axis=1),axis=1)
                                d3.__dict__[var]=tempspasum
                else:
                    raise ValueError("dimension for var '{0}' is not 4".format(var))
        if pftsum is None:
            return d0,d1,None,None
        else:
            if spaop is None:
                return d0,d1,d2,None
            else:
                return d0,d1,d2,d3


def ncread(filename):
    """
    Purpose: read a .nc file using netCDF4 package and read the data into netCDF4.Variable object
    Definition: ncread(file)
    Arguments:
        file--> file name
    Return: return a list of ncdata object; the 1st one contains original nctCDF4 objects and the 2nd one contains data with duplicate dimensions removed.
    Note:
        1. NEP and AUTO_RESP, BM_ALLOC_TOTAL, if not existing and can be calculated from input files, are automatically calculated.

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


    #calculate NEP and AUTO_RESP
    if 'NEP' not in datanew.__dict__.keys():
        if 'NPP' in f1.variables.keys() and 'HET_RESP' in f1.variables.keys():
            datanew.__dict__['NEP']=datanew2.__dict__['NPP']-datanew2.__dict__['HET_RESP']
            datanew2.__dict__['NEP']=datanew2.__dict__['NPP']-datanew2.__dict__['HET_RESP']
    if 'AUTO_RESP' not in datanew.__dict__.keys():
        if 'MAINT_RESP' in f1.variables.keys() and 'GROWTH_RESP' in f1.variables.keys():
            datanew.__dict__['AUTO_RESP']=datanew2.__dict__['MAINT_RESP']+datanew2.__dict__['GROWTH_RESP']
            datanew2.__dict__['AUTO_RESP']=datanew2.__dict__['MAINT_RESP']+datanew2.__dict__['GROWTH_RESP']

    #make the total allocated biomass
    alloc_flag=True
    alloc_list=['BM_ALLOC_FRUIT','BM_ALLOC_HEART_AB','BM_ALLOC_ROOT','BM_ALLOC_SAP_AB','BM_ALLOC_HEART_BE','BM_ALLOC_LEAF','BM_ALLOC_SAP_BE','BM_ALLOC_RES']
    alloc_shape=[]
    for var in alloc_list:
        if var not in f1.variables.keys():
            print var,'not in input file!'
            alloc_flag=False
            break
        else:
            alloc_shape.append(datanew2.__dict__[var].shape)
    if alloc_flag and ListSingleValue(alloc_shape):
        tempvar=np.zeros_like(datanew2.__dict__['BM_ALLOC_FRUIT'])
        for var in alloc_list:
            tempvar=tempvar+datanew2.__dict__[var]
        datanew.__dict__['BM_ALLOC_TOTAL']=tempvar
        datanew2.__dict__['BM_ALLOC_TOTAL']=tempvar
    return [datanew,datanew2]
    f1.close()

def duplicates(lst, item):
    return [i for i,x in enumerate(lst) if x==item]
def Get_PointValue(data,var,(lat,vlat),(lon,vlon)):
    """
    Purpose: get the value of variable 'var' for the point(vlat,vlon) from the grid.
    Definition: Get_PointValue(data,var,(lat,vlat),(lon,vlon))
    Arguments:
        data --> a ncdata object; var --> variable name for which value needs to be retrieved, data.var should be a netCDF4.Variable object.
        lat --> latitude coordinate; lon --> longitude coordinate
    Return: a ndarray of var specified by location
    Note: 
        1. the lat and lon must be the two last dimensions of variable 'var'.
        2. the lat must be a decreasing sequence while the lon must a increasing sequence.
        3. only applies to extract value for a single point
        4. var, lat, and lon must be provided as str
    Test:
        1. This function have been test again on 2012/06/26, if the lat value is exactly in the middle of two points (like 51 between 50.75 and 51.25), the lower bound 
            will be returned (50.75 is returned).
    Example:
        >>> data
         <g.ncdata object at 0x32b6990>
        >>> data.air
         <netCDF4.Variable object at 0x18b3cd0>
        >>> data.air.dimensions
         (u'time', u'lat', u'lon')
        >>> data.air.shape
         (1464, 94, 192)
        >>> temp=g.Get_GridValue(data,'air',('lat',54.49),('lon',150.65))
        >>> temp.shape
         (1464,)
    """

    lat=data.__dict__[lat][:]
    lon=data.__dict__[lon][:]
    if lat.ndim == 2:
        lat=lat[:,0]
    else:
        pass
    if lon.ndim == 2:
        lon=lon[0,:]
    else:
        pass

    index_more=np.where(lat>=vlat)[0][-1]
    index_less=np.where(lat<=vlat)[0][0]
    if abs(lat[index_more]-vlat) >= abs(lat[index_less]-vlat):
        index_lat=index_less
    else:
        index_lat=index_more
    index_more=np.where(lon>=vlon)[0][0]
    index_less=np.where(lon<=vlon)[0][-1]
    if abs(lon[index_more]-vlon) >= abs(lon[index_less]-vlon):
        index_lon=index_less
    else:
        index_lon=index_more
    target=data.__dict__[var][:][...,index_lat,index_lon]
    return target

def Get_GridValue(data,var,(lat,vlat1,vlat2),(lon,vlon1,vlon2)):
    """
    Purpose: get the value of variable 'var' for a subgrid (vlat1:vlat2,vlon1:vlon2) from the grid.
    Definition: Get_GridValue(data,var,(lat,vlat1,vlat2),(lon,vlon1,vlon2))
    Arguments:
        data --> a ncdata object; var --> variable name for which value needs to be retrieved, data.var should be a netCDF4.Variable object.
        lat --> latitude coordinate; lon --> longitude coordinate
        vlat1-->vlat2 (South-->North); vlon1-->vlon2(West-->East)
    Return: a ndarray of var specified by location
    Note: 
        1. the lat and lon must be the two last dimensions of variable 'var'.
        2. the lat must be a decreasing sequence while the lon must a increasing sequence.
        3. only applies to extract value for a single point
        4. var, lat, and lon must be provided as str
    Example:
        >>> ba=g.ncread('/home/orchidee01/ychao/FIRE_DATA/GFED3/monthly/BA2009.nc')
        >>> ba.ba.shape
         (12, 360, 720)
        >>> subba=g.Get_GridValue(ba,'ba',('lat',55,56),('lon',-99,-98))
        >>> subba.shape
         (12, 2, 2)

    """
    lat=data.__dict__[lat][:]
    lon=data.__dict__[lon][:]
    if lat.ndim == 2:
        lat=lat[:,0]
    else:
        pass
    if lon.ndim == 2:
        lon=lon[0,:]
    else:
        pass

    index_lat=np.nonzero((lat[:]>=vlat1)&(lat[:]<=vlat2))[0]
    index_lon=np.nonzero((lon[:]>=vlon1)&(lon[:]<=vlon2))[0]
    target=data.__dict__[var][:][...,index_lat[0]:index_lat[-1]+1,index_lon[0]:index_lon[-1]+1]
    return target

def pfload(filename):
    fob=file(filename,'r')
    data=pk.load(fob)
    fob.close()
    return data

def pfdump(data,filename):
    """
    Purpose: use python pickle to dump data object to a file
    Definition: pfdump(data,filename)
    """
    fob=file(filename,'w')
    data=pk.dump(data,fob)
    fob.close()

def ipsearch(string):
    """
    [str(l) for l in  _ih if l.startswith(string)]
    """
    [str(l) for l in  _ih if l.startswith(string)]


def list2array(al):
    """
    Purpose: transform a list of ndarrays(with different length) to a single masked ndarray. the ndarray as list members must be 1d array or list.
    In [27]: d=[range(i) for i in range(2,7)]

    In [28]: d
    Out[28]: [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]

    In [29]: pb.list2array(d)
    Out[29]: 
    masked_array(data =
     [[0.0 1.0 -- -- -- --]
      [0.0 1.0 2.0 -- -- --]
       [0.0 1.0 2.0 3.0 -- --]
        [0.0 1.0 2.0 3.0 4.0 --]
         [0.0 1.0 2.0 3.0 4.0 5.0]],
     mask =
      [[False False  True  True  True  True]
       [False False False  True  True  True]
        [False False False False  True  True]
         [False False False False False  True]
          [False False False False False False]],
                 fill_value = 1e+20)

    """
    nrow=len(al)
    ncol=max([len(i) for i in al])
    array=np.ma.empty((nrow,ncol))
    for ind,sub in enumerate(al):
        array[ind,0:len(sub)]=sub[:]
        array[ind,len(sub):]=np.nan
    array=np.ma.masked_where(np.isnan(array),array)
    return array

def List2Array(al):
    return list2array(al)

class calendar(object):
    leap=[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    noleap=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    DayMonleap=dict(zip(month,leap))
    DayMonnoleap=dict(zip(month,noleap))


    def __init__(self):
        pass

#    def getdoy2((mon,day),leap=False):
#        leap=[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#        noleap=[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#        if leap:
#            doy=np.sum(np.array(leap[0:mon-1]))+day
#        else:
#            doy=np.sum(np.array(noleap[0:mon-1]))+day
#        return int(doy)

#    doy_begin_month_noleap = []
#    doy_end_month_noleap = []
#    doy_begin_month_leap = []
#    doy_end_month_leap = []
#    for mon in range(1,13):
#        doy_begin_month_noleap.append(getdoy2((mon,1)))
#        doy_end_month_noleap.append(getdoy2((mon+1,1))-1)
#        doy_begin_month_leap.append(getdoy2((mon,1),True))
#        doy_end_month_leap.append(getdoy2((mon+1,1),True)-1)
#    index_first_day_month_noleap = np.array(doy_begin_month_noleap)-1
#    index_first_day_month_leap = np.array(doy_begin_month_leap)-1

    def getdoy(self,(mon,day),leap=False):
        """
        calculate the DOY(day of year) according to (mon,day), set leap as True for leapyear.
        """
        if leap:
            doy=np.sum(np.array(self.leap[0:mon-1]))+day
        else:
            doy=np.sum(np.array(self.noleap[0:mon-1]))+day
        return int(doy)

    def isleap(self,year):
        """
        True for leap year, otherwise False
        """
        if year%4==0:
            return True
        else:
            return False

    def get_day_number(self,year):
        """
        Return the number of days in each month by checking input year is
            a leap/noleap year.
        """
        if self.isleap(year):
            return self.leap
        else:
            return self.noleap

    def get_month_doy(self):
        """
        build the index for DOY at the beginning and end of the month.
        """
        self.doy_begin_month_noleap = []
        self.doy_end_month_noleap = []
        self.doy_begin_month_leap = []
        self.doy_end_month_leap = []
        for mon in range(1,13):
            self.doy_begin_month_noleap.append(self.getdoy((mon,1)))
            self.doy_end_month_noleap.append(self.getdoy((mon+1,1))-1)
            self.doy_begin_month_leap.append(self.getdoy((mon,1),True))
            self.doy_end_month_leap.append(self.getdoy((mon+1,1),True)-1)
        self.index_first_day_month_noleap = \
            np.array(self.doy_begin_month_noleap)-1
        self.index_first_day_month_leap = \
            np.array(self.doy_begin_month_leap)-1

def MaskArrayByNan(array):
    """
    Purpose: set np.nan to masked elements. the input can itself a masked array.
    """
    return np.ma.masked_array(array,np.isnan(array))

def linspace_array(input_array,num=20):
    """
    Purpose: create an array using the min and max of input_array equally divided by num of points.
    """
    input_array = np.ma.masked_invalid(input_array)
    return np.linspace(np.ma.min(input_array),np.ma.max(input_array),num=num,endpoint=True)

def shared_unmask_data(a,b):
    """
    Purpose: Return shared unmasked data between two arrays; the sequence for a,b is unchanged.
    Example:
        a_mutual_unmasked,b_mutual_unmasked=pb.shared_unmask_data(a,b)
    """
    a=MaskArrayByNan(a)
    b=MaskArrayByNan(b)
    return a[(~a.mask)&(~b.mask)].data,b[(~a.mask)&(~b.mask)].data


def Dic_Apply_Func(pyfunc,indic):
    """
    Purpose: apply a function to value of (key,value) pair in a dictionary and maintain the dic.
    Arguments:
        pyfunc --> a python function object.
    """
    outdic=OrderedDict()
    for key in indic.keys():
        outdic[key]=pyfunc(indic[key])
    return outdic

def Dic_Nested_Apply_Func(pyfunc,indic):
    """
    Use Dic_Apply_Func to apply func on nested dic.
    """
    outdic = OrderedDict()
    for key,subdata in indic.items():
        outdic[key] = Dic_Apply_Func(pyfunc,subdata)
    return outdic

def arg_less(inarray,threshold):
    return np.nonzero(inarray<threshold)

def arg_less_equal(inarray,threshold):
    return np.nonzero(inarray<=threshold)

def arg_greater(inarray,threshold):
    return np.nonzero(inarray>threshold)

def arg_greater_equal(inarray,threshold):
    return np.nonzero(inarray>=threshold)

def arg_between(inarray,lowerbound,upperbound):
    if lowerbound>=upperbound:
        raise ValueError('the first argument is lower bound, lower bound bigger than upper bound!')
    else:
        return np.nonzero((inarray>lowerbound)&(inarray<upperbound))

def arg_between_equal(inarray,lowerbound,upperbound):
    if lowerbound>upperbound:
        raise ValueError('the first argument is lower bound, lower bound bigger than upper bound!')
    else:
        return np.nonzero((inarray>=lowerbound)&(inarray<=upperbound))

def arg_outside(inarray,lowerbound,upperbound):
    if lowerbound>=upperbound:
        raise ValueError('the first argument is lower bound, lower bound bigger than upper bound!')
    else:
        return np.nonzero((inarray<lowerbound)|(inarray>upperbound))

def arg_outside_equal(inarray,lowerbound,upperbound):
    if lowerbound>upperbound:
        raise ValueError('the first argument is lower bound, lower bound bigger than upper bound!')
    else:
        return np.nonzero((inarray<=lowerbound)|(inarray>=upperbound))

def blankspace2csv(blank_sep_file,csvfile):
    """
    Purpose: change a blankspace separated file to a comma separated file. Different columns in the blank space delimited file can have arbitrary number of blank spaces.
    """
    csv.register_dialect('whitespace',delimiter=' ',skipinitialspace=True)
    fob=open(blank_sep_file)
    csvob=csv.reader(fob,dialect='whitespace')
    fob2=file(csvfile,'w')
    csv_writer = csv.writer(fob2)
    for d in csvob:
        csv_writer.writerow(d)
    fob2.close()

def Check_Dic_Equal_Length(inputdic):
    """
    Check if all the inputdic[key] have the same length
    """

def Is_Nested_Dic(indic):
    for value in indic.values():
        if isinstance(value,dict):
            return True
        else:
            pass
    return False

def Dic_Is_Nested(indic):
    return Is_Nested_Dic(indic)

def Dic_Extract_By_Subkeylist(indic,keylist):
    """
    Return a new dic by extracting the key/value paris present in keylist
    """
    outdic={}
    for key in keylist:
        try:
            outdic[key]=indic[key]
        except KeyError:
            raise KeyError("input key {0} not present!".format(key))
    return outdic

def Dic_Remove_By_Subkeylist(indic,keylist):
    """
    Return a new dic, with key/value pairs present in keylist removed.
    """
    outdic=indic.copy()
    for key in outdic.keys():
        if key in keylist:
            del outdic[key]
    return outdic


def List_Duplicate_Check(inlist):
    return any(inlist.count(x) > 1 for x in inlist)

def Lists_Intersection_Check(first_order_nestedlist):
    return (List_Duplicate_Check(iteflat(first_order_nestedlist)))

def Dic_Sort_Value_by_Key_Seq(indic,keylist):
    """
    Return a list of dic values, with their sequence as provided in keylist
    """
    outlist=[]
    for key in keylist:
        outlist.append(indic[key])
    return outlist

def Dic_Key_Value_Reverse(indic):
    """
    Reverse key:value pair to value:key pair, note indic must be 1-1 projection.
    """
    outdic={}
    for key,value in indic.items():
        outdic[value]=key
    return outdic

def List_Combine_to_List(*listmember):
    """
    Combine several "listmembers" to a list
    """
    outlist=[]
    for member in listmember:
        outlist.append(member)
    return outlist

def Dic_by_List_of_Tuple(inlist):
    """
    Convert a list of (key,value) tuples to {key:value} dictionary
    """
    outdic={}
    for key,value in inlist:
        outdic[key]=value
    return outdic

def Dic_Test_Empty_Dic(indic):
    """
    check if indic is a (nested) empty dictionary. 
    """
    if indic.keys()!=[]:
        for key,keydata in indic.items():
            if isinstance(keydata,dict):
                if keydata=={}:
                    pass
                else:
                    return Dic_Test_Empty_Dic(keydata)
            else:
                return False
        return True
    else:
        return False

def Dic_Create_by_Shared_Key(keydic=None,valuedic=None):
    """
    Create a third dictionary by using the key from keydic values, and the value from valuedic values.
    keydic and valuedic should share the same key.
    """
    outdic = {}
    for key in keydic.keys():
        if key in valuedic.keys():
            outdic[keydic[key]] = valuedic[key]
    return outdic


def Dic_Subset_End(indic,end_num):
    """
    subset a dictionary by retaining only the last "end_num" elements.
    Note:
        1. Test has been done for only the case that indic[key] is 1D ndarray.
    """
    outdic={}
    for key,value in indic.items():
        try:
            newvalue=value[-end_num:]
        except:
            newvalue=value
        outdic[key]=newvalue
    return outdic

def Dic_Subset_End_Since(indic,end_num):
    """
    subset a dictionary by retaining several last elements since index end_num.
    it works like [newdic[key] = olddic[key][end_num:] for key in old_dic.keys()]
    Note:
        1. Test has been done for only the case that indic[key] is 1D ndarray.
    """
    outdic={}
    for key,value in indic.items():
        try:
            newvalue=value[end_num:]
        except:
            newvalue=value
        outdic[key]=newvalue
    return outdic

def Dic_Subset_Begin(indic,end_num):
    """
    subset a dictionary by retaining only the beginning "end_num" elements.
    Note:
        1. Test has been done for only the case that indic[key] is 1D ndarray.
    """
    outdic={}
    for key,value in indic.items():
        try:
            newvalue=value[0:end_num]
        except:
            newvalue=value
        outdic[key]=newvalue
    return outdic

def List_Sort_by_Index(inlist,seq_index_list):
    """
    Return a list sorted by given seq_index_list; Give seq_index_list an 1_based normal sequence which correponds to member in inlist
    """
    dic_temp={}
    for i,list_member in zip(seq_index_list,inlist):
        dic_temp[i]=list_member
    return dic_temp.values()


def String_List_Remove_by_Keyword(keyword,string_list):
    remove_list=FilterStringList(keyword,string_list)
    return StringListAnotB(string_list,remove_list)

def TupleofList_to_ListofTuple(intuple):
    """
    Change a tuple of equal-length list to a list of equal-length tuples
    """

def Dic_Compare_2Dic_of_Nparray(dic1,dic2):
    """
    Purpose: compare two dictionaries with nparray or numerical list as their values; return the keys with different values.
    """
    keylist_samevalue=[]
    keylist_diffvalue=[]
    for key in dic1.keys():
        if key in dic2:
            if isinstance (dic1[key],list) and isinstance(dic2[key],list):
                value1 = np.array(dic1[key])
                value2 = np.array(dic2[key])
            elif isinstance (dic1[key],np.ndarray) and isinstance(dic2[key],np.ndarray):
                value1,value2 =dic1[key],dic2[key]
            else:
                raise TypeError("the value for key '{0}' in one of the dictionaries is not list or ndarray".format(key))

            if np.allclose(value1,value2):
                keylist_samevalue.append(key)
            else:
                keylist_diffvalue.append(key)

        else:
            pass
    return keylist_diffvalue

def Dic_from_2Dnparray(array,keylist=None,axis=1):
    """
    Purpose: change the 2Dndarray into dictionary by row(axis=0) or column(axis=1).
    """
    outdic={}
    if np.ndim(array) != 2:
        raise ValueError("The input array ndim is {0}".format(np.ndim(array)))
    else:
        nrow, ncol = np.shape(array)
        if axis == 0:
            if len(keylist) != nrow:
                raise ValueError("input array nrow is {0} but keylist length is {1}".format(nrow,len(keylist)))
            else:
                for i in range(nrow):
                    outdic[keylist[i]] = np.copy(array[i,:])
        elif axis == 1:
            if len(keylist) != ncol:
                raise ValueError("input array ncol is {0} but keylist length is {1}".format(ncol,len(keylist)))
            else:
                for i in range(ncol):
                    outdic[keylist[i]] = np.copy(array[:,i])
        else:
            raise ValueError("wrong axis value")
    return outdic


def Dic_Nested_Permuate_Key(nested_dic):
    """
    Purmuate keys for the nested dictionary, the nested_dic will have two tires of keys. The output dictionary will interchange the position of the 1-tire and 2-tire keys.

    Parameters:
    ----------
    nested_dic: all elements of the nested_dic.values() should be dictionary and they must have exactly the same keys.

    Examples:
    ---------
    >>> indic = {'m':dict(zip(['a','b','c'],[1,2,3])), 'n': dict(zip(['a','b','c'],[1000,2000,3000]))}
    >>> pb.Dic_Nested_Permuate_Key(pb.Dic_Nested_Permuate_Key(indic)) == indic
    """
    second_tire_key_length = len(nested_dic.values()[0])
    outdic = OrderedDict()
    for second_key in nested_dic.values()[0].keys():
        outdic[second_key] = OrderedDict()
    for first_key,subdic in nested_dic.items():
        if len(subdic) != second_tire_key_length:
            raise ValueError("the subdictionary {0} has not equal 2-tire key length with others!".format(first_key))
        else:
            for second_key in subdic.keys():
                outdic[second_key][first_key] = subdic[second_key]

    return outdic

def Dic_replace_key_by_dict(indic,keydic):
    """
    Build a new dictionary, with the `key` in indic being replaced by keydic[key].
    """
    outdic = OrderedDict()
    for key,val in keydic.items():
        outdic[val] = indic[key]
    return outdic



def indexable_check_same_order(a,b):
    """
    check if the two indexable are in the same, note the member of indexable must be numbers.
    """
    if (a[0]-a[1])*(b[0]-b[1]) > 0:
        return True
    else:
        return False


def object_list_check_unique_attribute(object_list,attr_name):
    """
    check if all the objects in the list have the same attribute value.
    """
    unique = True
    attr0 = object_list[0].__getattribute__(attr_name)
    for obj in object_list[1:]:
        if obj.__getattribute__(attr_name) == attr0:
            pass
        else:
            unique = False
            break
    return unique

def object_list_check_any_has_attribute(object_list,attr_name):
    """
    check if any object in the list has the attribute.
    """
    unique = False
    for obj in object_list:
        if hasattr(obj,attr_name):
            unique = True
            break
        else:
            pass
    return unique


