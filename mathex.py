#!/usr/bin/env python


import statsmodels.api as sm
import numpy as np
import scipy as sp
import re as re
from scipy import stats
import gnc
import netCDF4 as nc
import copy as pcopy
import pdb
import pb
import pandas as pa
import g
import datetime
from scipy import interpolate
from scipy import stats, linalg
from collections import OrderedDict

def rolling_window(a, window):
    '''
    Make an ndarray with a rolling window of the last dimension

    Parameters
    ----------
    a : array_like
        Array to add rolling window to
    window : int
        Size of rolling window

    Returns
    -------
    Array that is a view of the original array with a added dimension
    of size w.

    Examples
    --------
    >>> x=np.arange(10).reshape((2,5))
    >>> rolling_window(x, 3)
    array([[[0, 1, 2], [1, 2, 3], [2, 3, 4]],
           [[5, 6, 7], [6, 7, 8], [7, 8, 9]]])

    Calculate rolling mean of last dimension:
    >>> np.mean(rolling_window(x, 3), -1)
    array([[ 1.,  2.,  3.],
           [ 6.,  7.,  8.]])

    '''
    if window < 1:
        raise ValueError, "`window` must be at least 1."
    if window > a.shape[-1]:
        raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides) 

def move_ave1d(a,window):
    '''
    Purpose: Return the moving average of 1 dimensional ndarray 'a' by the window size of 'window'
    Note:
        1. apply only to non masked array.
    '''
    temp=rolling_window(a,window)
    if window==1:
        return np.mean(temp,-1)
    elif window==2:
        return np.concatenate((np.array([a[0]]),np.mean(temp,axis=-1)))
    else:
        pre=int(window)/2
        post=int(window)-pre-1
        return np.concatenate((a[...,0:pre],np.mean(temp,-1),a[...,-post:]),axis=-1)

def move_ave2d(a,window,mvaxis=1):
    '''
    Purpose: Return the moving average of 2 dimensional ndarray 'a' by the window size of 'window' along axis of 'mvaxis'
             The default moving average is rowwise.
    Note:
        1. apply only to non masked array.
    '''
    if mvaxis==1:
        data=[]
        for d in a:
            temp=move_ave1d(d,window)
            data.append(temp)
        return np.array(data)
    elif mvaxis==0:
        data=[]
        for d in a.transpose():
            temp=move_ave1d(d,window)
            data.append(temp)
        return np.transpose(np.array(data))


def group_by_interval(input_array,group_interval,output_list=None):
    """
    group the data by given interval list according to the given group_interval list.

    Output:
    -------
    ndarray with unique values as np.arange(1,len(group_interval))


    group_by_interval()
    """
    array=input_array.copy()
    boolen_list=[]
    for i in range(len(group_interval)-1):
        if i < i == len(group_interval)-2:
            try:
                boolen_list.append(np.logical_and(np.logical_and(array>=group_interval[i], array<group_interval[i+1]), ~array.mask))
            except AttributeError:
                boolen_list.append(np.logical_and(array>=group_interval[i], array<group_interval[i+1]))
        else:
            try:
                boolen_list.append(np.logical_and(np.logical_and(array>=group_interval[i], array<=group_interval[i+1]), ~array.mask))
            except AttributeError:
                boolen_list.append(np.logical_and(array>=group_interval[i], array<=group_interval[i+1]))

    for i in range(len(boolen_list)):
        if output_list is None:
            array[boolen_list[i]] = i+1
        else:
            array[boolen_list[i]] = output_list[i]

    try:
        boolen_list.append(array.mask)
    except AttributeError:
        pass
    dt3 = np.array(boolen_list)
    if not np.all(np.any(dt3,axis=0)):
        raise Warning("This given interval list for grouping is not exhaustive!")

    return array




def lintrans(x,inboundtuple,outboundtuple):
    """
    Purpose: make a linear transformation of data for enhancing contrast.
             Suppose, the input data is x which lies in [x1,x2], we want to transform it into range [y1,y2] by a linear way.
    Arguments:
        inboundtuple --> the original bound tuple (x1,x2)
        outboundtuple --> the transformed bound tuple (y1,y2)
    Example:
        >>> import mathex.lintrans as lintrans
        >>> lintrans(3,(2,4),(4,8))
        >>> plt.plot(np.arange(2,5),lintrans(np.arange(2,5),(2,4),(4,10)))
        >>> plt.hlines(y=lintrans(3,(2,4),(4,10)),xmin=0,xmax=3,linestyles='dashed',color='r')
        >>> plt.vlines(x=3,ymin=0,ymax=lintrans(3,(2,4),(4,10)),linestyles='dashed',color='r')
        >>> plt.plot(3,lintrans(3,(2,4),(4,10)),'ro')
        >>> plt.show()
    """
    x1,x2=inboundtuple
    y1,y2=outboundtuple
    y=(float(y2-y1)/(x2-x1))*(x-x1)+y1
    return y

def arraylintrans(x,inboundtuple,outboundtuple,copy=False):
    """
    Warning!
    --------
    Latest information: 22/05/2012 this is mainly deprecated and plot_array_transg is used. please check the before use. 

    Purpose: make a linear transformation of a scalar,1Darray,2Darray for enhancing contrast.
             Suppose, the input data is x which lies in [x1,x2], we want to transform it into range [y1,y2] by a linear way.
    Arguments:
        inboundtuple --> the original bound tuple (x1,x2)
        outboundtuple --> the transformed bound tuple (y1,y2)
    Example:
        >>> import mathex.lintrans as lintrans
        >>> lintrans(3,(2,4),(4,8))
        >>> plt.plot(np.arange(2,5),lintrans(np.arange(2,5),(2,4),(4,10)))
        >>> plt.hlines(y=lintrans(3,(2,4),(4,10)),xmin=0,xmax=3,linestyles='dashed',color='r')
        >>> plt.vlines(x=3,ymin=0,ymax=lintrans(3,(2,4),(4,10)),linestyles='dashed',color='r')
        >>> plt.plot(3,lintrans(3,(2,4),(4,10)),'ro')
        >>> plt.show()
    """
    x=np.array(x).astype(float)
    x1,x2=inboundtuple
    y1,y2=outboundtuple
    indarray=np.transpose(np.nonzero((x>=x1)&(x<=x2)))
    if copy:
        y=pcopy.deepcopy(x)
        if np.ndim(x)==1:
            for ind in indarray:
                y[ind]=lintrans(y[ind],(x1,x2),(y1,y2))
        elif np.ndim(x)==2:
            for ind in indarray:
                y[ind[0],ind[1]]=lintrans(y[ind[0],ind[1]],(x1,x2),(y1,y2))
        else:
            raise ValueError('input array has more than 2 dimensions!')
        return y
    else:
        if np.ndim(x)==1:
            for ind in indarray:
                x[ind]=lintrans(x[ind],(x1,x2),(y1,y2))
        elif np.ndim(x)==2:
            for ind in indarray:
                x[ind[0],ind[1]]=lintrans(x[ind[0],ind[1]],(x1,x2),(y1,y2))
        else:
            raise ValueError('input array has more than 2 dimensions!')
        return x

def plot_array_transg(pdata,interval_original,copy=False,interval_target=None):
    """
    Purpose: transform the pdata array (2D array, mask or not) according to list a, this is mainly for constracting the data for mapping.
    Note:
        1. interval_original can be anything that can work as input for function pb.iteflat(), this is for easy use of many convenient ways to specify the constract
           of data from *interval_original* to *interval_target*. few examples:
               -- a single list [1,2,3,4], or any other forms of list (it's very flexible), list members can be scalars or iterables. Eg., [(1),(2),(3)],
                  [(1,2),(3,4),(5,6)], [1,2,3,4,np.arange(5,8,0.5),np.PINF],[(np.NINF),np.arange(-10,10.1,2),(np.PINF)],[1,[2,3,4],5]
               -- a tuple, eg. ((1,2),(3,4)), (1,2,3,4). Everthing above as list can also be used as tuple.
               -- a single np array, eg. np.arange(1,10).
        2. this function will replace the np.NINF and np.PINP in the interval_original to the minimum and maximum value in the input data.
        3. it's possible to impose the interval you want the data to be transferred to by set interval_target. Note the target list length must be equal
           to length of pb.iteflat(interval_original). if interval_target is None, interval_target will be:
           np.linspace(1,len(pb.iteflat(interval_original)),num=len(pb.iteflat(interval_original)),endpoint=True)
        4*. This function is originally developed for purpose of making map contrasting in contour(f) or image plots. it's called in *bmap.contourf* function if
           the keyword flag data_transform is set as True. So if interval_original is set in this purpose, make sure all the extreme values in interval_original
           are not repeated, i.e. the pb.iteflat(interval_original) constitutes a sequential list without identical members.
        5. For 2D array, this function has been extensively tested against plot_array_trans and it's working correctly.
    Return:
        return pdata_trans,interval_target,labellist,trans_base_list
        pdata_trans: transformed data;
        interval_target:in case of interval_target is None, return np.linspace(1,len(pb.iteflat(interval_original)),num=len(pb.iteflat(interval_original)),endpoint=True)
        labellist: pb.iteflat(interval_original) with np.NINF and np.PINF replaced.
        trans_base_list:the true original interval from which data is transformed.
    See also:
        bamp.contourf, pb.iteflat.
    Examples:
        >>> b=np.arange(-9,9.1,0.5)
        >>> pdata=np.ones((37,37))
        >>> for i in range(37):
                pdata[i]=b
        >>> a=[(-9, -4), (-1, -0.5, 0, 0.5, 1), (4, 9)]
        >>> mathex.plot_array_trans(pdata,a,copy=True)
            Out[104]: 
            (array([[-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   ..., 
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ]]),
             [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0],
             [-9, -4, -1, -0.5, 0, 0.5, 1, 4, 9])
        >>> mathex.plot_array_transg(pdata,a,copy=True,interval_target=[-2.0,-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0])
            Out[142]: 
            (array([[-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   ..., 
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ]]),
             [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0],
             [-9, -4, -1, -0.5, 0, 0.5, 1, 4, 9])

        >>> mathex.plot_array_transg(pdata,[(np.NINF),(-4, -1, -0.5, 0, 0.5, 1, 4),(np.PINF)],copy=True,interval_target=[-2.0,-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0])
            Out[148]: 
            (array([[-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   ..., 
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ]]),
             [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0],
             [-inf, -4, -1, -0.5, 0, 0.5, 1, 4, inf])

        >>> mathex.plot_array_transg(pdata,[(np.NINF),(-4, -1, -0.5, 0, 0.5, 1, 4),(9)],copy=True,interval_target=[-2.0,-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0])
            Out[149]: 
            (array([[-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   ..., 
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ]]),
             [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0],
             [-inf, -4, -1, -0.5, 0, 0.5, 1, 4, 9])

        >>> mathex.plot_array_transg(pdata,a,copy=True)
        Out[159]: 
        (array([[ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ],
                [ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ],
                [ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ],
                    ..., 
                [ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ],
                [ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ],
                [ 1. ,  1.1,  1.2, ...,  8.8,  8.9,  9. ]]),
                array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]),
                [-9, -4, -1, -0.5, 0, 0.5, 1, 4, 9])
    """
    if isinstance(pdata,list):
        pdata=np.array(pb.iteflat(pdata))
    if copy:
        pdata_trans=pcopy.deepcopy(pdata.astype(float))
    else:
        pdata_trans=pdata
    labellist=pb.iteflat(interval_original)
    trans_base_list=labellist[:]
    #change NINF and PINF to min&max of input data
    if np.isneginf(trans_base_list[0]):
        trans_base_list[0]=np.ma.min(pdata_trans)
    if np.isposinf(trans_base_list[-1]):
        trans_base_list[-1]=np.ma.max(pdata_trans)
    if interval_target is None:
        interval_target=np.linspace(1,len(trans_base_list),num=len(trans_base_list),endpoint=True)
    #2D array
    if np.ndim(pdata_trans)==2:
        for row in range(pdata_trans.shape[0]):
            for col in range(pdata_trans.shape[1]):
                for i in range(len(interval_target)-1):
                    if pdata_trans[row,col]>=trans_base_list[i] and pdata_trans[row,col]<=trans_base_list[i+1]:
                        pdata_trans[row,col]=lintrans(pdata_trans[row,col],(trans_base_list[i],trans_base_list[i+1]),(interval_target[i],interval_target[i+1]))
                        break  #use break to jump out of loop as the data have to be ONLY transferred ONCE.
    elif np.ndim(pdata_trans)==1:
        for col in range(pdata_trans.shape[0]):
            for i in range(len(interval_target)-1):
                if pdata_trans[col]>=trans_base_list[i] and pdata_trans[col]<=trans_base_list[i+1]:
                    pdata_trans[col]=lintrans(pdata_trans[col],(trans_base_list[i],trans_base_list[i+1]),(interval_target[i],interval_target[i+1]))
                    break  #use break to jump out of loop as the data have to be ONLY transferred ONCE.

    else:
        raise ValueError('test has only been done for 1D & 2D array data, the input data dimension is {0}'.format(np.ndim(pdata_trans)))
    return pdata_trans,interval_target,labellist,trans_base_list

def plot_array_transg_np(pdata,interval_original,copy=False,interval_target=None):
    """
    """
    if isinstance(pdata,list):
        pdata=np.array(pb.iteflat(pdata))
    if copy:
        pdata_trans=pcopy.deepcopy(pdata.astype(float))
    else:
        pdata_trans=pdata
    labellist=pb.iteflat(interval_original)
    trans_base_list=labellist[:]
    #change NINF and PINF to min&max of input data
    if np.isneginf(trans_base_list[0]):
        trans_base_list[0]=np.ma.min(pdata_trans)
    if np.isposinf(trans_base_list[-1]):
        trans_base_list[-1]=np.ma.max(pdata_trans)
    if interval_target is None:
        interval_target=np.linspace(1,len(trans_base_list),num=len(trans_base_list),endpoint=True)
    #2D array

    pdata_new = np.interp(pdata_trans,trans_base_list,interval_target)
    try:
        pdata_new = np.ma.masked_array(pdata_new,mask=pdata_trans.mask)
    except AttributeError:
        pass
    return pdata_new,interval_target,labellist,trans_base_list

#2012-11-20 test has been done to verify the plot_array_transg_np gives exactly the result by previous plot_array_transg, 
#and 100 times faster, so the new numpy function is used.
plot_array_transg=plot_array_transg_np

def plot_array_trans(pdata,a,copy=False):
    """
    Warning!!!
    ----------
    Latest Information: 22/05/2012 this is deprecated and plot_array_transg is used instead.

    Purpose:
    --------
    Transform array according to speficication in list a. return a copy if copy is True.

    Example:
    --------
        >>> b=np.arange(-9,9.1,0.5)
        >>> pdata=np.ones((37,37))
        >>> for i in range(37):
                pdata[i]=b
        >>> a=[(-9, -4), (-1, -0.5, 0, 0.5, 1), (4, 9)]
        >>> plot_array_trans(pdata,a)
            In [104]: plot_array_trans(pdata,a)
            Out[104]: 
            (array([[-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   ..., 
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ],
                   [-2.  , -1.95, -1.9 , ...,  1.9 ,  1.95,  2.  ]]),
             [-2.0, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.0],
             [-9, -4, -1, -0.5, 0, 0.5, 1, 4, 9])
    """

    if copy:
        pdata_trans=pcopy.deepcopy(pdata)
    else:
        pdata_trans=pdata
    low_range=a[0]
    mid_range=a[1]
    high_range=a[2]
    if len(mid_range)==1:
        raise ValueError('there is only 1 element in middle range!')
    else:
        interval=mid_range[1]-mid_range[0]
        #
        if isinstance(low_range,tuple):
            low_range_plot=pcopy.deepcopy(list(low_range))
        else:
            low_range_plot=pcopy.deepcopy(list([low_range]))
        for i in range(len(low_range_plot)):
            low_range_plot[i]=mid_range[0]-interval*(len(low_range_plot)-i)
    
        if isinstance(high_range,tuple):
            high_range_plot=pcopy.deepcopy(list(high_range))
        else:
            high_range_plot=pcopy.deepcopy(list([high_range]))
        for i in range(len(high_range_plot)):
            high_range_plot[i]=mid_range[-1]+interval*(i+1)
    
        if len(low_range_plot)==1:
            pdata_trans=arraylintrans(pdata_trans,(low_range,mid_range[0]),(low_range_plot[0],mid_range[0]))
        else:
            for i in range(len(low_range_plot))[::-1]:
                if i != len(low_range_plot)-1:
                    pdata_trans=arraylintrans(pdata_trans,(low_range[i],low_range[i+1]),(low_range_plot[i],low_range_plot[i+1]))
                else:
                    pdata_trans=arraylintrans(pdata_trans,(low_range[i],mid_range[0]),(low_range_plot[i],mid_range[0]))
    
        if len(high_range_plot)==1:
            pdata_trans=arraylintrans(pdata_trans,(mid_range[-1],high_range),(mid_range[-1],high_range_plot[0]))
        else:
            for i in range(len(high_range_plot)):
                if i ==0:
                    pdata_trans=arraylintrans(pdata_trans,(mid_range[-1],high_range[0]),(mid_range[-1],high_range_plot[0]))
                else:
                    pdata_trans=arraylintrans(pdata_trans,(high_range[i-1],high_range[i]),(high_range_plot[i-1],high_range_plot[i]))

        if not hasattr(low_range,'__iter__'):
            low_range=list([low_range])
        if not hasattr(high_range,'__iter__'):
            high_range=list([high_range])
        levtemp=[low_range_plot,mid_range,high_range_plot]
        levels=[j for i in levtemp for j in i]
        labtemp=[low_range,mid_range,high_range]
        lab=[j for i in labtemp for j in i]
    return pdata_trans,levels,lab

class ma:
    def rolling_window(a, window):
        '''
        the same function as rolling_window but work only with masked_array.
        the fill_value for the masked array should be np.nan
        '''
        return rolling_window(a.filled(),window)
    
    def move_ave(a,window):
        '''
    
        '''
        return move_ave(a.filled(),window)



def datapick(name,targetfield,*varlist):
    '''
    Purpose: pickup column data from DataFrame object by using keyword pairs and targetfiled, np.array([]) if returned if 
    Note:
        1. name: pandas Dataframe object; targetfield: the field that data are to be selected; varlist: tuple list that are filtering field name and value
        2. 
    Example:
        >>> mathex.datapick(cpsm,'forest floor carbon (gC/m2)',('SiteName','Saskathewan'),('data_source','Gower et al'))
        >>> In [64]: mathex.datapick(man2,'year',('year',(1998,2004)))
        Out[64]: 
        array([1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998,
               1998, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999, 1999,
               1999, 1999, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
               2000, 2000, 2000, 2001, 2001, 2001, 2001, 2001, 2001, 2001, 2001,
               2001, 2001, 2001, 2001, 2002, 2002, 2002, 2002, 2002, 2002, 2002,
               2002, 2002, 2002, 2002, 2002, 2003, 2003, 2003, 2003, 2003, 2003,
               2003, 2003, 2003, 2003, 2003, 2003, 2004, 2004, 2004, 2004, 2004,
               2004, 2004, 2004, 2004, 2004, 2004, 2004], dtype=int64)
        In [65]: mathex.datapick(man2,'year',('year',[1998,2004]))
        Out[65]: 
        array([1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998,
               1998, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004, 2004,
               2004, 2004], dtype=int64)

    '''
    select=np.asarray([True]*len(name[targetfield]))
    for pair in varlist:
        fieldname=pair[0]
        keynum=pair[1]
        subselect=[]
        if isinstance(keynum,str):
            for member in name[fieldname]:
                if isinstance(member,str):
                    if member.find(keynum)==-1:
                        subselect.append(False)
                    else:
                        subselect.append(True)
                else:
                    subselect.append(False)
        else:
            for member in name[fieldname]:
                #use interval or multi number to select/filter
                if hasattr(keynum,'__iter__'):
                    if isinstance(keynum,tuple) and len(keynum)==2:   #use low and high range to select in-between
                        subkey_low=keynum[0]
                        subkey_high=keynum[1]
                        if member >= subkey_low and member <= subkey_high:
                            subselect.append(True)
                        else:
                            subselect.append(False)
                    else:                                            #list out key number that are desired.
                        if member in keynum:
                            subselect.append(True)
                        else:
                            subselect.append(False)
                #use only a number 
                else:
                    if member==keynum:
                        subselect.append(True)
                    else:
                        subselect.append(False)
        select=np.column_stack((select,np.asarray(subselect)))  #subselect has lenth of len(name) with True/False as its member, the select combines all the subselect to find the mutual Trues.
    if select.ndim==1:
        selectfinal=select
    else:
        selectfinal=np.all(select,axis=1)
    targetarray=np.asarray(name[targetfield])
    if np.any(selectfinal)==False:
        return None
    else:
        return targetarray[selectfinal]


def DBHfromBA(ba,density):
    """
    Purpose: Calculate average DBH from basal area and tree density
    Definition: DBHfromBA(ba,density)=np.sqrt(40000*ba/(np.pi*density))
    """
    return np.sqrt(40000*ba/(np.pi*density))

def OLS_RegGrid(prmf,checkna=False,nobs_threshold=7):
    """
    Purpose: Do OLS linear regression across a grid against the index (ie. sequential time axis)

    Parameters:
    -----------
    prmf: should be a three dimension ndarray (eg. l X m X n), the
        regression is done for the grid mXn again time axis (axis=0).
        In case of a unmasked array, the regression will be done directly;
        In case of a masked array
         -- If checkna == False, it will assume the mask is the same
            across time axis, and the mask of prmf[0] will be used as the
            final result mask. In this case, there is no need to fill the
            input masked arrays with np.nan value, as it will be automatically
            done.
         -- If checkna == True, only unmasked values will be used as in regression
            along the axis 0, the returned array will have 4 as the length of the
            first dim, as (slope,R2,pvalue,validnum)
    nobs_threshold: The minimum valid sample size to do the regression.

    Returns:
    --------
    The result is a 3XmXn array in case of checkna=False or unmasked data,
        with the first dimension in the sequence of slope, R2, pvalue; The
        result is 4XmXn array in case of checkna=True, with the fourth as
        the number of valid values that parcipates in the regression.

    Example:
    >>> prmf.shape
        (56, 360, 720)
    >>> OLS_RegGrid(prmf)
    """
    if np.ma.isMA(prmf):
        if checkna == True:
            outarr = np.apply_along_axis(lambda x:_grid_linreg_checkna(x,nobs_threshold=nobs_threshold),0,prmf.filled(np.nan))
            outarr = np.ma.masked_invalid(outarr)
            return outarr
        else:
            valid_grid=np.nonzero(~prmf[0].mask)
            pretr=np.ma.array([prmf[0]]*3)
            for i,j in zip(valid_grid[0],valid_grid[1]):
                result=stats.linregress(np.arange(len(prmf[:,i,j])),prmf[:,i,j]) #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                pretr[:,i,j]=[result[0],result[2]**2,result[3]]  #slope,R2,p_value
            return pretr
    #not a masked array
    else:
        xsize,ysize = prmf.shape[1:]
        pretr=np.ma.array([prmf[0]]*3)
        for i in range(xsize):
            for j in range(ysize):
                result=stats.linregress(np.arange(len(prmf[:,i,j])),prmf[:,i,j]) #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                pretr[:,i,j]=[result[0],result[2]**2,result[3]]  #slope,R2,p_value
        return pretr

reggrid_ols = OLS_RegGrid

def reggrid_dic(dic,method='OLS',regstrlist=None):
    """
    Recieve a dic of multi-dim array, make the regression for each array,
        and return the regression result by separating R2,p-value,and slope.

    Parameters:
    -----------
    method: could be a string or function.
        For the case of string:
        'OLS': mathex.OLS_RegGrid function.
    """

    if regstrlist is None:
        regstrlist = ['slope','R2','p-value']

    if isinstance(method,str):
        if method == 'OLS':
            func = OLS_RegGrid
        else:
            raise ValueError("only accept OLS string currently")
    else:
        if callable(method):
            func = method
        else:
            raise TypeError("Unsupported type of method!")

    regdic = OrderedDict()
    for key,arr in dic.items():
        dtdic = {}
        reg_arr = func(arr)
        for i,tag in enumerate(regstrlist):
            dtdic[tag] = reg_arr[i]
        regdic[key]=dtdic

    return regdic

def m2ysum(cmi):
    """
    Transform the monthly array data to yearly sum/mean; the first dimension should be 12*n
    Handle masked array automatically. now can handly as much as 4dim data.
    """
    d0=cmi.shape
    if len(d0)==1:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==2:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==3:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],d0[2],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==4:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],d0[2],d0[3],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    else:
        raise ValueError('the array has more than 4 dimension!')
    return pb.Remove_dupdim(cmiyear)

def d2ysum(cmi):
    """
    Transform the monthly array data to yearly sum/mean; the first dimension should be 365*n
    Handle masked array automatically. 
    """
    d0=cmi.shape
    if len(d0)==1:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==2:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==3:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],d0[2],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==4:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],d0[2],d0[3],order='F')
            cmiyear=np.ma.sum(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    else:
        raise ValueError('the array has more than 4 dimension!')
    return cmiyear

def m2ymean(cmi):
    """
    Transform the monthly array data to yearly sum/mean; the first dimension should be 12*n
    Handle masked array automatically, now can handly as much as 4dim data.
    """
    d0=cmi.shape
    if len(d0)==1:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12)
            cmiyear=np.ma.mean(cminew,axis=0,order='F')
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==2:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==3:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],d0[2],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    elif len(d0)==4:
        if np.mod(d0[0],12)==0:
            cminew=cmi.reshape(12,d0[0]/12,d0[1],d0[2],d0[3],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 12!')
    else:
        raise ValueError('the array has more than 4 dimension!')
    return pb.Remove_dupdim(cmiyear)

def d2ymean(cmi):
    """
    Transform the monthly array data to yearly sum/mean; the first dimension should be 365*n
    Handle masked array automatically
    """
    d0=cmi.shape
    if len(d0)==1:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==2:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==3:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],d0[2],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    elif len(d0)==4:
        if np.mod(d0[0],365)==0:
            cminew=cmi.reshape(365,d0[0]/365,d0[1],d0[2],d0[3],order='F')
            cmiyear=np.ma.mean(cminew,axis=0)
        else:
            raise ValueError('the first dimension is not product of 365!')
    else:
        raise ValueError('the array has more than 4 dimension!')
    return cmiyear


def d2m(data,mode='sum'):
    """
    Transform the daily data to monthly data.

    Notes:
    ------
    1. Default calendar is noleap and the length of the first dimension
    """
    noleap = np.array([ 31,  59,  90, 120, 151, 181, 212, 243, 273, 304, 334, 365])

    if data.shape[0] == 365:
        indarr = noleap
    else:
        raise ValueError("Can only handle noleap data.")

    if mode == 'sum':
        func = lambda x:x.sum(axis=0)
    elif mode == 'mean':
        func = lambda x:x.mean(axis=0)
    else:
        raise ValueError("mode could only be sum or mean")

    snowf_daily_into_month = np.split(data,indarr,axis=0)[:-1]
    month_data_list = map(func,snowf_daily_into_month)
    snowf_month = np.ma.dstack(month_data_list)
    snowf_month = np.rollaxis(snowf_month,snowf_month.ndim-1,0)
    return snowf_month


def ncGet_OLS_Trend(infile,outfile,varlist='all',reginterval=None):
    """
    Purpose: Do OLS linear regression for varialbe list in the 'infile' and write the data to 'outfile' 
    Arguments:
        1. varlist=['var1','var2'] to specify the varialbes that are included in regression
        2. reinterval=[(10,55),(20,189)] to specify the regression start and end index for regression. None for all present length; if len(reinterval)==1,
           it will be broadcasted to have the same length of varlist, otherwise len(interval) have to be equal with len(varlist)
    Note: 
        1. the input netCDF file must have only 3 dimesions with the first dimension as the unlimited time dimension.
        2. the time dimension is the first dimension of the output variable and has a length of 3, in the sequence of slope, R2, pvalue.
        3. the function can handle masked array.
        4. the function copy the dimension name, variable name and data for the other 2 dimensions other than the time dimension direclty from the input file to output file.
        5. the regression use the stats.lineregrss function from scipy package. (it use OLS_RegGrid function in the mathex module)
        6. the output varialbe name is the original name appended by '_trend', eg. 'cmi' --> 'cmi_trend'
        7. by default, if varlist is 'all', trend for all the variables in input file will be calculated.

        8. Finally, this function has been test against R OLS regreesion to show it's valid.
    Example:
    >>> ncGet_OLS_Trend('in.nc','out.nc',varlist=['var1','var2'])
    """
    f1=nc.Dataset(infile,'r')
    if len(f1.dimensions) >3:
        raise ValueError('the input file has more than 3 dimensions!')

    #identity dimension varialbe and unlimited variable
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
    
    print 'Begin to calculate trend and write to netcdf file ',outfile
    rootgrp=nc.Dataset(outfile,'w',format='NETCDF3_CLASSIC')
    
    #creat dimension
    lat=rootgrp.createDimension(globe_limdimname1,len(limdimvar1_full))
    lon=rootgrp.createDimension(globe_limdimname2,len(limdimvar2_full))
    rootgrp.createDimension(globe_unlimdimname,None)
    print 'Dimensions ',globe_limdimname1,globe_limdimname2,globe_unlimdimname,' created'
    
    #creat dimension variable
    lat=rootgrp.createVariable(str(limdimvar1_full._name),limdimvar1_full.dtype,(globe_limdimname1,))
    lon=rootgrp.createVariable(str(limdimvar2_full._name),limdimvar2_full.dtype,(globe_limdimname2,))
    try:
        lat.long_name = limdimvar1_full.long_name.encode()
        lon.long_name = limdimvar2_full.long_name.encode()
    except AttributeError:
        pass
    lat.units=limdimvar1_full.units.encode()
    lon.units=limdimvar2_full.units.encode()
    lat[:]=limdimvar1_full[:].copy()
    lon[:]=limdimvar2_full[:].copy()
    time=rootgrp.createVariable(unlimdimvar_full._name,unlimdimvar_full.dtype,(globe_unlimdimname,))
    time[:]=np.arange(1,4)
    time.long_name = 'time'
    print 'Dimension variables ','--'+str(limdimvar1_full._name)+'--','--'+str(limdimvar2_full._name)+'--','--time-- created'

    #prepare for the variable list for regression
    varlist_all=[name.encode() for name in f1.variables.keys()]
    varlist_all.remove(str(limdimvar1_full._name))
    varlist_all.remove(str(limdimvar2_full._name))
    varlist_all.remove(str(unlimdimvar_full._name))
    if varlist=='all':
        varlist2=varlist_all
    else:
        varlist2=varlist

    #prepare the reg_start and reg_end index for the regression interval
    if reginterval is None:
        reg_start_end_index=[None]*len(varlist2)
    else:
        if not isinstance(reginterval,list):
            raise TypeError('please use a list to specify reginterval')
        else:
            if len(reginterval)==1:
                reg_start_end_index=reginterval*len(varlist2)
            else:
                if len(reginterval)!=len(varlist2):
                    raise ValueError('index list length is {0} and regression variablist list length is {1}'.format(len(reginterval),len(varlist2)))
                else:
                    reg_start_end_index=reginterval

    #creat ordinary variable
    for varindex,varname in enumerate(varlist2):
        var_full=f1.variables[varname]
        if str(var_full.dimensions[0])!=globe_unlimdimname:
            print 'the time dimension is not the first dimension for varialbe --',varname,'--!'
        #select only data required by reginterval
        if reg_start_end_index[varindex] is None:
            var_value=pcopy.deepcopy(var_full[:])
        else:
            var_value=pcopy.deepcopy(var_full[reg_start_end_index[varindex][0]:reg_start_end_index[varindex][1],:,:])
        #make regression
        var_value_trend=OLS_RegGrid(var_value)
        print 'Trend calculation for varialbe --',varname,'-- is done'

        vardata=(str(var_full._name)+'_trend',str(var_full._name)+'_trend',str(var_full.units)+' per timestep',var_value_trend)
        ba=rootgrp.createVariable(vardata[0],'f4',(str(var_full.dimensions[0]),str(var_full.dimensions[1]),str(var_full.dimensions[2]),))
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
    
        print 'Variable  --',vardata[0],'--  is fed into the file ','--',outfile,'--'
    
    f1.close()
    rootgrp.history='trend by time calculated from file '+'--'+infile+'--'
    rootgrp.close()

def ndarray_split(inarray,index,mode=None):
    """
    Purpose: split a ndarray by the first dimension by specifying index and make the transformation by the first dimension.
             Especially, it's useful for obtaining monthly sum from daily data.
    Note:
        1. it can handle only the first dimension of input array.
        2. index can be a list, tuple, or 1D array.
    Example:
        In [104]: air.shape
        Out[104]: (366, 94, 192)

        In [105]: noleap
        Out[105]: array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

        In [106]: air.shape
        Out[106]: (366, 94, 192)

        In [107]: leap
        Out[107]: array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

        In [108]: monair=array_split(air,leap,mode='mean')

        In [109]: monair.shape
        Out[109]: (12, 94, 192)
    """
    index=np.array(index)
    indexreal=np.cumsum(index)
    if inarray.shape[0]!=indexreal[-1]:
        raise ValueError('the dimension of first array does not equal to sum of interval numbers')
    else:
        indexreal=list(indexreal)
        indexreal.insert(0,0)
        d=[]
        for i in range(len(indexreal)-1):
            if mode=='mean':
                monthmean=np.ma.mean(inarray[indexreal[i]:indexreal[i+1],...],axis=0)
                d.append(monthmean)
            elif mode=='sum':
                monthmean=np.ma.sum(inarray[indexreal[i]:indexreal[i+1],...],axis=0)
                d.append(monthmean)
            elif mode is None:
                raise ValueError('mode can only be sum or mean')
            else:
                raise ValueError('mode can only be sum or mean')
        outarray=np.ma.array(d)
        return outarray

def CVRMSD(a,b):
    '''
    calculate the coefficient of Root Mean Square Deviation (Error) for two 1-dimension ndarray a & b
    a: model data; b: observation data.
    http://en.wikipedia.org/wiki/Root_mean_square_deviation

    Note: 1. handle masked array and Nan values (via masked array) automatically.
    '''
    a=pb.MaskArrayByNan(a)
    b=pb.MaskArrayByNan(b)
    if np.ma.isMA(a) and np.ma.isMA(b):
        diff=a-b 
        return (np.sqrt(np.ma.sum(np.square(a-b))/float(len(np.nonzero(~diff.mask)[0]))))/np.ma.mean(b)
    else:
        if True in np.isnan(a) or True in np.isnan(b):
            if True in np.isnan(a):
                a=np.ma.masked_array(a,mask=np.isnan(a))
                return CVRMSD(a,b)
            else:
                b=np.ma.masked_array(b,mask=np.isnan(b))
                return CVRMSD(a,b)
        else:
            return (np.sqrt(np.square(a-b).sum()/float(len(a))))/b.mean()

def RMSD(sim,obs):
    """
    Return Root Mean Square Deviation (Error).
    """
    a=sim
    b=obs
    a=pb.MaskArrayByNan(a)
    b=pb.MaskArrayByNan(b)
    a,b=pb.shared_unmask_data(a,b)
    rmsd=np.sqrt(np.sum(np.square(a-b))/len(b))
    return rmsd

def RMSD_sys_unbias_RTO(sim,obs):
    """
    Purpose: Calculate systematic and unbias RMSD by using RTO regression.
    Return:
        return rmsd_sys,rmsd_unbias
    """

    a=sim
    b=obs
    a=pb.MaskArrayByNan(a)
    b=pb.MaskArrayByNan(b)
    drydata=np.vstack((a,b))
    regres=linreg_RTO_OLS_2varyx(drydata[0],drydata[1])
    observe_predict=regres['slope']*drydata[1]
    rmsd_sys=RMSD(observe_predict,drydata[1])
    rmsd_unbias=RMSD(observe_predict,drydata[0])
    return rmsd_sys,rmsd_unbias


def get_mean_std(data,move_ave=None):
    """
    return 3Xn ndarray. with the 1st row as mean, the 2nd row as mean-std, the 3rd row as mean+std
    """
    if data.ndim !=2:
        raise ValueError ('please provide 2D array!')
    if move_ave is not None:
        data=mathex.move_ave2d(data,move_ave)
    dmean=np.ma.mean(data,axis=0)
    dstd=np.ma.std(data,axis=0)
    d=np.ma.vstack((dmean,dmean-dstd,dmean+dstd))
    return d

def get_mean_maxmin(data,move_ave=None):
    """
    return 3Xn ndarray. with the 1st row as mean, the 2nd row as max, the 3rd row as min
    """
    if data.ndim !=2:
        raise ValueError ('please provide 2D array!')
    if move_ave is not None:
        data=mathex.move_ave2d(data,move_ave)
    dmean=np.ma.mean(data,axis=0)
    dmax=np.ma.max(data,axis=0)
    dmin=np.ma.min(data,axis=0)
    d=np.ma.vstack((dmean,dmax,dmin))
    return d

def linreg_OLS_2var(x,y):
    """
    Purpose: This function use scipy.stats.linregress to reproduce simple linear regression in R.
    Note:
        1. x,y can be masked array, or array with nan as missing data, or input
            which could be converted to numpy array by using np.array function
        2. only 1D array is supported.
        3. x is independent variable while y is depedent variable, this follows argument posotions in scipy.stats.linregress function
           For a more intuitive way, linreg_OLS_2varyx is recommended.
    Return:
       return a pandas dataframe containing regression matrix and a dictionary containing information like R2, N, etc. 
    Test Example:
        In python:
        >>> comgpp=pa.read_csv('/home/chaoyue/python/testdata/comgpp.csv',sep=',')
        >>> res,res1=mathex.linreg_OLS_2var(comgpp.mod,comgpp.GEP_Amiro)
        >>> fig,ax=g.Create_1Axes()
        >>> ax.plot(comgpp.mod,comgpp.GEP_Amiro,'ro')
        >>> g.Set_Equal_XYlim(ax)
        >>> g.plot_OLS_reg(ax,comgpp.mod,comgpp.GEP_Amiro)
        >>> ax.plot(res1['xdata'],res1['predict'],'yo',ms=5)
        >>> ax.legend([ax.lines[0],ax.lines[2],ax.lines[3]],['observed','regression line','predicted'],loc='upper left')

        In R:
        data=read.csv('~/python/testdata/comgpp.csv',sep=',',header=TRUE)
        use <- !is.nan(data$GEP_Amiro)
        linfit=lm(data$GEP_Amiro ~ data$mod, subset=use)
        summary(linfit)
        confint(linfit)
        Results by R and python are the same and test has been done on 2012/05/07.
    Check also (supplemented 2012/06/05),
        source for the method to calclulate standard error.
        http://www.chem.utoronto.ca/coursenotes/analsci/StatsTutorial/ErrRegr.html
        Note the T value in this fuction is only for case that True Value = 0

    """
    if y.ndim>=2 or x.ndim>=2:
        raise ValueError('only 1D array could be supplied')
    else:
        y=np.ma.masked_invalid(y)
        x=np.ma.masked_invalid(x)
        x,y=pb.shared_unmask_data(x,y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        result=dict()
        result['Intercept']=dict()
        result['var']=dict()
        result['Intercept']['Estimated']=intercept
        result['var']['Estimated']=slope
        predict=x*slope+intercept
        SSe=np.sum((y-predict)**2)
        Sxx=np.sum((x-np.mean(x))**2)
        SSr=np.sum((predict-np.mean(y))**2)
        SSt=np.sum((y-np.mean(y))**2)
        result['Intercept']['Std.Error']=np.sqrt(SSe/(len(x)-2))*np.sqrt(np.sum(x**2)/(len(x)*Sxx))
        result['var']['Std.Error']=np.sqrt(SSe/(len(x)-2))/np.sqrt(Sxx)
        result['Intercept']['tvalue']=result['Intercept']['Estimated']/result['Intercept']['Std.Error']
        result['var']['tvalue']=result['var']['Estimated']/result['var']['Std.Error']
        result['Intercept']['Prob']=(1-stats.t.cdf(abs(result['Intercept']['tvalue']),len(x)-2))*2
        result['var']['Prob']=(1-stats.t.cdf(abs(result['var']['tvalue']),len(x)-2))*2
        res1=dict()
        res1['SSe']=SSe
        res1['Sxx']=Sxx
        res1['SSr']=SSr
        res1['SSt']=SSt
        Rsqure=r_value**2
        AdjustRsqure=1-(1-Rsqure)*(len(x)-1)/(len(x)-2)
        res1['R2']=Rsqure
        res1['AdjustR2']=AdjustRsqure
        res1['r_value'] = r_value
        res1['size']=len(x)
        res1['predict']=predict
        res1['xdata']=x
        t025right=stats.t.ppf(0.975,len(x)-2)
        t05right=stats.t.ppf(0.95,len(x)-2)
        result['var']['CI95']=[slope-result['var']['Std.Error']*t025right,slope+result['var']['Std.Error']*t025right]
        result['var']['CI90']=[slope-result['var']['Std.Error']*t05right,slope+result['var']['Std.Error']*t05right]
        result['Intercept']['CI95']=[intercept-result['Intercept']['Std.Error']*t025right,intercept+result['Intercept']['Std.Error']*t025right]
        result['Intercept']['CI90']=[intercept-result['Intercept']['Std.Error']*t05right,intercept+result['Intercept']['Std.Error']*t05right]
        result=pa.DataFrame(result)
        return result,res1

def Pdata_OLS_regression(pd):
    """
    Get OLS linear regression information for Pdata object pd for (y~x)
        for each tag, using linreg_OLS_2var

    Returns:
    -------
    A pandas dataframe of regression information.
    """
    namelist = ['Slope','Intercept','r_value','R2','Adjusted_R2','p_value','N']
    dic = OrderedDict()
    for tag in pd._taglist:
        datadic = pd.data[tag]
        result,res1 = linreg_OLS_2var(datadic['x'],datadic['y'])
        reslist = [result.ix['Estimated']['var'],
                   result.ix['Estimated']['Intercept'],
                   res1['r_value'],res1['R2'],res1['AdjustR2'],
                   result.ix['Prob']['var'],res1['size']]
        dic[tag] = pa.Series(reslist,index=namelist)
    return pa.DataFrame(dic)

def NestPdata_OLS_regression(npd):
    """
    Get OLS linear regression information for NestPdata object for
        (y~x) for each pair of (ptag,ctag).

    Returns:
    -------
    A pandas panel of regression information.
    """
    dic = OrderedDict()
    for ptag in npd.parent_tags:
        dic[ptag] = Pdata_OLS_regression(npd.child_pdata[ptag])
    panel = pa.Panel(dic)
    panel = panel.swapaxes(0,2)
    return panel

def linreg_OLS_2varyx(y,x=None):
    """
    Purpose: This is a simple wrapper of linreg_OLS_2var, the difference is that it puts argument for depedent variable before independent variable.
    """
    if x is None:
        x = np.arange(len(y))
    return linreg_OLS_2var(x,y)

def linreg_RTO_OLS_2varyx(y,x):
    """
    Purpose: do OLS regression line for y~x through origin. RTO is short for "Regression Through Origin", it used lm model in R.
    Note:
        1. being able to automatically mask np.nan  
        2. only 2 variables are tested.
        3. r_value is square root of R2.
        4. It's been tested that in R, the R2 for RTO regression is calculated using Eq(3') in Joseph G. Eisenhauer 2003. 
           (see '/home/chaoyue/python/testdata/RTO_check.R' for more details.)
    Return:
        a dictionary with keys: ['slope', 'std_err', 'p_value', 'N', 't_value', 'r_value']
    """
    #extract data that are not NaN or masked
    xnew=pb.MaskArrayByNan(x)
    ynew=pb.MaskArrayByNan(y)
    x_reg,y_reg=pb.shared_unmask_data(xnew,ynew)
    #use R to do regression
    rpy.r.assign('x_reg',x_reg)
    rpy.r.assign('y_reg',y_reg)
    lmRTO=rpy.r('lmRTO=lm(y_reg ~ x_reg-1)')  #Regression through origin
    summary_lmRTO=rpy.r('summary_lmRTO=summary(lmRTO)')
    #creat result dictionary 
    resdic={}
      #note the 1st, 2nd, 3rd, 4th elements for summary_lmRTO['coefficients'] is Estimate; Std. Error; t value; Pr(>|t|)
    resdic['slope']=summary_lmRTO['coefficients'][0][0]
    resdic['std_err']=summary_lmRTO['coefficients'][0][1]
    resdic['t_value']=summary_lmRTO['coefficients'][0][2]
    resdic['p_value']=summary_lmRTO['coefficients'][0][3]
    resdic['r_value']=np.sqrt(summary_lmRTO['r.squared'])
    resdic['N']=len(x_reg)
    return resdic

def com_array_by_plot(array1, array2, **plt_kwargs):
    arr1 = np.ma.ravel(array1)
    arr2 = np.ma.ravel(array2)
    fig, ax = g.Create_1Axes()
    ax.plot(arr1,arr2,'ro',mfc='none',**plt_kwargs)
    return fig,ax

def group_sequential(a):
    """
    Purpose: Group the numerical list or 1D array by sequential blocks.
    Return: a list of list.
    Example:
        >>> a=[2,3,4,67,78,79,102]
        >>> print mathex.group_sequential(a)
        >>> a=[2,3,4,67,78,79,102,103]
        >>> print mathex.group_sequential(a)
        >>> a=[0,3,4,67,78,79,102,103]
        >>> print mathex.group_sequential(a)
        >>> a=[0,3,4,67,78,79,102,103,120]
        >>> print mathex.group_sequential(a)

    """
    if isinstance(a,int):
        raise TypeError('Only list or 1D numpy array is accepted')
    else:
        if len(a)==1:
            return [[a[0]]]  #to make sure for np.array(5) or [5], [[5]] is returned.
        else:
            ind_begin=0
            d=[]
            for i in range(len(a)-1):
                if a[i+1]==a[i]+1:
                    #handle the case if last several members of a are connceted; such as [23,24,25,56,57,89,103,104]
                    if i==len(a)-2:
                        d.append(list(a[ind_begin:len(a)]))
                else: 
                    ind_end=i
                    d.append(list(a[ind_begin:ind_end+1]))
                    ind_begin=i+1
                    #handle the case when the last one element is not the sequential for its previous one; such as [23,24,25,56,57,89,103] 
                    if i==len(a)-2:
                        d.append([a[i+1]])
            return d

def file_add_data_files(filelist,filetype='blankspace'):
    """
    This is to add several data files directly and return a nparray.
    """
    final_array_list = []
    for filename in filelist:
        if filetype == 'blankspace':
            temparr = np.genfromtxt(filename)
        else:
            raise TypeError("please use filetype flag to specify type of the input file!")
        final_array_list.append(temparr[np.newaxis,...])

    outarr_temp = np.ma.concatenate(final_array_list, axis=0)
    outarr = np.ma.sum(outarr_temp,axis=0)
    return outarr


def file_compare_files_sum(filelist1,filelist2,filetype='blankspace'):
    """
    This is built on file_add_data_files to compare and plot the two list of files by checking if their respective summing is equal.

    Notes
    -----
    Now only the 1Darray is allowed to contained in each file
    """
    outarr1 = file_add_data_files(filelist1,filetype=filetype)
    outarr2 = file_add_data_files(filelist2,filetype=filetype)
    fig, (ax1,ax2) = g.Create_2VAxes()
    ax1.plot(outarr1,label='filelist1')
    ax1.plot(outarr2,label='filelist2')
    ax1.legend()
    ax2.plot(outarr1-outarr2, label='difference')
    ax2.legend()
    return (outarr1, outarr2, fig)


def ndarray_apply_mask(indata,mask):
    """
    Apply a mask to array indata.

    Notes
    -----
    1. mask is a boolean array
    """
    ones = np.ones(indata.shape)
    newmask = ones*mask
    return np.ma.masked_array(indata,mask=newmask)

def ndarray_duplicate_element_by_array(arr1,arr2):
    """
    Duplicate each element in arr1[i] by the corresponding value of
        arr2[i], if arr2[i]==0, then arr1[i] will be dropped in final
        output, if all elements of arr2 is zero, then a None value
        will be returned.

    Parameters:
    -----------
    arr1,arr2: 1-dim ndarray with equal length, arr2 must be interger type.
    """
    if len(arr1.shape) > 1 or len(arr2.shape) > 1:
        raise TypeError("could only be 1-dim array")
    elif len(arr1) != len(arr2):
        raise TypeError("the length not equal for two input array")
    else:
        if not issubclass(arr2.dtype.type,np.integer):
            raise TypeError("arr2 is not integer type!")
        else:
            clist = []
            for i,j in zip(arr1,arr2):
                if j == 0:
                    pass
                else:
                    clist.append(np.array([i]*j))
            if clist == []:
                return None
            else:
                return np.concatenate(clist)

def ndarray_multi_func(func,*arrays):
    """
    Apply func (such as np.add, np.multiply, np.subtract, np.divide) recursively to arrays.

    Parameters:
    -----------
    func: currently only can be np.add, np.multiply, np.subtract, np.divide; np.ma corresonding fucntions doesn't work.
    """
    result = func(arrays[0],arrays[1])
    for arr in arrays[2:]:
        func(result, arr, result)
    return result


def ndarray_group_txt_by_func(filelist,outputfile=None,func=None,fmt='%.18e', delimiter=' ', newline='\n'):
    """
    Apply a function on the file1 and file2 to generate outputfile.

    Parameters:
    -----------
    func: a python function that use the ndarray from file1 and file2 as inputs.
    outputfile: if outputfile is None, the derived numpy ndarray will be returned, otherwise write the ndarray into output file.

    Tests:
    ------
    Test against the gnc.txt2nc_HalfDegree and then use the gnc.Ncdata.combine_vars and np.sum(axis=0); the two methods give the same results.

    >>> varlist = ['agriculture','peat','woodland','forest','deforestation','savanna']
    >>> filelist = ['/homel/ychao/python/testdata/GFED3.1_200409_C_'+var+'.txt' for var in varlist]
    >>> gnc.txt2nc_HalfDegree('/homel/ychao/python/testdata/test2.nc',['agriculture','peat','woodland','forest','deforestation','savanna'],'/homel/ychao/python/testdata/GFED3.1_200409_C_',land_mask=False)
    >>> d = gnc.Ncdata('/homel/ychao/python/testdata/test2.nc')
    >>> allvar = d.combine_vars(varlist)
    >>> allvarsum = np.sum(allvar,axis=0)
    >>> mathex.ndarray_group_txt_by_func(filelist,'/homel/ychao/python/testdata/allvar.txt',func=np.add,fmt='%.8e')
    >>> allvarnew = np.genfromtxt('/homel/ychao/python/testdata/allvar.txt')
    >>> print np.allclose(allvarsum,allvarnew)
    """
    array_list = [np.genfromtxt(filename) for filename in filelist]
    arr = ndarray_multi_func(func,*array_list)
    if outputfile is None:
        return arr
    else:
        np.savetxt(outputfile, arr, fmt=fmt, delimiter=delimiter, newline=newline)


def ndarray_mask_smart_apply(indata,mask):
    """
    This "smart" mask apply can handle only the situation of ndim of indata
        is 2/3/4 and the mask ndim is 2 or equal of indata ndim.
    """
    if indata.ndim not in [2,3,4]:
        raise ValueError("could only handle ndim of 2/3/4 for input data")
    else:
        if mask.ndim > indata.ndim:
            raise ValueError("mask.ndim bigger than indata.ndim")
        elif mask.ndim == indata.ndim:
            return np.ma.masked_array(indata,mask=mask)
        else:
            if mask.ndim != 2:
                raise ValueError("""mask.ndim could only be 2 in case of
                                    indata.ndim != mask.ndim""")

            else:
                inshape = indata.shape
                if mask.shape != (inshape[-2],inshape[-1]):
                    raise ValueError("""mask shape must be the same as the 
                                        last 2-dim shape of indata""")
                else:
                    if indata.ndim == 3:
                        mask_new = np.tile(mask,(inshape[0],1,1))
                    elif indata.ndim == 4:
                        mask_new = np.tile(mask,(inshape[0],inshape[1],1,1))
                    else: #not necessary as the ndim could only be 3/4
                        pass
                    return np.ma.masked_array(indata,mask=mask_new)

def ndarray_mask_by_threshold(pdata,map_threshold):
    '''
    Mask a ndarray by map_threhold

    Parameters:
    -----------
    map_threshold --> dictionary like {'lb':2000,'ub':5000}, data
        less than 2000 and greater than 5000 will be masked.
    '''
    #mask by map_threshold
    if map_threshold is None:
        return pdata
    else:
        if not isinstance(map_threshold,dict):
            raise ValueError('please provide a dictionary for map_threshold')
        else:
            for bound_key in map_threshold.keys():
                if bound_key not in ['lb','ub']:
                    raise ValueError ('Incorrect key is used.')
            if len(map_threshold.keys())==1:
                if map_threshold.keys()[0]=='lb':
                    lower_bound=map_threshold['lb']
                    pdata=np.ma.masked_less(pdata,lower_bound)
                elif map_threshold.keys()[0]=='ub':
                    upper_bound=map_threshold['ub']
                    pdata=np.ma.masked_greater(pdata,upper_bound)
            else:
                lower_bound=map_threshold['lb']
                upper_bound=map_threshold['ub']
                pdata=np.ma.masked_where(np.logical_not((pdata>lower_bound)&(pdata<upper_bound)),pdata)
        return pdata


def ndarray_get_index_by_interval(array,interval,left_close=True,
        right_close=True):
    """
    Rturn the index of the array members who fall in the given interval.

    Parameters:
    -----------
    array: only one-dimensional array
    interval: 2-len tuple
    left_close/right_close: To indicate the interval brink point.
    """
    if interval[0] > interval[1]:
        interval = interval[::-1]

    if left_close and right_close:
        extract_slice = np.nonzero((array >= interval[0])&(array <= interval[1]))
    elif left_close:
        extract_slice = np.nonzero((array >= interval[0])&(array < interval[1]))
    elif right_close:
        extract_slice = np.nonzero((array > interval[0])&(array <= interval[1]))
    else:
        extract_slice = np.nonzero((array > interval[0])&(array < interval[1]))
    return extract_slice

def ndarray_categorize_data(array,interval_sequence,numeric_output=False,
                            force_keys=None):
    """
    Return an string array corresponding which is the result of categorizing
        the input array by the given interval sequence.

    Notes:
    ------
    1. the interval_range is supposed to include the min and max values of
        the given array, otherwise all the array values falling outside
        interval_range will be untreated. Except for the last two elements
        in the interval_sequence, all the elements before them form a
        close-open interval, i.e., the interval is like
        [a_1,a_2),[a_2,a_3),...,[a_{n-2},a_{n-1}),[a_{n-1},a_{n}].

    Parameters:
    -----------
    1. numeric_output: boolean type. If it's True, then the output
        array will be integer array containing 1--> number of intervals
    2. force_keys: when givevn a list of strings, this will be used as
        the categorical values of the array.

    Returns:
    --------
    (interval_list,array): interval_list is the string list of intervals,
        and array is the categorized data.
    """
    if array.ndim > 1:
        raise ValueError("only accept one dimenional array!")
    else:
        if force_keys is None:
            keys = [str(interval_sequence[i])+'-'+str(interval_sequence[i+1]) for
                    i in range(len(interval_sequence)-1)]
        else:
            keys = force_keys[:]
        maxlen = max(map(len,keys))
        outarray = np.empty_like(array,dtype='S'+str(maxlen))
        numeric_outarray = np.ones(array.shape,dtype=int)

        for i in range(len(interval_sequence)-1):
            if i != len(interval_sequence)-2:
                idx = ndarray_get_index_by_interval(array,
                    (interval_sequence[i],interval_sequence[i+1]),
                    left_close=True,right_close=False)
            else:
                idx = ndarray_get_index_by_interval(array,
                    (interval_sequence[i],interval_sequence[i+1]),
                    left_close=True,right_close=True)
            outarray[idx] = keys[i]
            numeric_outarray[idx] = i+1

        if numeric_output:
            return (keys,numeric_outarray)
        else:
            return (keys,outarray)

def np_get_index(ndim,axis,slice_number):
    """
    Construct an index for used in slicing numpy array by specifying the
        axis and the slice in the axis.

    Parameters:
    -----------
    1. axis: the axis of the array.
    2. slice_number: the 0-based slice number in the axis.
    3. ndim: the ndim of the ndarray.
    """
    full_idx = [slice(None)] * ndim
    full_idx[axis] = slice_number
    return full_idx

def ndarray_mask_equal(ndarray,axis=0):
    """
    Check if the mask along the specified axis for all subarrays are equal.
    """
    if not np.ma.isMA(ndarray):
        raise ValueError("input array is not mask array")
    else:
        base_array = ndarray[np_get_index(ndarray.ndim,axis,0)]
        for i in range(1,ndarray.shape[axis]):
            idx = np_get_index(ndarray.ndim,axis,i)
            try:
                if np.array_equal(base_array.mask,ndarray[idx].mask):
                    pass
                else:
                    return False
            except AttributeError:
                return False
    return True

def ndarray_arraylist_equal_shape(array_list):
    """
    check if all the arrays in the list share the same shape.
    """
    shapelist = map(np.shape,array_list)
    for i in range(1,len(shapelist)):
        if shapelist[0] == shapelist[i]:
            pass
        else:
            return False
    return True


def ndarray_index_continous_True_block(arr):
    """
    Return the beginning and end index (0-based) for the consecutive True block
        in the input boolean array arr. Note the end index indicates the exact
        position where the True blocks stop, so the True block should be indexed
        as arr[beginning_index:end_index+1]

    Parameters:
    -----------
    arr: a boolean array.

    Returns:
    --------
    nX2 array, with 1st column indicating beginning index, the 2nd column
        indicating the end index. Return an empty array for if all the values
        in the input arr are False.

    Notes:
    ------
    This method is developped in the use case of trying to find the consecutive
        fire occurrence days from the model output.

    Examples:
    ---------
    In [14]: from mathex import ndarray_index_continous_True_block

    In [15]: arr = np.array([True,True,False,False,True,False,False])
    In [16]: ndarray_index_continous_True_block(arr)
    Out[16]: 
    array([[0, 1],
           [4, 4]])
    In [17]: arr = np.array([False,True,True,False,False,True,False,False])
    In [18]: ndarray_index_continous_True_block(arr)
    Out[18]: 
    array([[1, 2],
           [5, 5]])
    In [19]: arr = np.array([False,False,False,False,False,False,False,False])
    In [20]: ndarray_index_continous_True_block(arr)
    Out[20]: array([], shape=(0, 2), dtype=int64)

    In [21]: arr = np.array([True,True,True,True])
    In [22]: ndarray_index_continous_True_block(arr)
    Out[22]: array([[0, 3]])

    In [23]: arr = np.array([True,True,False,False,True,False,False,True])
    In [24]: ndarray_index_continous_True_block(arr)
    Out[24]: 
    array([[0, 1],
           [4, 4],
           [7, 7]])

    In [25]: arr = np.array([True,True,False,False,True,False,False,True,True])
    In [26]: ndarray_index_continous_True_block(arr)
    Out[26]: 
    array([[0, 1],
           [4, 4],
           [7, 8]])

    In [27]: arr[7:8]
    Out[27]: array([ True], dtype=bool)
    In [28]: arr[7:9]
    Out[28]: array([ True,  True], dtype=bool)
    """
    fireON = np.concatenate([[False],arr,[False]])
    start = np.ones(len(fireON),dtype=int) * (-1)
    end = start.copy()

    for i in range(1,len(fireON)):
        if fireON[i] == True:
            #FT?
            if fireON[i-1] == False:
                start[i] = 1
                #FTT
                if fireON[i+1] == True:
                    end[i] = 0
                #FTF
                else:
                    end[i] = 1
            #TT?
            else:
                start[i] = 0
                #TTT
                if fireON[i+1] == True:
                    end[i] = 0
                #TTF
                else:
                    end[i] = 1
        else:
            pass

    start_plus_end = start+end
    fire_PluralDay_index = np.nonzero(start_plus_end==1)[0] - 1
    fire_SingleDay_index = np.nonzero(start_plus_end==2)[0] - 1
    fire_index = np.concatenate([fire_PluralDay_index.reshape(-1,2),np.tile(fire_SingleDay_index,(1,2)).reshape(-1,2,order='F')])
    fire_index = np.sort(fire_index,axis=0)
    return fire_index

def ndarray_check_multipeak(data):
    """
    Check if the data has more than 1 peak value.

    Examples:
    ---------
    #case 1:
    data = np.array([1,2,3,4,5])
    plot(data,drawstyle='steps-pre',marker='o')
    print ndarray_check_multipeak(data)

    #case 2:
    data = np.array([1,1,1])
    plot(data,drawstyle='steps-pre',marker='o')
    print ndarray_check_multipeak(data)

    #case 3:
    data = np.array([5,4,3,2,1])
    plot(data,drawstyle='steps-pre',marker='o')
    print ndarray_check_multipeak(data)

    #case 4:
    data = np.array([1,1,2,1,1])
    plot(data,drawstyle='steps-pre',marker='o')
    print ndarray_check_multipeak(data)
    """
    valmin = data.min()
    valmax = data.max()
    for val in range(valmin,valmax+1):
        dt = data - val
        arr_index = ndarray_index_continous_True_block(dt>=0)
        if arr_index.shape[0] > 1:
            return True
    return False

def ndarray_regrid(arr,numlat=2,numlon=2,weight=None,dense=False):
    """
    Regrid an array by simple mean method.

    Parameters:
    -----------
    dense: True to make the grid denser, False for more sparse.
    weight: the weight factor when making the grid more sparse, can only
        be 2-dim array. weight is not used when dense=True
    """
    if np.ndim(arr) == 3:
        shape = arr.shape
        if dense:
            dtmp1 = np.tile(arr[:,np.newaxis,:,np.newaxis,:],[1,numlat,1,numlon,1])

            dtmp2 = dtmp1.reshape(shape[0],numlat*shape[1],numlon*shape[2],order='F')
            return dtmp2
        else:
            if weight is None:
                dtmp1 = arr.reshape(shape[0],-1,shape[1]/numlat,shape[2],order='F').mean(axis=1)
                dtmp2 = dtmp1.reshape(shape[0],shape[1]/numlat,numlon,shape[2]/numlon,order='F').mean(axis=2)
                return dtmp2
            elif isinstance(weight,np.ndarray):
                if np.ndim(weight) != 2:
                    raise ValueError("Can only accommodate 2-dim ndarray as weight")
                else:
                    wshape = weight.shape
                    weight_latlon = weight.reshape(numlat,wshape[0]/numlat,numlon,wshape[1]/numlon,order='F')
                    dtmp_latlon = arr.reshape(shape[0],numlat,shape[1]/numlat,numlon,shape[2]/numlon,order='F')
                    dtmp = (dtmp_latlon*weight_latlon).sum(axis=(1,3))/weight_latlon.sum(axis=(0,2))
                    return dtmp
            else:
                raise ValueError("Unknow weight")

    elif np.ndim(arr) == 2:
        dt = ndarray_regrid(arr[np.newaxis,...],numlat=numlat,numlon=numlon,
                            weight=weight,dense=dense)
        return dt[0]
    elif np.ndim(arr) == 4:
        dt = np.zeros((arr.shape[0],arr.shape[1],arr.shape[2]/numlat,arr.shape[3]/numlon))
        for i in range(arr.shape[0]):
            dt[i] = ndarray_regrid(arr[i],numlat=numlat,numlon=numlon,
                                   weight=weight,dense=dense)
        return dt
    else:
        raise ValueError("ndim of input array higher than 4, not implemented!")

def ndr_grid_convert(arr,shape,area_weight=True,area_frac=None):
    """
    Change the arr from its original grid to the new grid with 'shape'. Shape
        refers only to (lat,lon) and should be of 2-len tuple.

    Parameters:
    ----------
    arr: Input array that are to be re-gridded, could of dim 2, 3, 4, with last
        two dims as lat/lon.
    shape: new shape of grid, in (numlat,numlon).
    area_weight: True to use weight in the regriding. Defualt is to use the
        whole grid cell area (including land and ocean) calculated by
        `geo_area_latlon` over the common finer grid.
    area_frac: Sometimes, for example, we want to use the vegetation area
        as the regridding factor, we have to first regrid the vegetation
        area fraction onto a finer grid, then being multiplied by the default
        whole grid cell area, to get the finer vegetation area (over the finer
        grid), to be used as weighting factor. This array should have the same
        (lat,lon) as the input array.
    """
    import fractions
    def lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0

    if len(shape) != 2:
        raise ValueError("Shape must have length 2")
    else:
        orgshape = (arr.shape[-2],arr.shape[-1])

        lenlatCom = lcm(orgshape[0],shape[0])
        lenlonCom = lcm(orgshape[1],shape[1])

        # we first put the original data on a finer grid for futher
        # being made on the final (coarse) grid
        arr_fine = ndarray_regrid(arr,numlat=lenlatCom/orgshape[0],
                                  numlon=lenlonCom/orgshape[1],dense=True)

        if not area_weight:
            weight=None
        else:
            # calculate the default whole grid cell area
            area_whole_grid = geo_area_latlon(lenlatCom,lenlonCom)

            if area_frac is None:
                weight = area_whole_grid
            else:
                if area_frac.shape[0] != orgshape[0] or area_frac.shape[1] != orgshape[1]:
                    raise ValueError("area_frac must have the same latlon shape as input array")
                else:
                    area_frac_fine = ndarray_regrid(area_frac,numlat=lenlatCom/orgshape[0],
                                    numlon=lenlonCom/orgshape[1],dense=True)
                    weight = area_frac_fine * area_whole_grid

        # make the final regriding on the coarse new grid.
        arr_coarse = ndarray_regrid(arr_fine,numlat=lenlatCom/shape[0],
                        numlon=lenlonCom/shape[1],weight=weight)

        return arr_coarse


def dataframe_reg_as_panel(df,against='index',regfunc='OLS'):
    """
    Perform regression for each of the columns of the dataframe, and return
        the regression result as panel.

    Parameters:
    -----------
    against: the variable used as the independent variable in the regression,
        default is the index values of the dataframe.
    regfunc: the regression function used to do the regression.
    """
    if against == 'index':
        against = df.index.values

    if regfunc == 'OLS':
        regfunc = lambda yxlist:linreg_OLS_2varyx(yxlist[0],yxlist[1])[0]

    dic = OrderedDict()
    for varname in df.columns:
        dic[varname] = regfunc([df[varname].values,against])

    return pa.Panel(dic)


def dataframe_by_flatting_arraydict(array_dic):
    """
    Create a DataFrame by flating input ndarrays and use their keys as
        the DataFrame colomn names.

    Notes:
    ------
    1. the masked elements in ndarray will be changed into NaN when being
        converted to dataframe.
    """
    if ndarray_arraylist_equal_shape(array_dic.values()):
        return pa.DataFrame(pb.Dic_Apply_Func(lambda arr:arr.flatten(),array_dic))
    else:
        raise ValueError("arrays for input dict do not shape same shape!")

def dataframe_extract_statistic_info(df,target_field=None,groupby_field='geoindex'):
    """
    Extract information for target_field in dataframe "df" grouped by
        groupby_field.

    Returns:
    --------
    dataframe with the groupby_field as index.

    Arguments:
    ----------
    1. target_field: the field for statistical information. Could be provided
        as a string or a list of strings.
    2. groupby_field: the field used to group data.

    Returns:
    --------
    Dataframe if target_field is a string; or pandas Panel if target_field
        is a list of strings.

    Notes:
    ------
    1. extracted statstical information are:
        sum, mean, std, 95percentile, 50percentile(median), 5percentile
    2. the groupby method will drop the null row in the groupby_field
        automatically.
    """
    df_size_grp = df.groupby(groupby_field)

    if isinstance(target_field,str):
        fieldlist = [target_field]
    elif isinstance(target_field,list):
        fieldlist = target_field[:]
    else:
        raise TypeError("target_field must be a string or a list of str")

    resdic = OrderedDict()
    for subfield in fieldlist:
        dft = df_size_grp[subfield].agg([np.sum,np.mean,np.std])
        df_number = df_size_grp.size()
        dft['number'] = df_number

        dflist = []
        statnamelist = ['per95','median','per05']
        perlist = [95,50,5]
        func_dict = {}
        for name,per in zip(statnamelist,perlist):
            dflist.append(df_size_grp[subfield].agg({name:lambda x: np.percentile(x.values,per)}))
        dflist.extend([dft])
        dfall = pa.concat(dflist,axis=1)
        resdic[subfield] = dfall

    if isinstance(target_field,str):
        return resdic[target_field]
    else:
        return pa.Panel(resdic)


def dataframe_to_panel_by_colgroup(df,keylist,prefix='',surfix=''):
    """
    transfer a dataframe into panel by separating the columns into different
        groups by keywords.

    Parameters:
    -----------
    keylist: a list of strings indicating the keylists used to group the
        columns.
    """
    if not isinstance(keylist,(list,tuple,np.ndarray)):
        raise TypeError("wrong keylist type")
    else:
        dft = df.transpose()
        gplist = []
        for subcol in dft.index.values:
            for key in keylist:
                if key in subcol:
                    gplist.append(key)
                    break
        dftgp = dft.groupby(gplist)
        ordic = dict(iter(dftgp))
        dic = {}
        for key,dft in ordic.items():
            dft = dft.transpose()
            dic[key] = dft.rename(columns=lambda s:s.replace(prefix+key+surfix,''))
        return pa.Panel(dic)
        #return dic



def dataframe_change_geoindex_to_tuple(df):
    """
    change the string type of geoindex to tuple. The dataframe will be changed
        in place but a new dataframe will also be returned.
    """
    #change geoindex from string to tuple
    def change_string_to_tuple(dt):
        if not isinstance(dt,str):
            pass
        else:
            s1,s2 = dt.split(',')
            return (int(s1[1:]),int(s2[1:-1]))

    for i in df.index:
        dt = df['geoindex'][i]
        df['geoindex'][i] = change_string_to_tuple(dt)
    return df

def dataframe_remove_monthly_mean(dft):
    """
    Remove monthly mean for a monthly time-step dataframe.
    """
    dft_mon_mean = dft.groupby(lambda x:x.month).agg(np.mean)
    df_mon_mean_rep = pa.DataFrame(np.tile(dft_mon_mean.values,(len(dft)/12,1)),columns=dft.columns,index=dft.index)
    dft_mon_anomaly = dft - df_mon_mean_rep
    return dft_mon_anomaly

def ndarray_string_categorize(array,mapdict):
    """
    Recategorize a 1D string array by the dictionary which indicates the
        mapping between old strings and new strings.

    Parameters:
    -----------
    mapdict: a dictionary with keys as the new string values, its values are
        lists which give the scope of which the old string values should
        be changed into new ones.

    Notes:
    ------
    1. The original string that are not included in the mapdict.values()
        will have value of "NOCLASS"

    Example:
    >>> s1 = np.array(','.join(string.lowercase).split(','))
    >>> mapdict = dict(abcdefg=['a', 'b', 'c', 'd',
        'e','f','g'],hijklmn=['h', 'i', 'j', 'k', 'l', 'm',
        'n'],opqrst=['o', 'p', 'q', 'r', 's', 't'],uvwxyz=['u', 'v',
        'w', 'x', 'y', 'z'])
    >>> s2 = mathex.ndarray_string_categorize(s1,mapdict)

    """
    if array.ndim > 1:
        raise ValueError("only accept one dimensional array!")
    else:
        if not isinstance(array[0],str):
            raise TypeError("the first element type is not string!")
        else:
            pass

    def mapfunc(element):
        bool_find = False
        for key,vlist in mapdict.items():
            if element in vlist:
                bool_find = True
                return key
        if not bool_find:
            if element == 'nan':
                return np.nan
            else:
                return 'NOCLASS'

    return map(mapfunc,array)

def list_string_categorize(list_of_string,mapdict):
    """
    Recategorize a list of string by mapdict.

    See also,
    ---------
    mathex.ndarray_string_categorize
    """
    return ndarray_string_categorize(np.array(list_of_string),mapdict)

def dataframe_MultiIndex_to_column(df):
    """
    Build a new dataframe with the columns as the MultiIndex first level
        index.
    """
    if len(df.columns) > 1:
        raise ValueError("input MultiIndex dataframe column length > 1")
    else:
        column_name = list(df.columns)[0]
        first_index_list = list(np.unique(df.index.get_level_values(0)))
        dft = df.ix[first_index_list[0]]
        dft = dft.rename(columns={column_name:first_index_list[0]})
        for other_index_name in first_index_list[1:]:
            dft[other_index_name] = df.ix[other_index_name][column_name]
        return dft

def Series_split_by_boolean(series,bool_arr):
    """
    Split the pandas series by the boolean array bool_arr, only the series values
    corresponding to the True values in bool_arr will be retained.

    Returns:
    --------
    A list of sereis. An empty list is returned if all the values in the input
    boolean array are False.

    Example:
    --------
    In [14]: s = pa.Series(np.array([1,1,0,1,0,0,0,1,1,1]))
    
    In [15]: slist = mathex.Series_split_by_boolean(s,s==0)
    
    In [16]: len(slist)
    Out[16]: 2
    
    In [17]: print slist[0],slist[1]
    2    0
    dtype: int64 4    0
    5    0
    6    0
    dtype: int64
    
    In [18]: slist = mathex.Series_split_by_boolean(s,s>0)
    
    In [19]: len(slist)
    Out[19]: 3
    
    In [20]: print slist[0],slist[1],slist[2]
    0    1
    1    1
    dtype: int64 3    1
    dtype: int64 7    1
    8    1
    9    1
    dtype: int64
    """
    if len(bool_arr) != len(series):
        raise ValueError("Input series and boolean array length no equal")
    else:
        index_remain_value = ndarray_index_continous_True_block(bool_arr)
        series_list = []
        if len(index_remain_value) != 0:
            for start,end in index_remain_value:
                s_temp = pa.Series(series.values[start:end+1],index=series.index.values[start:end+1])
                series_list.append(s_temp)

        return series_list

def interp_level(level,num=2,kind='linear'):
    """
    Use scipy.interpolate.interp1d to interpolate the contourf levels.
    """
    orindex = np.arange(len(level)*num-(num-1))
    f = interpolate.interp1d(orindex[0::num],level,kind=kind)
    return f(orindex)



def Ncdata_dict_to_dataframe_panel(Ncdata_dict,variables,mode='spasum',index=None):
    """
    Add to dataframe or panel data from a dictionary of Ncdata.

    Parameters:
    -----------
    mode: could be spasum or spamean
    variables:
        in case of a single variables, a dataframe will be returned.
        in case of a list of variables, a panel will be returned.
    """

    def retrieve_by_mode(Ncdata_obj,mode):
        if mode in ['spasum','spamean']:
            return Ncdata_obj.__getattribute__(mode)
        else:
            raise KeyError

    if isinstance(variables,str):
        dic= {}
        for key,ncobj in Ncdata_dict.items():
            dic[key] = retrieve_by_mode(ncobj,mode).__dict__[variables]
        return pa.DataFrame(dic,index=index)
    elif isinstance(variables,list):
        dic = {}
        for key,ncobj in Ncdata_dict.items():
            subdic = {}
            for var in variables:
                subdic[var] = retrieve_by_mode(ncobj,mode).__dict__[var]
            dic[key] = pa.DataFrame(subdic,index=index)
        return pa.Panel(dic)

def ndarray_year_to_decade(arr):
    dt = arr/10
    decade_list = [str(s)+'0s' for s in dt]
    return decade_list

def ndarray_month_to_season(arr,func):
    """
    Arange month data into seasons.

    Parameters:
    -----------
    arr: must have 3 dim and the first dim length as an even number of 12.
    func: func used to integrate monthly data into seasons. E.g.,
        lambda x:np.sum(x*area,axis=0), or simply np.sum np.mean etc.
        Note axis is the first dim of generated 3-month data, with 2nd dim
        as lengh of years.

    Returns:
    --------
    ndarray with 4 dimensions, first dim as seasonal dim, second one as year.
    """
    if np.ndim(arr) != 3:
        raise ValueError("Can handle only 3-dim data")
    else:
        if arr.shape[0]%12 != 0:
            raise ValueError("The first dim is not even length of 12")
        else:
            arr_mon = arr.reshape(12,-1,arr.shape[-2],arr.shape[-1],order='F')
            datalist = [arr_mon[0:3],arr_mon[3:6],arr_mon[6:9],arr_mon[9:12]]
            #func = lambda x:np.ma.sum(x*area_km2,axis=0)
            datalist_short = map(func,datalist)
            datalist_short = [dt[np.newaxis,...] for dt in datalist_short]

            if np.ma.isMA(datalist_short[0]):
                confunc = np.ma.concatenate
            else:
                confunc = np.concatenate
            season_data = confunc(datalist_short,axis=0)
            return season_data

def ndarray_month_detrend(tp):
    """
    Use scipy.signal.detrend to remove the linear trend for each month.

    Parameters:
    -----------
    tp: data are of 3-dim and the first dim length should be even number
        of 12, containting sequentially the data for each month.
    """
    if np.ndim(tp) != 3:
        raise ValueError("Can handle only 3-dim data")
    else:
        if tp.shape[0]%12 != 0:
            raise ValueError("The first dim is not even length of 12")
        else:
            tp_month = tp.reshape((12,-1,tp.shape[-2],tp.shape[-1]),order='F')
            from scipy import signal

            monthlist = []

            for i in range(12):
                dt = np.ma.apply_along_axis(signal.detrend,0,tp_month[i])
                monthlist.append(dt)

            newlist = [dt[np.newaxis,...] for dt in monthlist]
            if np.ma.isMA(newlist[0]):
                confunc = np.ma.concatenate
            else:
                confunc = np.concatenate
            month_data = confunc(newlist,axis=0)
            month_datanew = month_data.reshape((month_data.shape[0]*month_data.shape[1],month_data.shape[-2],month_data.shape[-1]),order='F')
            return month_datanew

def ndarray_nonmask_nonnan(arr):
    """
    Take the non-masked or non-nan part of the input array.
    """
    if np.ma.isMA(arr):
        if isinstance(arr.mask,np.bool_):
            pass
        elif isinstance(arr.mask,np.ndarray):
            arr = arr[~arr.mask]
        else:
            raise TypeError("Unknown mask type")


    return arr[~np.isnan(arr)]


def dataframe_year_mon_day_to_DOY(df,year='year',month='month',day='day'):
    doylist = []
    for year,mon,day in zip(df[year].values.astype('int'),df[month].values.astype('int'),df[day].values.astype('int')):
        dt = datetime.date(year,mon,day)
        doylist.append(dt.timetuple().tm_yday)
    return doylist

def ndarray_pearson_correlation(arr1,arr2,checkna=False,nobs_threshold=7):
    """
    Calculate the pearson correlation coefficient between arr1 and arr2. The
        correlation dimension must be the first dimension of arr1,arr2.

    Parameters:
    -----------
    arr1,arr2 currently must be 3-dim arrays, the pearson correlation
        coefficient was calculated between the first dim of arr1 and arr2
        for each pixel (of dim2Xdim3).
    nobs_threshold: The minimum sample size to calculate the correlation.

    Returns:
    --------
    If checkna=True or masked arrays are provided, return 3XmXn array, 
        with the first dim as (coef,pvalue,num);
        otherwise return 2XmXn array, with the first dim as (coef,pvalue). In case
        of one of input arrays are masked arrays, checkna will be set as True.
        The masked arrays will be filled with np.nan values being used.
    """

    if np.ma.isMA(arr1):
        arr1 = arr1.filled(np.nan)
        checkna = True

    if np.ma.isMA(arr2):
        arr2 = arr2.filled(np.nan)
        checkna = True

    if arr1.shape != arr2.shape:
        raise ValueError("The shape of the two input array not equal")
    else:
        if len(arr1.shape) != 3:
            raise ValueError("Currently only accept 3-dim array")
        else:
            first_len = arr1.shape[0]
            arr = np.concatenate([arr1,arr2])
            func = lambda x:_get_pearson(x[:first_len],x[first_len:],
                                         checkna=checkna,
                                         nobs_threshold=nobs_threshold)
            outarr = np.apply_along_axis(func,0,arr)

            if checkna:
                return np.ma.masked_invalid(outarr)
            else:
                return outarr

def _grid_linreg_checkna(y,x=None,nobs_threshold=7):
    """
    Note this is only for internal use in the griddata based regression.
    """
    if x is None:
        x = np.arange(len(y))
    x,y = ndarray_nonnan_common(x,y)
    num = len(x)
    if num < nobs_threshold:
        slope = np.nan
        R2 = np.nan
        p_value = np.nan
        validnum = np.nan
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        R2 = r_value**2
        validnum = num
    return (slope,R2,p_value,validnum)


def _get_pearson(x,y,checkna=False,nobs_threshold=7):
    """
    Use sp.stats.stats.pearsonr to get the perason correlation between x and y.

    Returns:
    --------
    If checkna=True, return coef,pvalue; otherwise return (coef,pvalue,num)
    """
    if checkna == False:
        coef,pvalue = sp.stats.stats.pearsonr(x,y)
        return coef,pvalue
    else:
        x,y = ndarray_nonnan_common(x,y)
        num = len(x)
        if num < nobs_threshold:
            coef = np.nan
            pvalue = np.nan
            validnum = np.nan
        else:
            coef,pvalue = sp.stats.stats.pearsonr(x,y)
            validnum = num
        return (coef,pvalue,validnum)

def ndarray_partial_corr(yarr,xarr,nobs_threshold=7):
    """
    Use dataframe_partial_corr_sta to calculate the partial corr ceof
        between y~x for each pixel.

    Return:
    -------
    An ndarray with dimension of (:,4,:,:), with the rest dimension length
        the same as xarr, pixels mased in orginal data or with valid
        number of obs lower than nobs_threshold are filled with np.nan
        The second dimension contains:
        u'r_partial',u'pvalues', u'nobs', u'params'
    """
    xcols = ['x{0}'.format(num) for num in range(len(xarr))]
    if np.ma.isMA(yarr):
        yarr = yarr.filled(np.nan)

    if np.ma.isMA(xarr):
        xarr = xarr.filled(np.nan)

    if np.shape(yarr) != np.shape(xarr)[1:]:
        raise ValueError("dimensions of y and x arrays are not the same!")
    else:
        if np.ndim(yarr) != 3:
            raise ValueError("ndim of yarr not equal 3")
        else:
            xshape = xarr.shape
            resarr = np.ones((xshape[0],4,xshape[2],xshape[3]))
            for i in range(xarr.shape[2]):
                for j in range(xarr.shape[3]):
                    dft = pa.DataFrame(xarr[:,:,i,j].T,columns=xcols)
                    dft['y'] = yarr[:,i,j]

                    dft = dft.dropna()
                    if len(dft) < nobs_threshold:
                        resarr[:,:,i,j] = np.nan
                    else:
                        try:
                            dft_res = dataframe_partial_corr_sta(dft,'y',xcols)
                            resarr[:,:,i,j] = dft_res[[u'r_partial',u'pvalues', u'nobs', u'params']].values
                        # this ValueError capatures the corner cases
                        # that solutions could not be find, e.g., all y
                        # values are zero while xs are not because of the inconsistency
                        # among data sets.
                        except ValueError:
                            resarr[:,:,i,j] = np.nan
            return resarr


def ndarray_nonnan_common(x,y):
    if np.ma.isMA(x):
        x = x.filled(np.nan)
    if np.ma.isMA(y):
        y = y.filled(np.nan)
    validcom = np.logical_and(~np.isnan(x),~np.isnan(y))
    return (x[validcom],y[validcom])

def ndarray_mask_merge(x,y):
    """
    Merge masks from two arrays, if neither of the input array is masked,
        return None; otherwise return a logical type array.
    """
    if x.shape != y.shape:
        raise ValueError("Two input arrays do not have the same shape")
    else:
        if np.ma.isMA(x):
            xmask = x.mask
        else:
            xmask = None

        if np.ma.isMA(y):
            ymask = y.mask
        else:
            ymask = None

        if xmask is None and ymask is None:
            return None
        else:
            if xmask is None:
                return ymask
            else:
                if ymask is None:
                    return xmask
                else:
                    return np.logical_or(xmask,ymask)


def ndarray_midpoint_to_length(soilmid):
    """
    Convert the mid-point positions of soilmid to the length of each interval.
    """
    soillen = np.zeros(len(soilmid))
    soillen[0] = soilmid[0] * 2
    for i in range(1,len(soillen)):
        soillen[i] = (soilmid[i] - soillen.cumsum()[i-1]) * 2
    return soillen

def apply_func(glob_data,pyfunc=None):
    if pyfunc is not None:
        if callable(pyfunc):
            glob_data = pyfunc(glob_data)
        else:
            raise TypeError("pyfunc not callable")
    return glob_data


def geo_area_latlon(numlat,numlon):
    """
    Return a grid of land area based on the number of grids in terms of
    latitude and longitude.

    The equations is from here:
    http://mathforum.org/library/drmath/view/63767.html

    Return
    ------
    a numpy array with the shape of (numlat,numlon) indicating the area
    in unit of km2.

    Notes:
    ------
    In case numlat is not an even number, the area of (numlat*2,numlon)
    is first calculated and then regrided to (numlat,numlon)
    """
    def get_area_grid(lat1,lat2,numlon):
        """
        Calculate the area of one grid cell.
        """
        R=6370.
        lat2 = lat2*np.pi/180
        lat1 = lat1*np.pi/180
        area = 2*np.pi*np.power(R,2)*np.abs(np.sin(lat2)-np.sin(lat1))/numlon
        return area

    if numlat%2 != 0:
        numlat = numlat*2
        regrid = True
    else:
        regrid = False

    lat_north = np.linspace(0,90,numlat/2+1)
    area_north = []
    for i in range(len(lat_north)-1):
        lat1 = lat_north[i]
        lat2 = lat_north[i+1]
        area = get_area_grid(lat1,lat2,numlon)
        area_north.append(area)
    area_north = np.array(area_north)
    area_globe = np.tile(np.concatenate([area_north[::-1],area_north]),(numlon,1)).transpose()
    if regrid:
        area_globe = area_globe.reshape(2,-1,numlon).sum(axis=0)

    return area_globe

def dataframe_partial_corr(df):
    """
    Calcualte partial correlation coefficient. The function is taken from
    https://gist.github.com/fabianp/9396204419c7b638d38f

    The data are first standardized before feeding into the function.

    Notes:
    ------
    1. rows containing any Nan will be dropped.
    """
    df = df.dropna()
    data = df.values
    data_stdized = (data - data.mean(axis=0))/np.std(data,axis=0)
    mpar = partial_corr(data_stdized)
    dft = pa.DataFrame(mpar,columns=df.columns,index=df.columns)
    return dft

def dataframe_partial_corr_sta(dft,ycol,xcols):
    """
    Get partial correlation coefs of ycol ~ xcols but with statistical
        information being returned as well.

    Parameters:
    -----------
    ycol: string
    xcols: list.

    Notes:
    ------
    No nan check will be done.
    """
    cols = [ycol]+xcols
    dft = dft[cols]
    dft_res = dataframe_reg_OLS_multivar(dft,ycol=ycol,xcol=xcols).ix[xcols]
    dft_res['r_partial'] = dataframe_partial_corr(dft)[ycol].ix[xcols]
    return dft_res


def partial_corr(C):
    """
    Partial Correlation in Python (clone of Matlab's partialcorr)
    This uses the linear regression approach to compute the partial 
    correlation (might be slow for a huge number of variables). The 
    algorithm is detailed here:
        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
    the algorithm can be summarized as
        1) perform a normal linear least-squares regression with X as the target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 
        The result is the partial correlation between X and Y while controlling for the effect of Z
    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com

    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

def dataframe_reg_OLS_multivar(df,ycol=None,xcol=None,attrs=None):
    """
    Use import statsmodels.api as sm to conduct multivariate linear
    regressions.

    Parameters:
    -----------
    attrs: attributes to extract from the results of
        statsmodels.regression.linear_model.RegressionResultsWrapper.
        Default is ['params','tvalues','pvalues']

    Notes:
    ------
    Nan values are taken with special care.
    """
    if ycol is None:
        ycol = df.columns[0]

    if xcol is None:
        xcol = pb.StringListAnotB(df.columns,[ycol])

    X = df[xcol].values
    X = sm.add_constant(X)
    model = sm.OLS(df[ycol].values, X)
    results = model.fit()

    if attrs is None:
        attrs = ['params','tvalues','pvalues','nobs']

    dic = OrderedDict()
    for name in attrs:
        dic[name] = results.__getattribute__(name)
    dft = pa.DataFrame(dic,index=['Intercept']+xcol)
    return dft


def Panel_partial_corr(panel,ycol,xcols):
    """
    Refer to the docstrings of Panel4D_partial_corr
    """
    dic1 = OrderedDict()
    for item in p4d.items:
        dft = p4d[lab,item,:,:].dropna()
        dft_res = mathex.dataframe_reg_OLS_multivar(dft,ycol=ycol,xcol=xcols).ix[xcols]
        dft_res['r_partial'] = mathex.dataframe_partial_corr(dft)[ycol].ix[xcols]
        dic1[item] = dft_res

def Panel4D_partial_corr(p4d,ycol,xcols):
    """
    Calculate partial correlation coef and p-values using
        mathex.dataframe_reg_OLS_multivar and mathex.dataframe_partial_corr.
        Returns a Panel4D, with labels and items unchanged; major_axis as
        regression variables, minor_axis as statistical variables (r_partial,
        p-value etc.)

    Parameters:
    -----------
    p4d: Panel4D object, with major_axis as observations, minor_axis as variables.
    xcols: columns (variables) serving as independent variables; could be a
        subset list p4d.minor_axis
    ycol: string. Dependent variable.
    """
    cols = xcols+[ycol]
    dic2 = OrderedDict()
    for lab in p4d.labels:
        dic1 = OrderedDict()
        for item in p4d.items:
            dft = p4d[lab,item,:,cols].dropna()
            if len(dft) == 0:
                print "lab,item = ",(lab,item)
                raise ValueError("dataframe length is zero!")

            dft_res = dataframe_reg_OLS_multivar(dft,ycol=ycol,xcol=xcols).ix[xcols]
            dft_res['r_partial'] = dataframe_partial_corr(dft)[ycol].ix[xcols]
            dic1[item] = dft_res
        dic2[lab] = dic1
    p4d_reg = pa.Panel4D(dic2)

    return p4d_reg



def cart2pol(x, y):
    """
    Change from x,y to polar cords.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
