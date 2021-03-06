#!/usr/bin/env python


import matplotlib
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pa
import pickle as pk
import os as os
import re as re
import scipy as sp
import mpl_toolkits.basemap as bmp
from mpl_toolkits.basemap import cm
import pdb
import netCDF4 as nc
from matplotlib.backends.backend_pdf import PdfPages
import pb
import numpy as np
import weakref
from collections import Iterable
from collections import OrderedDict
import g
import copy as copy
import bmap
import Pdata_test as Ptest
import gnc
import LabelAxes
import tools

def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f
    return decorator

def gsetp(*artist,**kwargs):
    """
    Purpose: set artist properties by kwagrs pairs in an easy and flexible way.
    Note:
        1.  artist will be flat using pb.iteflat so you can use mixed types of matplotlib artists as long as they have the same keyword properties.
        2.  when artist is a tuple or list,kwargs[key] can also be set as tuple or list, but when kwargs[key] is only one value, it will be broadcast 
            to the same length with artist automatically.
    """
    print "Deprecating Warning!"
    if len(artist)==1 and isinstance(artist,(tuple,list)):
        artist_list=pb.iteflat(artist[0])
    else:
        artist_list=pb.iteflat(artist)

    for key in kwargs:
        value=kwargs[key]
        if not isinstance(value,Iterable) or isinstance(value,str):
            value_list=[value]*len(artist_list)
        else:
            if len(value)==1:
                value_list=value*len(artist_list)
            else:
                value_list=pb.iteflat(value)
        if len(artist_list)!=len(value_list):
            raise ValueError('artist list lenght {0} is not equal to value list length {1}'.format(len(artist_list),len(value_list)))
        else:
            for art,val in zip(artist_list,value_list):
                plt.setp(art,key,val)
        print key,value_list,'has been set'
    return artist_list,[key]*len(artist_list),value_list

#choose default colorlist
def _replace_none_colorlist(colors=None,num=None):
    if colors is None:
        if num <= len(g.pcolor):
            return g.pcolor[0:num]
        else:
            return g.pcolor*(num/len(g.pcolor)+1)
            #raise ValueError("g.pcolor is not long enough when using default colorlist")
    else:
        return colors

def _replace_none_axes(ax):
    if ax is None:
        fig, axnew = g.Create_1Axes()
        return axnew
    else:
        return ax

def _replace_none_by_given(orinput,default):
    if orinput is None:
        return default
    else:
        return orinput

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

def Dic_Extract_By_Subkeylist(indic,keylist):
    """
    Return a new dic by extracting the key/value paris present in keylist
    """
    outdic={}
    for key in keylist:
        try:
            outdic[key]=indic[key]
        except KeyError:
            pass
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

def StringListAnotB(listA,listB):
    return [i for i in listA if i not in listB]

def FilterStringList(keyword,input_list):
    return [x for x in input_list if re.search(keyword,x)]

def Is_Nested_Dic(indic):
    for value in indic.values():
        if isinstance(value,dict):
            return True
        else:
            pass
    return False

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

def _get_attr_value_from_objins_dic(objins_dic,*attr_list):
    """
    get attribute value list from a group of object instances by specifying attribute names.
    Return a dict with attr name as key and attr_value list as as key values.
    objins: object instance
    """
    attr_dic={}
    for attr_name in attr_list:
        attr_dic[attr_name]=dict()
        for tag,objins in objins_dic.items():
            attr_dic[attr_name][tag]=objins.__getattribute__(attr_name)
    return attr_dic


def _build_list_of_axes_by_num(num,force_axs=None,ncols=None,
                               sharex=True, sharey=False,
                               column_major=False, **kwargs):
    if force_axs is None:
        if ncols == 1:
            nrows=num
        else:
            if num%ncols == 0:
                nrows=num/ncols
            else:
                nrows=num/ncols+1
        fig,axt=plt.subplots(nrows=nrows, ncols=ncols,
                                 sharex=sharex, sharey=sharey,
                                 **kwargs)
        if isinstance(axt,mat.axes.Axes):
            axs=[axt]
        else:
            if column_major == False:
                axs=axt.flatten()[0:num]
            else:
                axs = axt.flatten(order='F')[0:num]
    else:
        if isinstance(force_axs,mat.axes.Axes):
            axs = [force_axs]
        elif isinstance(force_axs,(list,tuple,np.ndarray)):
            if num<=len(force_axs):
                axs=force_axs[0:num]
            else:
                raise ValueError("given force_axs length is smaller than required.")
        else:
            raise TypeError("force_axs TypeError")
    return axs

def _creat_dict_of_tagaxes_by_tagseq(force_axs=None,tagseq=None,
                                 default_tagseq=None,
                                 ncols=1, column_major=False,
                                 sharex=True,sharey=False,
                                 force_axdic=None,
                                 **subplot_kwargs):
    """
    Return a dictionary of tag/axes.

    Parameters:
    -----------
    force_axs: force axes, the length could be bigger than tagseq
    default_tagseq: tagseq that's used as default.

    """

    if force_axdic is not None:
        return force_axdic
    else:
        tag_list=_replace_none_by_given(tagseq, default_tagseq)
        num=len(tag_list)
        axs = _build_list_of_axes_by_num(num,force_axs=force_axs,ncols=ncols,
                                         sharex=sharex, sharey=sharey,
                                         column_major=column_major,
                                         **subplot_kwargs)
        return dict(zip(tag_list,axs))

def _treat_axes_dict(axdic,tagpos='ul',unit=None,xlim=None):
    """
    Label tag and unit for a dictionary of axes with tag as keys.
    """
    print "Deprecating _treat_axes_dict"
    for tag,axt in axdic.items():
        g.Set_AxText(axt,tag,tagpos)
        if unit is not None:
            if isinstance(unit,str):
                #print "forced unit is used"
                axt.set_ylabel(unit)
            else:
                raise ValueError("Strange unit")
        else:
            pass

        if xlim is not None:
            axt.set_xlim(xlim)

def _treat_taglist_by_tagpresurfix(taglist,tagprefix='',tagbracket='normal'):
    """
    Treat taglist by prefix or surfix to allow easy plotting.

    Parameters:
    ----------
    tagprefix:
        1. default value is empty string.
        2. in case of 'alpha', alphabetic values will be appended.
        2. in case of 'numeric', numbers will be appended.

    tagbracket:
        1. default value is 'normal', i.e., "()" will be used.
    """
    import string
    if tagprefix == '':
        tagprefix_list = ['']*len(taglist)
    elif tagprefix == 'alpha':
        tagprefix_list = string.ascii_letters[0:len(taglist)]
    elif tagprefix == 'numeric':
        tagprefix_list = map(str,range(1,len(taglist)+1))
    else:
        raise ValueError("wrong tagprefix value!")

    def decorate_tagbracket(tagprefix_list,tagbracket):
        if tagprefix_list[0] == '':
            return tagprefix_list
        else:
            if tagbracket == 'normal':
                return ['('+s+') ' for s in tagprefix_list]
            else:
                raise ValueError("wrong tagbracket value")

    complete_tagprefix_list = decorate_tagbracket(tagprefix_list,tagbracket)
    taglist_final = [s1+s2 for (s1,s2) in zip(complete_tagprefix_list,taglist)]
    return taglist_final

def _creat_dict_of_tagaxes_by_tagseq_g(**kwargs):
    """
    This is to try supersede the functions of
    _creat_dict_of_tagaxes_by_tagseq and _treat_axes_dict
    Return a dictionary of tag/axes.

    Parameters:
    -----------
    kwargs:
        force_axs: force the axes.
        tagseq: the sequence for parent tags.
        force_axdic: force a dictionary of parent_tag/axes pairs.
        ncols: num of columns when force_axs is None
        sharex,sharey: the same as plt.subplots
        tagpos: the position of parent_tag
        column_major: True if parent tags are deployed in column-wise.
        unit: used as ylabel for each subplot
        xlim: xlim
        subkw: kwarg in plt.subplots function
        tagprefix:
            1. default value is empty string.
            2. in case of 'alpha', alphabetic values will be appended.
            2. in case of 'numeric', numbers will be appended.
        tagbracket:
            1. default value is 'normal', i.e., "()" will be used.
        tagcolor: the tag color
    """
    paradict = dict(force_axs=None,tagseq=None,
                         default_tagseq=None,
                         ncols=1, column_major=False,
                         sharex=False, sharey=False,
                         force_axdic=None,
                         tagpos=None, default_tagpos='ul',
                         unit=None, xlim=None,
                         subkw={},tagcolor='b',
                         tagprefix='',tagbracket='normal')

    extra_keylist = pb.StringListAnotB(kwargs.keys(),paradict.keys())
    if len(extra_keylist) != 0:
        raise ValueError('''keyword '{0}' not in paradict'''
                            .format(extra_keylist[0]))
    else:
        paradict.update(kwargs)

    #retrieve the values
    force_axs = paradict['force_axs']
    tagseq = paradict['tagseq']
    default_tagseq = paradict['default_tagseq']
    ncols = paradict['ncols']
    column_major = paradict['column_major']
    sharex = paradict['sharex']
    sharey = paradict['sharey']
    force_axdic = paradict['force_axdic']
    default_tagpos = paradict['default_tagpos']
    tagpos = paradict['tagpos']
    unit = paradict['unit']
    xlim = paradict['xlim']
    subkw = paradict['subkw']
    tagcolor = paradict['tagcolor']
    tagprefix = paradict['tagprefix']
    tagbracket = paradict['tagbracket']

    if force_axdic is not None:
        axdic = force_axdic
    else:
        tag_list=_replace_none_by_given(tagseq, default_tagseq)
        num=len(tag_list)
        axs = _build_list_of_axes_by_num(num,force_axs=force_axs,ncols=ncols,
                                         sharex=sharex, sharey=sharey,
                                         column_major=column_major,
                                         **subkw)
        axdic = OrderedDict(zip(tag_list,axs))

    #treat the tagprefix
    tags_final = _treat_taglist_by_tagpresurfix(axdic.keys(),
                    tagprefix=tagprefix, tagbracket=tagbracket)
    tags_final_dic = dict(zip(axdic.keys(),tags_final))

    #print 'default_tagpos before',default_tagpos
    for tag,axt in axdic.items():
        default_tagpos = _replace_none_by_given(default_tagpos, 'ul')
        tagpos = _replace_none_by_given(tagpos, default_tagpos)
        #print 'default_tagpos',default_tagpos
        #print 'tagpos',tagpos
        g.Set_AxText(axt,tags_final_dic[tag],tagpos,color=tagcolor)
        if unit is not None:
            if isinstance(unit,str):
                #print "forced unit is used"
                axt.set_ylabel(unit)
            else:
                raise ValueError("Strange unit")
        else:
            pass

        if xlim is not None:
            axt.set_xlim(xlim)

    return axdic


def pleg_merge(*pdlist):
    """
    Return a merged ProxyLegend for the Pdata objects.
    """
    pleglist = [pd.get_proleg() for pd in pdlist]
    return g.ProxyLegend.merge_pleg(*pleglist)

class Pdata(object):
    """
    There are two ways to set customized keyword, it can be done using add_attr_by_tag function or passed when calling ploting function. 
    Yet the seting by add_attr_by_tag has high priority.
    Pdata is an object constructed for easy ploting in matplotlib.
    """
    #declare base keylist
    _data_base_keylist=['x','y','xerrl','xerrh','yerrl','yerrh']
    _extra_base_keylist=['label','bwidth','bbottom','bleftshift']
    _extra_nonplot_keylist=['unit','ylab']
    _all_nonplot_keylist = _extra_nonplot_keylist + _data_base_keylist +\
                           _extra_base_keylist
    _extra_base_keylist_default=dict(zip(_extra_base_keylist,[None,0.5,0,0]))
    _new_entry=dict.fromkeys(_data_base_keylist,None)
    _new_entry.update(_extra_base_keylist_default)

    _scatter_attr_base_keylist=['ssize','scolor','smarker','scmap',
                                'snorm','svmin','svmax',]
    _error_attr_base_keylist=['efmt','ecolor','elinewidth','capsize',
                              'barsabove','lolims','uplims','xlolims',
                              'xuplims']
    _bar_attr_base_keylist=[]
    _plot_attr_dic=dict(scatter=_scatter_attr_base_keylist,
                        errorbar=_error_attr_base_keylist,
                        bar=_bar_attr_base_keylist)

    _plot_attr_keylist_all = _scatter_attr_base_keylist + \
                             _error_attr_base_keylist + \
                             _bar_attr_base_keylist

    #initiate default plot attritubtes
    _error_attr_default={             \
                 'efmt':None,\
                 'ecolor':'r',   \
                 'elinewidth':None, \
                 'capsize':3, \
                 'barsabove':False, \
                 'lolims':False,\
                 'uplims':False,\
                 'xlolims':False,\
                 'xuplims':False \
                 }
    _scatter_attr_default={             \
                 'ssize':20,   \
                 'scolor':'b', \
                 'smarker':'o',\
                 'scmap':None, \
                 'snorm':None, \
                 'svmin':None, \
                 'svmax':None, \
                 }
    _bar_attr_default={}

    #set default extra plot attribute keyword argument values
    _plot_attr_default={}
    _plot_attr_default.update(_error_attr_default)
    _plot_attr_default.update(_scatter_attr_default)
    _plot_attr_default.update(_bar_attr_default)

    _default_plottype_list = ['Line2D','Scatter_PathC','Bar_Container']

    #TESTED
    def _plot_attr_keylist_check(self):
        inlist = Pdata._plot_attr_keylist_all
        if any(inlist.count(x) > 1 for x in inlist):
            raise ValueError('plot base attribute lists have duplicates!')



    def __init__(self,data=None):
        self._plot_attr_keylist_check()

        if data is None:
            self.data = {}
        else:
            self.data = copy.deepcopy(data)

        if self.data == {}:
            self._taglist = []
        else:
            self._taglist = self.data.keys()
            self._data_complete_check_all()

    def add_tag(self,tag=None):
        """
        add_tag will retain the sequence of entering tags.
        """
        self.data[tag]=Pdata._new_entry.copy()
        self._taglist.append(tag)

    def _add_indata_by_tag_and_column(self,indata,tag,column):
        if isinstance(indata,list):
            indata = np.array(indata)
        elif isinstance(indata,np.ndarray):
            pass
        else:
            raise TypeError("Wrong data type {0}".format(type(indata)))
        self.data[tag][column]=indata
    def addx(self,indata,tag):
        self._add_indata_by_tag_and_column(indata,tag,'x')
    def addy(self,indata,tag):
        self._add_indata_by_tag_and_column(indata,tag,'y')
    def _broadcast_scalar_byx(self,tag,scalar):
        return np.repeat(scalar,len(self.data[tag]['x']))
    def addxerrl(self,indata,tag):
        if not isinstance(indata,Iterable):
            indata=self._broadcast_scalar_byx(tag,indata)
        self._add_indata_by_tag_and_column(indata,tag,'xerrl')
    def addxerrh(self,indata,tag):
        if not isinstance(indata,Iterable):
            indata=self._broadcast_scalar_byx(tag,indata)
        self._add_indata_by_tag_and_column(indata,tag,'xerrh')
    def addyerrl(self,indata,tag):
        if not isinstance(indata,Iterable):
            indata=self._broadcast_scalar_byx(tag,indata)
        self._add_indata_by_tag_and_column(indata,tag,'yerrl')
    def addyerrh(self,indata,tag):
        if not isinstance(indata,Iterable):
            indata=self._broadcast_scalar_byx(tag,indata)
        self._add_indata_by_tag_and_column(indata,tag,'yerrh')

    #TESTED
    def add_entry_by_dic(self,**kwargs):
        """
        Add a complete entry by dictionary. note that kwargs[key] must
            have keys()=['x','y',...]
        Parameters:
        ----------
        kwargs: kwargs are tag/tag_value pairs, tag_value is again a dict
            with 'x'/'y'/'yerrh' ... etc as keys.

        """
        for tag,tag_value in kwargs.items():
            self.add_tag(tag)
            for attr_name,attr_value in tag_value.items():
                self.data[tag][attr_name] = attr_value

    #TESTED
    def add_entry_noerror(self,x=None,y=None,tag=None):
        '''
        Add an entry with no x/y error

        Notes:
        ------
        1. if x is None, default index is used.
        2. as add_tag, add_entry_noerror will retain the tag sequence.
        '''
        self.add_tag(tag)
        if x is None:
            self.addx(np.arange(len(y))+1,tag)
        else:
            if len(y)!=len(x):
                raise ValueError('''lenght of ydata for 'tag' {0} is {1},
                    not equal to length of xdata with length of {2}'''
                    .format(tag,len(y),len(x)))
            else:
                self.addx(x,tag)
        self.addy(y,tag)

    @classmethod
    def from_ndarray(cls,arr,tagaxis=0,taglist=None,x=None):
        """
        Arguments:
        ----------
        x: default is the 1-based sequential array.
        """
        if np.ndim(arr) != 2:
            raise ValueError('''array ndim is {0}, only 2 is valid'''.
                                format(arr.ndim))
        if tagaxis == 0:
            datalist = [arr[i] for i in range(arr.shape[0])]
        elif tagaxis == 1:
            datalist = [arr[:,i] for i in range(arr.shape[1])]
        else:
            raise ValueError("unknown tagaxis!")
        taglist = _replace_none_by_given(taglist,['tag'+str(s) for s in np.arange(1,len(datalist)+1)])
        pd = Pdata()
        pd.add_entry_sharex_noerror_by_dic(dict(zip(taglist,datalist)),x=x)
        pd.set_tag_order(taglist)
        return pd

    @classmethod
    def from_dataframe_groupby(cls,dfgroup,mapdict):
        """
        Create Pdata object from dataframe.groupby object.

        Parameters:
        -----------
        mapdict: a dictionary specifying the mapping between
            dataframe columns and the keynames in Pdata, i.e.,
            ['x'/'y'/'yerrh'...]. Note the column names must
            be the keys of mapdict.
        """
        def replace_dict_by_mapping(dfdict,mapdict):
            """
            Change the column name of dfdict to the names needed by
                Pdata as in Pdata._data_base_keylist

            Parameters:
            -----------
            mapdict: a dictionary specifying the mapping between
                dataframe columns and the keynames in Pdata, i.e.,
                ['x'/'y'/'yerrh'...]
            """
            datadict = Pdata._new_entry.copy()
            for col in dfdict.keys():
                if col in mapdict:
                    datadict[mapdict[col]] = dfdict[col]
            return datadict

        pd = Pdata()
        for tag,df in dfgroup:
            dfdict = df.to_dict(outtype='list')
            datadict = replace_dict_by_mapping(dfdict,mapdict)
            pd.add_entry_by_dic(**{tag:datadict})
        return pd

    @classmethod
    def from_dict(cls,dic,x=None):
        pd = Pdata()
        pd.add_entry_sharex_noerror_by_dic(dic,x=x)
        return pd

    @classmethod
    def from_dataframe(cls,df,df_func=None,index_func=None,
                       force_sharex=None):
        """
        Create a sharex Pdata.Pdata object from pandas DataFrame

        Parameters:
        -----------
        df_func: function that applies on DataFrame before feeding data
            into Pdata.
        index_func: index function that will be applied before using the
            DataFrame index as shared xaxis of the Pdata object, this is
            useful as sometimes DataFrame index could be a bit strange
            and not readily compatible with matplotlib functions.
        force_sharex: In case index_func could not achieve the object to
            transform the index to desired sharex xaxis, force_sharex
            is used to force write the Pdata shared xaxis.

        Notes:
        ------
        1. the column sequence of pa.DataFrame will be retained as taglist.
        """
        pd = Pdata()
        if df_func is not None:
            df = df_func(df)
        if force_sharex is None:
            if index_func is None:
                pd.add_entry_sharex_noerror_by_dic(df,x=df.index.values)
            else:
                pd.add_entry_sharex_noerror_by_dic(df,x=index_func(df.index))
        else:
            pd.add_entry_sharex_noerror_by_dic(df,x=force_sharex)
        pd.set_tag_order(list(df.columns))
        return pd

    @classmethod
    def from_dataframe_error(cls,df,df_func=None,index_func=None,
                       force_sharex=None):
        """
        Automatically put the error value to corresponding tags.
        The error value should be given as:
        ['_xerrl','_xerrh','_yerrl','_yerrh']

        """

        if force_sharex is None:
            if index_func is None:
                xvar = df.index.values
            else:
                xvar = index_func(df.index)
        else:
            xvar = force_sharex

        error_list = ['_xerrl','_xerrh','_yerrl','_yerrh']
        r = re.compile(r'(_xerrl$|_xerrh$|_yerrl$|_yerrh$)')
        taglist = [tag for tag in df.columns if not r.search(tag)]
        subtags_with_error = [tag[:-6] for tag in df.columns if r.search(tag)]

        def treat_tag(tag):
            subdic = {}
            subdic['x'] = xvar
            subdic['y'] = df[tag].values
            if tag not in subtags_with_error:
                return subdic
            else:
                for errname in error_list:
                    if tag+errname in df.columns:
                        subdic[errname[1:]] = df[tag+errname].values
                return subdic



        pd = Pdata()
        for tag in taglist:
            pd.add_entry_by_dic(**{tag:treat_tag(tag)})

        return pd

    @classmethod
    def from_panel(cls,panel,xname=None,yname=None):
        """
        Use the panel.items as taglist to construct a Pdata.Pdata object.

        Parameters:
        -----------
        1.xname/yname: the column names that are used as the x/y values
            in the Pdata.
        """
        taglist = panel.items.tolist()
        pd = Pdata()
        for tag in taglist:
            pd.add_tag(tag=tag)
            pd.addx(panel[tag][xname].values,tag)
            pd.addy(panel[tag][yname].values,tag)

        pd.set_tag_order(taglist)
        return pd

    def add_entry_singleYerror(self,x,y,yerr,tag):
        self.add_tag(tag)
        self.addx(x,tag)
        self.addy(y,tag)
        self.addyerrl(yerr,tag)

    def add_entry_singleYerror3(self,data_array,tag):
        """
        add an entry by giving one single 3Xn np.ndarray, with 1st row as X,
            2nd row as Y, 3rd row as Yerr.
        """
        x=data_array[0];y=data_array[1];yerr=data_array[2]
        self.add_tag(tag)
        self.addx(x,tag)
        self.addy(y,tag)
        self.addyerrl(yerr,tag)

    def add_entry_doubleYerror(self,x,y,yerrl,yerrh,tag):
        self.add_tag(tag)
        self.addx(x,tag)
        self.addy(y,tag)
        self.addyerrl(yerrl,tag)
        self.addyerrh(yerrh,tag)

    def add_entry_sharex_noerror_by_dic(self,ydic,x=None):
        """
        Add several tags which share the same x data; ydic will be a dictionary
            (or pandas DataFrame) with tag:ydata pairs. the length of x must
            be equal to all that of ydic[tag]

        Notes:
        ------
        1. the sequence of pa.DataFrame columns will be retained in the taglist
        2. the ydic keys squence may change as no OrderedDict is availabel with
            python 2.6
        """
        #convert pandas dataframe to dict
        if isinstance(ydic,pa.DataFrame):
            newydic=OrderedDict()
            for key in ydic.columns:
                newydic[key]=np.array(ydic[key])
            taglist = list(ydic.columns)
            ydic=newydic
        elif isinstance(ydic,dict):
            taglist = ydic.keys()
        else:
            raise TypeError("input can only ba pa.DataFrame or dict")

        for tag in taglist:
            ydata=ydic[tag]
            self.add_entry_noerror(x=x,y=ydata,tag=tag)

    def add_entry_noerror_by_dic_default_xindex(self,ydic):
        print "DeprecatingWarning! add_entry_noerror_by_dic_default_xindex"
        for tag,ydata in ydic.items():
            x=np.arange(len(ydata))+1
            self.add_entry_noerror(x=x,y=ydata,tag=tag)

    #TODO: no distinction has been made for keys 'x' and 'scolor'
    def subset_end(self,end_num):
        """
        Subset the pdata by retaining only the last 'end_num' number of
            last elements.
        """
        pdata=self.copy()
        dic = {}
        for tag in pdata.data.keys():
            dic[tag]=Dic_Subset_End(pdata.data[tag],end_num)
        pdatanew = Pdata(dic)
        pdatanew.set_tag_order(self.taglist)
        return pdatanew

    #TODO: same problem as subset_begin
    def subset_begin(self,end_num):
        """
        Subset the pdata by retaining only the beginning 'end_num' number of
            last elements.
        """
        pdata=self.copy()
        dic = {}
        for tag in pdata.data.keys():
            dic[tag]=Dic_Subset_Begin(pdata.data[tag],end_num)
        pdatanew = Pdata(dic)
        pdatanew.set_tag_order(self.taglist)
        return pdatanew

    def copy(self):
        data=copy.deepcopy(self.data)
        pdata=Pdata(data)
        pdata.set_tag_order(self._taglist)
        return pdata

    #NON_TESTED
    def add_entry_df_groupby_column(self,indf,tag_col=None,**kwargs):
        """
        Purpose: Add tag:tag_dic pairs by grouping a pandas DataFrame by a
            column. The unique values in the column which is used as
            groupby will serve as tags. x,y,xerrl... etc. are specified
            by using kwargs. supported keywords: ['x','y','xerrl','xerrh',
            'yerrl','yerrh']
        Example:
            >>> comgpp_fluxdailyobe=pa.read_csv('/home/chaoyue/python/
                testdata/comgpp_fluxdailyobe_data.csv')
            >>> pdata=Pdata.Pdata()
            >>> pdata.add_entry_df_groupby_column(comgpp_fluxdailyobe,
                tag_col='site',x='mod',y='GEP_Amiro')
            >>> pdata.add_attr_by_tag(scolor=['g','b','r'])
            >>> fig,ax=g.Create_1Axes()
            >>> pdata.scatter(ax)
            >>> pdata.set_legend_scatter(ax,taglab=True)
        """
        indf_groupby=indf.groupby(tag_col)
        for tag,tag_df in indf_groupby:
            entry_dic={}
            for basic_key,df_colname in kwargs.items():
                entry_dic[basic_key]=np.array(tag_df[df_colname])
            self.add_entry_by_dic(**{tag:entry_dic})


    def to_csv(self,tag,fname):
        """
        use pandas dataframe object to output the tag into csv file.
        """
        df = pa.DataFrame({'x':self.data[tag]['x'],
                           'y':self.data[tag]['y']})
        df.to_csv(fname,index=False)

    def list_tags(self,tagkw=None):
        """
        Method to formally retrieve Pdata._taglist
        """
        if tagkw is None:
            return self._taglist
        else:
            return FilterStringList(tagkw, self._taglist)

    @property
    def taglist(self):
        return self._taglist

    def list_keys_for_tag(self,tag):
        """
        list the keys for the specified tag.
        """
        return self.data[tag].keys()


    #TESTED
    def set_tag_order(self,tagseq=None):
        """
        Set tag order and this order will be kept throughout all the class
        method when default taglist is used.
        """
        if isinstance(tagseq,np.ndarray):
            tagseq = list(tagseq)
        if sorted(self._taglist) == sorted(tagseq):
            self._taglist = tagseq[:]
        else:
            raise ValueError('ordered tag list not equal to present taglist')

    def _set_default_tag(self,taglist='all'):
        if taglist=='all':
            return self._taglist
        else:
            return taglist

    def list_attr(self,*attr_name):
        outdic={}
        for tag,tag_data in self.data.items():
            outdic[tag]=Dic_Extract_By_Subkeylist(tag_data,iteflat([attr_name]))
        return outdic

    def list_attr_extra_base(self):
        attr_extra_dic=self.list_attr(*self._extra_base_keylist)
        if Dic_Test_Empty_Dic(attr_extra_dic):
            return None
        else:
            return attr_extra_dic

    def list_attr_plot(self):
        attr_extra_dic=self.list_attr(*Pdata._plot_attr_keylist_all)
        if Dic_Test_Empty_Dic(attr_extra_dic):
            return None
        else:
            return attr_extra_dic

    def get_data_as_dic(self,attr_name,taglist='all'):
        """
        Get the spedcified x/y/yerr... or other attribute data as a
            dictionary with tags as keys
        """
        taglist = self._set_default_tag(taglist)
        data_dic=OrderedDict()
        for tag in taglist:
            data_dic[tag] = self.data[tag][attr_name]
        return data_dic

    def shift_ydata(self,shift=None):
        """
        Shift the y data by given shift value in a progressive way, this
            is mainly for comparing the data with the smae y value in a
            more sensible way. i.e., to shift the data a little bit for
            aoviding the overlapping of the lines.
        """
        print "DeprecatingWarning, use shift_data"
        for i,tag in enumerate(self._taglist):
            self.data[tag]['y']=self.data[tag]['y']-shift*i

    def shift_data(self,shift,axis='y'):
        """
        Shift the y data by given shift value in a progressive way, this
            is mainly for comparing the data with the smae y value in a
            more sensible way. i.e., to shift the data a little bit for
            aoviding the overlapping of the lines.

        Parameters:
        -----------
        axis: 'x' or 'y'
        """
        for i,tag in enumerate(self._taglist):
            self.data[tag][axis]=self.data[tag][axis]-shift*i

    def apply_function(self, func=None, axis=None, taglist='all', copy=False):
        """
        Apply a function either 'x' or 'y' axis or 'both' or 'diff', if
            axis=='diff', func should be supplied with a dictionary by
            using ('x'/'y',x_func/y_func) pairs.

        Parameters:
        -----------
        copy: return if copy if copy==True.
        """
        if copy == True:
            pdtemp = self.copy()
        else:
            pdtemp = self

        taglist=pdtemp._set_default_tag(taglist)
        for tag in taglist:
            if axis == 'x':
                pdtemp.data[tag]['x']=func(pdtemp.data[tag]['x'])
            elif axis == 'y':
                pdtemp.data[tag]['y']=func(pdtemp.data[tag]['y'])
            elif axis == 'both':
                pdtemp.data[tag]['x']=func(pdtemp.data[tag]['x'])
                pdtemp.data[tag]['y']=func(pdtemp.data[tag]['y'])
            elif axis == 'diff':
                if isinstance(func,dict):
                    try:
                        pdtemp.data[tag]['x']=func['x'](pdtemp.data[tag]['x'])
                        pdtemp.data[tag]['y']=func['y'](pdtemp.data[tag]['y'])
                    except KeyError,error:
                        print error
                        print """func should be dictionary of ('x'/'y',
                            x_func/y_func) pairs."""
                else:
                    raise ValueError("""func should be a dictionary
                        by using ('x'/'y', x_func/y_func) pairs.""")
            elif axis in pdtemp.data[tag]:
                pdtemp.data[tag][axis]=func(pdtemp.data[tag][axis])
            else:
                raise ValueError("Unknown axis value")

        return pdtemp


    def _data_complete_check_by_tag(self,tag):
        """
        Check if all the 6 base keys with value not as None has the same
            length and return key list with valid and invalid(None) value.
        """
        tagdic=Dic_Extract_By_Subkeylist(self.data[tag],
                                         Pdata._data_base_keylist)
        #get valid key and value list
        valid_key_list=[]
        valid_value_list=[]
        for key,value in tagdic.items():
            if value is None:
                pass
            else:
                valid_key_list.append(key)
                valid_value_list.append(value)
        #check if valid_key_list have unequal length of value
        for key1 in valid_key_list:
            for key2 in valid_key_list:
                if key1!=key2:
                    len1=len(tagdic[key1])
                    len2=len(tagdic[key2])
                    if len1!=len2:
                        raise ValueError("length of '{0}' data is {1} but lenghth of '{2}' data is {3} in tag '{4}'".format(key1,len1,key2,len2,tag))
        invalid_key_list=StringListAnotB(tagdic.keys(),valid_key_list)
        return valid_key_list,invalid_key_list

    def _data_complete_check_all(self):
        for tag in self._taglist:
            _,_=self._data_complete_check_by_tag(tag)

    def _check_empty_attribute(self,attr_name):
        """
        True if the attribute for all tags not specified.
        """
        single_value = pb.ListSingleValue(self.list_attr(attr_name).values())
        if single_value == False:
            return False
        else:
            if single_value.values() in [[],[None]]:
                return True
            else:
                return False

    def _set_random_color_attribute(self,attr_name):
        """
        Set the relevant color attribute values to random values.
        """
        if self._check_empty_attribute(attr_name):
            colorlist = _replace_none_colorlist(None,len(self._taglist))
            self.add_attr_by_tag(**{attr_name:colorlist})
        else:
            pass

    def _get_err_by_tag(self,tag):
        """
        Get xerr,yerr for tag; if both errl and errh is not None, the (n,2)
            array returned; if errl is not None & errh is None,nx1 array returned;
            otherwise None returned.

        Notes:
        ------
        We assume errl=None while errh is not None will not occur.
        """
        tag_data=self.data[tag]
        valid_key_list,invalid_key_list=self._data_complete_check_by_tag(tag)
        #here we assume when equal low and high end of error is used, it's
        #been assigned to x/yerrl with x/yerrh as None. if you want to set 
        #error bar for only side, then set the other one explicitly to 0.
        if 'xerrl' in invalid_key_list and 'xerrh' in invalid_key_list:
            xerr=None
        elif 'xerrl' in invalid_key_list and 'xerrh' not in invalid_key_list:
            raise ValueError('''strange that 'xerrl' is None but 'xerrh' not
                for tag '{0}', set one side as zero if you want
                single-side errorbar'''.format(tag))

        elif 'xerrl' not in invalid_key_list and 'xerrh' in invalid_key_list:
            xerr=tag_data['xerrl']
        else:
            xerr=np.ma.vstack((tag_data['xerrl'],tag_data['xerrh']))

        if 'yerrl' in invalid_key_list and 'yerrh' in invalid_key_list:
            yerr=None
        elif 'yerrl' in invalid_key_list and 'yerrh' not in invalid_key_list:
            raise ValueError('''strange that 'yerrl' is None but 'yerrh' not
                for tag '{0}', set one side as zero if you want
                single-side errorbar'''.format(tag))
        elif 'yerrl' not in invalid_key_list and 'yerrh' in invalid_key_list:
            yerr=tag_data['yerrl']
        else:
            yerr=np.ma.vstack((tag_data['yerrl'],tag_data['yerrh']))

        return xerr,yerr

    @staticmethod
    #TESTED
    def _expand_by_keyword(taglist,list_tagkw_tagkwvalue_tuple):
        """
        Purpose: convert a list of [(tag_kw1,v1),(tag_kw2,v2)] tuples to a
            list like [(tag1_kw1:v1),(tag2_kw1,v1),(tag3_kw2,v2),
            (tag4_kw2,v2),...], where tag1_kw1,tag2_kw1 are tags containg
            keyword tag_kw1, tag3_kw2,tag4_kw2 are tags containg keyword
            tag_kw2,..
        list_tag_tagvalue_tuple is like [('dry','r'),('wet','b')] where
            'dry' and 'wet' are keywords used for classifying tags.

        Examples:
        ---------
        >>> tags = ['wet1', 'wet2', 'wet3', 'dry1', 'dry3', 'dry2']
        >>> Pdata._expand_by_keyword(tags,[('dry','r'),('wet','b')])
        [('dry1', 'r'), ('dry3', 'r'), ('dry2', 'r'), ('wet1', 'b'),
         ('wet2', 'b'), ('wet3', 'b')]
        """
        full_tagkw_list=[]
        for tagkw,tagkwvalue in list_tagkw_tagkwvalue_tuple:
            for tag in FilterStringList(tagkw,taglist):
                full_tagkw_list.append((tag,tagkwvalue))
        return full_tagkw_list


    #TESTED
    @staticmethod
    def _expand_tag_value_to_dic(taglist,tag_attr_value,tagkw=False):
        '''
        Check notes for Pdata.add_attr_by_tag for more details.

        Notes:
        ------
        1. If tag_attr_value is a dict, the taglist will not be used at all.
            the tag_attr_value will be returned without any change.
        '''
        #This if/else build the final dic to be used.
        if not isinstance(tag_attr_value,dict):
            if isinstance(tag_attr_value,list):
                #tag_attr_value is a list of (tag,attr_value) tuples.
                if isinstance(tag_attr_value[0],tuple):
                    #tag is keyword
                    if tagkw==True:
                        expand_tag_attr_value_list = \
                            Pdata._expand_by_keyword(taglist,
                                                     tag_attr_value)
                        final_dic=dict(expand_tag_attr_value_list)
                    elif tagkw == False:
                        final_dic=dict(tag_attr_value)
                    else:
                        raise TypeError('''tagkw must be True or False
                            when tag_attr_value is a list''')
                #a list of values
                else:
                    if len(tag_attr_value)!=len(taglist):
                        raise ValueError('''taglist has len '{0}' but input
                            list len is {1}'''.format(len(taglist),
                            len(tag_attr_value)))
                    else:
                        final_dic=dict(zip(taglist,tag_attr_value))
            #assume a single value (number or string)
            else:
                #use tagkw as string to broadcast vlaue to few tags
                if isinstance(tagkw,str):
                    taglist_bykeywd = FilterStringList(tagkw,taglist)
                    final_dic=dict(zip(taglist_bykeywd, len(taglist_bykeywd)*
                                   [tag_attr_value]))
                #broadcast value to all tags
                elif tagkw == False:
                    final_dic=dict(zip(taglist, len(taglist)*
                                   [tag_attr_value]))
                else:
                    raise TypeError('''tagkw must be string when a single
                        value to be broadcase to all the tags''')
        #tag_attr_value is a dict
        else:
            final_dic=tag_attr_value
        return final_dic

    #TESTED
    def add_attr_by_tag(self,tagkw=False,**nested_attr_tag_value_dic):
        """
        Add extra base attribute or ploting attribute by using key/keyvalue
            pairs, keys can principally be any keys of self._taglist; but this
            method is suggested to be used to add attr_name in
            _extra_base_keylist or _plot_attr_keylist_all

        Notes:
        ------
        Note the keyvalue is very flexible:
        1. In case of a single value, it will be broadcast to all tags for the
            attr concerned.
        2. In case of a list with lenght equal to number of tags, it will be
            add to tags by sequence of Pdata._taglist
        3. In case of a dictionary of tag/attr_value pairs, add attr_value
            accordingly to the tag corresponded.
        4. In case of a list of (tag,value) tuples:
            4.1 if tagkw==True (tagkw is only for this purpose):
                treat the tag in tag/attr_value as tag_keyword to set the same
                attr_value for all tags that contains this tag_keyword.
            4.2 if tagkw==False:
                treat the tag in tag/attr_value as a full tag and will not do
                the keyword search, it will change the tuple directly to a
                dictionary and apply the dictionary in seting tag/attr_value
                directly.

        Available keys are:
        **extra base attribute:
            bleftshift --> bar plot left shift, when bars are very close to
                each other with identical x values, some could be shift
                leftward to seperate them.(default 0)
            bwidth --> barplot width,(default: 0.5)
            bbottom --> barplot bottom (default:0)
            label
        **scatter plot:
             ssize
             scolor
             smarker
             scmap
             snorm
             svmin
             svmax
             + all other kwargs in axes.scatter()
         **errorbar plot:
             'efmt':None
             'ecolor':'b'
             'elinewidth':None
             'capsize':3
             'barsabove':False
             'lolims':False
             'uplims':False
             'xlolims':False
             'xuplims':False
             + all other kwargs in axes.errorbar()
        **plot plot:
        **bar:
        """
        if not isinstance(tagkw,(bool,str)):
            raise TypeError('''tagkw not bool or str type.''')
        for attr_name,tag_attr_value in nested_attr_tag_value_dic.items():
            final_dic = Pdata._expand_tag_value_to_dic(self._taglist,
                                                     tag_attr_value, tagkw)
            #apply value to tag
            for tag,attr_value in final_dic.items():
                if tag not in self.data:
                    raise ValueError("tag '{0}' dose not exist".format(tag))
                else:
                    self.data[tag][attr_name]=attr_value

    def set_default_plot_attr(self):
        pass

    def _fill_errNone_with_Nan(self,tag):
        tag_data=self.data[tag].copy()
        data_len=len(tag_data['x'])
        if tag_data['xerrl'] is None and tag_data['xerrh'] is None:
            tag_data['xerrl']=tag_data['xerrh']=np.repeat(np.nan,data_len)
        elif tag_data['xerrl'] is not None and tag_data['xerrh'] is None:
            tag_data['xerrh']=tag_data['xerrl']
        elif tag_data['xerrl'] is None and tag_data['xerrh'] is not None:
            raise ValueError('''err low end is None but high end not for xerr
                in tag '{0}' '''.format(tag))
        else:
            pass
        if tag_data['yerrl'] is None and tag_data['yerrh'] is None:
            tag_data['yerrl']=tag_data['yerrh']=np.repeat(np.nan,data_len)
        elif tag_data['yerrl'] is not None and tag_data['yerrh'] is None:
            tag_data['yerrh']=tag_data['yerrl']
        elif tag_data['yerrl'] is None and tag_data['yerrh'] is not None:
            raise ValueError('''err low end is None but high end not for yerr
                in tag '{0}' '''.format(tag))
        else:
            pass
        return tag_data

    @staticmethod
    def Hstack_Dic_By_Key(dict_list,key_list):
        """
        Horizontal stack dic values from dict list for keys present in
            key_list. Return a dict.
        Dictionaries in dict_list must have exactly the same keys with
            value as np.ndarray type
        """
        outdic={}
        for key in key_list:
            value=[]
            for subdict in dict_list:
                value.append(subdict[key])
            outdic[key]=np.ma.hstack(tuple(value))
        return outdic

    def Vstack_By_Tag(self,tagseq=None,axis='y'):
        """
        Vertial stack the data into numpy array for the tags in tagseq, the
            1st tag' data is on bottom output array.

        Notes:
        ------
        1. implicite case to call this function is tags have sharex data.
        """
        tagseq = _replace_none_by_given(tagseq,self._taglist)
        data_list = [self.data[tag][axis] for tag in tagseq]
        #we need to reverse the data_list as
        #we want the first come-in to stay on the bottom
        return np.ma.vstack(data_list[::-1])

    def _build_pdata_vstack(self):
        arr = self.Vstack_By_Tag()
        #we want the cumsum with direction from bottom-->top.
        arr_cumsum = arr[::-1].cumsum(axis=0)[::-1]
        pdnew = self.copy()
        for index,tag in enumerate(self._taglist):
            pdnew.data[tag]['y'] = arr_cumsum[::-1][index]
        return pdnew

    #TESTED
    def pool_data_by_tag(self,tagkw=False,**group_dic):
        """
        Pool the data together by specifying new_tag=[old_tag_list] pairs;
            when tagkw==True, it's allowed to use new_tag=old_tag_keyword to
            specify what old tags will be pooled together which will
            include old_tag_keyword. Other attributes outside the
            _data_base_keylist will be copied from the first old_tag of
            the old_tag_list.
        Example:
            pool_data_by_tag(alldry=['1stdry','2nddry','3rddry'],
                allwet=['wet1','wet2'])
            pool_data_by_tag(tagkw=True,alldry='dry')
        """
        if tagkw==True:
            group_dic_final={}
            tags=self._taglist
            for newtag,newtag_tagkw in group_dic.items():
                group_dic_final[newtag]=FilterStringList(newtag_tagkw,tags)
        else:
            group_dic_final=group_dic

        pdata=Pdata()
        for newtag in group_dic_final.keys():
            old_tag_list=group_dic_final[newtag]
            old_entry_list=[self._fill_errNone_with_Nan(tag) for
                                tag in old_tag_list]
            new_entry=Pdata.Hstack_Dic_By_Key(old_entry_list,
                                              Pdata._data_base_keylist)
            #for _extra_base_keylist attributes, their default value in
            #_new_entry will be supplied to the new tag (pooled) data.
            #all other ploting attributes in _plot_attr_dic.values(),
            #they're lost.
            first_old_tag=old_tag_list[0]
            new_entry.update(Dic_Remove_By_Subkeylist(self.data[first_old_tag],
                                                      Pdata._data_base_keylist))
            pdata.add_entry_by_dic(**{newtag:new_entry})
        return pdata

    def regroup_data_by_tag(self,taglist,exclude=False):
        """
        Subset data by "taglist" and return as a new Pdata instance.
        With all features for tag reserved.

        Parameters:
        -----------
        1. exclude: use exclude for reverse selection.

        """
        if exclude == False:
            taglist = taglist
        else:
            taglist = StringListAnotB(self._taglist,taglist)

        if len([tag for tag in taglist if tag not in self._taglist]) > 0:
            raise KeyError("extract tag not present in the taglist!")
        else:
            targetdic=Dic_Extract_By_Subkeylist(self.data,taglist)
            pdata=Pdata(targetdic)
            return pdata

    def __getitem__(self,taglist):
        """
        simple implementation of Pdata.Pdata.regroup_data_by_tag.
        """
        if isinstance(taglist,str):
            taglist = [taglist]
        pd = self.regroup_data_by_tag(taglist)
        pd.set_tag_order(taglist)
        return pd

    def data_ix(self,ls):
        """
        Use [tag,item] list to retrieve data for one tag.
        """
        return self.data[ls[0]][ls[1]]

    def __repr__(self):
        return '\n'.join([repr(self.__class__),"tags:",','.join(self.taglist)])

    def __len__(self):
        return len(self.taglist)

    def regroup_data_by_tag_keyword(self,tag_keyword):
        tags=self._taglist
        return self.regroup_data_by_tag(FilterStringList(tag_keyword,tags))

    def leftshift(self,shift=0,taglist='all'):
        taglist=self._set_default_tag(taglist)
        for tag in taglist:
            self.data[tag]['x']=self.data[tag]['x']-shift

    #a wrapper of scatter plot function
    def _gscatter(self,axes,x,y,**kwargs):
        attr_dic={             \
                 'ssize':20,   \
                 'scolor':'k', \
                 'smarker':'o',\
                 'scmap':None, \
                 'snorm':None, \
                 'svmin':None, \
                 'svmax':None, \
                 }
        attr_dic.update(Dic_Extract_By_Subkeylist(kwargs,Pdata._scatter_attr_base_keylist))
        remain_kwargs=Dic_Remove_By_Subkeylist(kwargs,Pdata._plot_attr_keylist_all)
        #print 'attr_dic kwargs passed to axes.scatter()',attr_dic
        #print 'scatter kwargs passed to axes.scatter()',remain_kwargs
        return axes.scatter(x, y,s=attr_dic['ssize'], c=attr_dic['scolor'],
                            marker=attr_dic['smarker'], cmap=attr_dic['scmap'],
                            norm=attr_dic['snorm'], vmin=attr_dic['svmin'],
                            vmax=attr_dic['svmax'], alpha=None,
                            linewidths=None, faceted=True,
                            verts=None, **remain_kwargs)

    def scatter(self,axes=None,erase=True,legend=True,**kwargs):
        """
        Make a scatter plot.
        Parameters:
        -----------
        erase : set erase as True if a scatter plot has already been made
            for this pd object and you want to make a new scatter plot
            and "erase" the existing one
        """
        self._set_random_color_attribute('scolor')
        axes=_replace_none_axes(axes)
        if erase==True:
            if hasattr(self,'Scatter_PathC') and self.Scatter_PathC != {}:
                self.remove_scatter_by_tag()

        self._data_complete_check_all()
        if Is_Nested_Dic(kwargs):
            raise ValueError('''tag specific plot attributes should be set
                              by using add_attr_by_tag''')
        else:
            self.Scatter_PathC={}  #Scatter_PathC ---> Scatter PathCollection
            for tag,tag_data in self.data.items():
                tag_kwargs=kwargs.copy()
                tag_plot_attr_dic=Dic_Remove_By_Subkeylist(tag_data,Pdata._all_nonplot_keylist)
                tag_kwargs.update(tag_plot_attr_dic)
                #print 'tag & tag_kwargs',tag,tag_kwargs
                self.Scatter_PathC[tag]=self._gscatter(axes,tag_data['x'],tag_data['y'],**tag_kwargs)
        if legend == True:
            self.set_legend('scatter')
        return self.Scatter_PathC


    def plot(self,*args,**kwargs):
        """
        A wrapper of matplotlib.axes.Axes.plot

        kwargs:
        -------
        axes: forced axes.
        legend: boolean type. Set False to supress the legend.
        """

        prior_keylist = ['ax','axes','legend']
        ax=kwargs.get('ax',None)
        axes=kwargs.get('axes',None)
        if ax is None and axes is None:
            axes = None
        elif ax is not None:
            axes =ax
        elif axes is not None:
            axes = axes
        else:
            raise ValueError("receive both axes and ax")
        legend=kwargs.get('legend',True)

        for key in prior_keylist:
            kwargs.pop(key,None)

        axes=_replace_none_axes(axes)
        if hasattr(self,'Line2D') and self.Line2D!={}:
            self.remove_line_by_tag()
        self.Line2D={}
        for tag,tag_data in self.data.items():
            kwargs.update({'label':tag_data['label']})
            #pdb.set_trace()
            line2D=axes.plot(tag_data['x'],tag_data['y'],*args,**kwargs)
            self.Line2D[tag]=line2D[0]
        if legend == True:
            if self._check_empty_attribute('label'):
                self.set_legend('line',taglab=True)
            else:
                self.set_legend('line',taglab=False)
        return self.Line2D

    def plot_twinx(self,left=None,right=None,
                   leftcolor='r',rightcolor='b',
                   set_axis=True,show_ylab=True,
                   axes=None,
                   **kwargs):
        """
        Plot two different tags on the twinx plot.

        Parameters:
        -----------
        left,right: the tags used to plot on the left and right vertical axis.
        leftcolor/rightcolor: the colors used for the plot on the left and
            right axis.
        set_axis: boolean value. Set True to change accorindingly the
            axis color, label color, and tickline color as the same for the
            plot color.
        axes: used for the base axes.
        """
        axes=_replace_none_axes(axes)
        axt = axes.twinx()
        self.Line2D = OrderedDict()

        if left is None:
            left = self._taglist[0]
        if right is None:
            right = self._taglist[1]

        line1 = axes.plot(self.data[left]['x'],self.data[left]['y'],color=leftcolor,
                  **kwargs)
        line2 = axt.plot(self.data[right]['x'],self.data[right]['y'],color=rightcolor,
                 **kwargs)

        if set_axis == True:
            axes.spines['left'].set_color(leftcolor)
            axes.spines['right'].set_color(rightcolor)
            g.setp(axes.get_yticklines(),color=leftcolor);
            g.setp(axes.get_yticklabels(),color=leftcolor);

            g.setp(axt.get_yticklines(),color=rightcolor);
            g.setp(axt.get_yticklabels(),color=rightcolor);

        self.axdic=OrderedDict(zip([left,right],[axes,axt]))
        self.lax = LabelAxes.LabelAxes(tags=[left,right],axl=[axes,axt])
        self.Line2D[left] = line1[0]
        self.Line2D[right] = line2[0]
        if show_ylab:
            axes.set_ylabel(left)
            axt.set_ylabel(right)
        return self.Line2D

    def plot_stackline(self,axes=None,tagseq=None,colors=None,
                       bottom_fill=True,legend=True,legdic={},fillkw={}):
        """
        Make stacked line plot for sharex pdata.

        Parameters:
        -----------
        tagseq: the tag list for which stacked line plot will be made.
            Notice the sequece for tagseq is from bottom to the top.
        colors: color list, the length should be equal to the number of
            filled area. In case of bottom_fill == True, len(colors) should
            be equal to len(tagseq), otherwise should be equal to
            len(tagseq)-1.
        bottom_fill: set True if the area between xaxis and the bottom
            line (the first tag) is to be filled.
        legdic: the legend is set by using g.ProxyLegend(), legdic could
            be passed as kwargs for the function of
            g.ProxyLegend.create_legend
        fillkw: used for plt.fill_between functions.

        Returns:
        --------
        pdata,proleg
        pdata: the new pdata that stores the cumsum 'y' data.
        proleg: the g.ProxyLegend object that's used to creat legend for
            the stacked plots.

        Notes:
        ------
        1. the **kwargs is also passed directly as kwargs for the
            function mat.patches.Rectangle and this may lead to
            conflicts in some special cases.

        See also:
        ---------
        g.ProxyLegend, mat.patches.Rectangle
        """
        #build the new pdata that stores the cumsum data
        tagseq = _replace_none_by_given(tagseq,self._taglist)
        arr = self.Vstack_By_Tag(tagseq)
        #we want the cumsum with direction from bottom-->top.
        arr_cumsum = arr[::-1].cumsum(axis=0)[::-1]
        pdnew = self.copy()
        for index,tag in enumerate(tagseq):
            pdnew.data[tag]['y'] = arr_cumsum[::-1][index]

        axes=_replace_none_axes(axes)
        #treat case of bottom_fill
        if bottom_fill == True:
            colors = _replace_none_colorlist(colors,num=len(tagseq))
            leg_tagseq = tagseq[:]
        else:
            colors = _replace_none_colorlist(colors,num=len(tagseq)-1)
            leg_tagseq = tagseq[1:]

        proleg = g.ProxyLegend()
        if bottom_fill == True:
            bottom_tag = tagseq[0]
            bottom_data = pdnew.data[bottom_tag]
            axes.fill_between(bottom_data['x'],
                              np.zeros(len(bottom_data['x'])),
                              bottom_data['y'],color=colors[0],**fillkw)

            proleg.add_rec_by_tag(bottom_tag,color=colors[0],**fillkw)
            colors_remain = colors[1:]
        else:
            colors_remain = colors

        for index in range(len(tagseq)-1):
            below_tag = tagseq[index]
            focus_tag = tagseq[index+1]
            below_data = pdnew.data[below_tag]
            focus_data = pdnew.data[focus_tag]
            axes.fill_between(focus_data['x'],below_data['y'],
                              focus_data['y'],color=colors_remain[index],
                              **fillkw)

            proleg.add_rec_by_tag(focus_tag,
                                  color=colors_remain[index],**fillkw)
            #if index == len(tagseq)-2:
                #axes.set_xlim(focus_data['x'][0],focus_data['x'][-1])

        if legend:
            proleg.create_legend(axes,tagseq=leg_tagseq[::-1],**legdic)
        else:
            pass
        self.stackline_proleg = proleg
        self.stackline_actual_pd = pdnew
        self.axes = axes

    def _gerrorbar(self,axes,x,y,yerr=None,xerr=None,**kwargs):
        attr_dic={             \
                 'efmt':None,\
                 'ecolor':'r',   \
                 'elinewidth':None, \
                 'capsize':3, \
                 'barsabove':False, \
                 'lolims':False,\
                 'uplims':False,\
                 'xlolims':False,\
                 'xuplims':False \
                 }
        attr_dic.update(Dic_Extract_By_Subkeylist(kwargs,Pdata._error_attr_base_keylist))
        remain_kwargs=Dic_Remove_By_Subkeylist(kwargs,Pdata._plot_attr_keylist_all)
        #print 'attr_dic in passed to errorbar()',attr_dic
        #print 'kwargs passed to errorbar()',remain_kwargs
        return axes.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=attr_dic['efmt'],
                             ecolor=attr_dic['ecolor'],
                             elinewidth=attr_dic['elinewidth'],
                             capsize=attr_dic['capsize'],
                             barsabove=attr_dic['barsabove'],
                             lolims=attr_dic['lolims'],
                             uplims=attr_dic['uplims'],
                             xlolims=attr_dic['xlolims'],
                             xuplims=attr_dic['xuplims'], **remain_kwargs)



    def _set_ecolor_by_scatter(self):
        taglist=[]
        colorlist=[]
        for tag in self.data.keys():
            if tag in self.Scatter_PathC:
                taglist.append(tag)
                try:
                    fc = self.Scatter_PathC[tag].get_facecolor()[0]
                    #if facecolor is white, we want to use edgecolor
                    if np.allclose(fc,np.ones(4)):
                        colorlist.append(self.Scatter_PathC[tag].get_edgecolor()[0])
                    else:
                        colorlist.append(fc)
                except IndexError:
                    #print "the scatter facecolor for tag '{0}' is none, scatter edgecolor is used as errorbar color.".format(tag)
                    colorlist.append(self.Scatter_PathC[tag].get_edgecolor()[0])
        self.add_attr_by_tag(ecolor=dict(zip(taglist,colorlist)))
        return dict(zip(taglist,colorlist))

    def _set_ecolor_by_bar(self):
        taglist=[]
        colorlist=[]
        for tag in self.data.keys():
            if tag in self.Bar_Container:
                taglist.append(tag)
                try:
                    colorlist.append(self.Bar_Container[tag][0].get_facecolor())
                except IndexError:
                    #print "the bar facecolor for tag '{0}' is none, bar edgecolor is used as errorbar color.".format(tag)
                    colorlist.append(self.Bar_Container[tag][0].get_edgecolor())
        self.add_attr_by_tag(ecolor=dict(zip(taglist,colorlist)))



    @staticmethod
    def _get_attr_value_from_handle(handle,attr_name):
        """
        Return an attribute from the specified handle.
        Example:
            _get_attr_from_handle(matplotlib.collections.PathCollection,
                                  '_facecolors')
            is the same as
            matplotlib.collections.PathCollection.get_facecolor()
        """
        return handle.__getattribute__(attr_name)

    #TESTED
    @staticmethod
    def _get_attr_sequential_check(handle,func_attr_condition_tuple_list):
        """
        Purpose: Return a single attribute by checking sequentially the
            attribute name list and if the condition has been made. if
            the condition has been made, then the returned attribute
            value will be discarded, otherwise it will be used and the
            following check will not be done.

        FirstCase: We want to make the program as intelligent as possible,
            if we want to set the errorbar color according to the facecolor
            or edgecolor of the scatters. Sometimes the facecolor of a
            scatter could be None, then we want to use the edgecolor as
            the errorbar color. In this case, we need to set some priority:
            if the facecolor is not None, then the facecolor will be
            returned; otherwise the edgecolor will be returned.
        Parameters:
        -----------
        attr_condition_tuple_list:
            a list of 3-length tuple
            (func, attr_name, discard_condition_value). For the tuple, the
            first element of is a function, the second element is the
            attribute name, and and the third element is used to check if
            the attribute value should be discard. Sometimes the attributed
            value could be checked directly as equal or not to the
            given (third) value (in this case the first element
            will be give as None), sometimes it's not possible to check
            this directly, so we need to use a function (first tuple element)
            to check if the condtion to discard the attribute value is met.
            If the condition is met (either through a function) or by equal
            relation operation directly, the attribute value will be discard.
            If all given conditions for all the attribute names are met,
            then an error will be raised.
        Example:
            _get_attr_sequential_check(
                matplotlib.collections.PathCollection,
                [(__builtin__.len,'_facecolors',0),
                 (__builtin__.len,'_edgecolors',0)]).
                This means if both facecolor and edgecolor is np.array([]),
                then an error will be raised. But if facecolor is not
                np.array([]), then the color of the facecolor will be
                used (This valid attribute value that stays earlier has
                higher priority in determining the final return value).
        Doctest:
            >>> fig, ax = g.Create_1Axes()
            >>> a = np.arange(10)
            >>> col = ax.scatter(a,a,facecolor=np.array([ 0.,  0.,  1.,  1.]),
                edgecolor=np.array([ 1.,  0.,  0.,  1.]))
            >>> _get_attr_sequential_check(col,[(__builtin__.len,'_facecolors',0),(__builtin__.len,'_edgecolors',0)])[0] == np.array([ 0.,  0.,  1.,  1.])
            >>> _get_attr_sequential_check(col,[(__builtin__.len,'_edgecolors',0),(__builtin__.len,'_facecolors',0)]) == np.array([ 1.,  0.,  0.,  1.])
        """
        error_state = True
        for (func, attr_name, discard_condition_value) in func_attr_condition_tuple_list:
            valid_attribute_value = False
            attribute_value = Pdata._get_attr_value_from_handle(handle,attr_name)
            #attribute value and discard_condition_value could be compared directly
            if func is None:
                if attribute_value == discard_condition_value:
                    pass
                else:
                    valid_attribute_value = True
            #we needs a function to check the attribute value is valid
            else:
                if func(attribute_value) == discard_condition_value:
                    pass
                else:
                    valid_attribute_value = True

            #we return the valid attribute value
            if valid_attribute_value:
                error_state = False
                return attribute_value
        if error_state:
            raise ValueError("not valid attribute value found for the list of attributes: {0}".format(zip(*attr_condition_tuple_list)[0]))

    #Note the previous errorbar is not removed before the new one is drawn
    def errorbar(self,axes,ef='scatter',**kwargs):
        """
        ef is short for 'ecolor follow', ef could be 'scatter','bar', if None, ecolor in self.data[tag]['ecolor'] is used, or it not present in self.data,
        _error_attr_default value will be used.
        """
        self._errleftdic = self.get_data_as_dic('x')

        self._data_complete_check_all()
        if ef=='scatter':
            self._set_ecolor_by_scatter()
        elif ef == 'bar':
            self._set_ecolor_by_bar()
            self._errleftdic = self._barleftdic
        elif ef=='none':
            pass
        else:
            raise ValueError("incorrect ef value '{0}'".format(ef))
        if Is_Nested_Dic(kwargs):
            raise ValueError('tag specific plot attributes should be set by using add_attr_by_tag')
        else:
            self.Errorbar_Lines={}
            for tag,tag_data in self.data.items():
                tag_xerr,tag_yerr=self._get_err_by_tag(tag)
                if (tag_xerr is None and tag_yerr is None):
                    continue
                else:
                    #print tag_xerr,tag_yerr
                    tag_kwargs=kwargs.copy()
                    tag_plot_attr_dic=Dic_Remove_By_Subkeylist(tag_data,Pdata._all_nonplot_keylist)
                    tag_kwargs.update(tag_plot_attr_dic)  #here, tag_plot_attr_dic contains keys for all ploting types; tag_kwargs get contaminated.
                    self.Errorbar_Lines[tag]=self._gerrorbar(axes,self._errleftdic[tag],tag_data['y'],xerr=tag_xerr,yerr=tag_yerr,**tag_kwargs)
        return self.Errorbar_Lines

    def _gbar(self,axes,left,height,width=0.5,bottom=0,label=None,**kwargs):
        attr_dic=Pdata._bar_attr_default.copy()
        attr_dic.update(Dic_Extract_By_Subkeylist(kwargs,Pdata._bar_attr_base_keylist))
        kwargs=Dic_Remove_By_Subkeylist(kwargs,Pdata._plot_attr_keylist_all)
        #print '_gbar',kwargs
        #print '_bar',attr_dic
        return axes.bar(left, height, width=width, bottom=bottom, label=label, **kwargs)

    def bar(self,axes=None,xticklabel=None,stacked=False,legend=True,**kwargs):

        if stacked==False:
            for i,tag in enumerate(self._taglist):
                self.data[tag]['bbottom'] = 0
        else:
            pdstack = self._build_pdata_vstack()
            for i,tag in enumerate(self._taglist):
                if i ==0:
                    pass
                else:
                    pretag = self._taglist[i-1]
                    self.data[tag]['bbottom'] = pdstack.data[pretag]['y']

        self._barleftdic = {}
        axes=_replace_none_axes(axes)
        self._data_complete_check_all()
        if Is_Nested_Dic(kwargs):
            raise ValueError('tag specific plot attributes should be set by using add_attr_by_tag')
        else:
            self.Bar_Container={}  #Bar_Container --> <Container object of 10 artists> with matplotlib.patches.Rectangle as list members.
            for i,tag in enumerate(self._taglist):
                tag_data = self.data[tag]
                tag_kwargs=kwargs.copy()
                tag_plot_attr_dic=Dic_Remove_By_Subkeylist(tag_data,Pdata._all_nonplot_keylist)
                tag_kwargs.update(tag_plot_attr_dic)
                #print 'tag & tag_kwargs',tag,tag_kwargs
                _barleft = tag_data['x']-tag_data['bleftshift']*i-tag_data['bwidth']*0.5
                self.Bar_Container[tag]=self._gbar(axes,_barleft,tag_data['y'],width=tag_data['bwidth'],\
                                                   bottom=tag_data['bbottom'],\
                                                   label=tag_data['label'],**tag_kwargs)
                self._barleftdic[tag] = _barleft+tag_data['bwidth']*0.5
                temp_bwidth = tag_data['bwidth']
            xarray = np.array(self._barleftdic.values())
            xtickpos = xarray.mean(axis=0)
            xmin = np.min(xarray)-temp_bwidth
            xmax = np.max(xarray)+temp_bwidth
            axes.set_xlim(xmin,xmax)
            axes.set_xticks(xtickpos)
            if xticklabel is not None:
                axes.set_xticklabels(xticklabel)
            if 'color' not in kwargs:
                colors = _replace_none_colorlist(num=len(self))
                self.setp_tag(plottype='bar',color=colors)
            if legend == True:
                if self._check_empty_attribute('label'):
                    self.set_legend('bar',taglab=True)
                else:
                    self.set_legend('bar',taglab=False)
            self.bar_xticks = xtickpos
            return self.Bar_Container

    def add_text(self, text=None, fontdict=None, withdash=False, **kwargs):
        """
        A wrapper of mat.axes.Axes.text to add text at the corresponding
            (x,y) positions.

        kwargs:
        -------
        text: could be of type string of list. In case of list, the length
            must be equal to the length of (x,y) pairs.
        """

        prior_keylist = ['ax','axes']
        ax=kwargs.get('ax',None)
        axes=kwargs.get('axes',None)
        if ax is None and axes is None:
            axes = None
        elif ax is not None:
            axes =ax
        elif axes is not None:
            axes = axes
        else:
            raise ValueError("receive both axes and ax")

        for key in prior_keylist:
            kwargs.pop(key,None)

        axes=_replace_none_axes(axes)

        for tag,tag_data in self.data.items():
            if isinstance(text,str):
                text = [text] * len(tag_data['x'])
            elif isinstance(text,list):
                if len(text) != len(tag_data['x']):
                    raise ValueError("text list not in equal length of x/y data")
                else:
                    pass
            else:
                raise TypeError("text could only be str or list")

            for x,y,s in zip(tag_data['x'],tag_data['y'],text):
                axes.text(x,y,s,fontdict=fontdict, withdash=withdash, **kwargs)


    def _get_plot_attr_value_from_data(self,*attr_list):
        """
        Get plot attribute value from data. attr_list must be subset of Pdata._plot_attr_keylist_all
        """
        attr_dic={}
        for attr_name in attr_list:
            attr_dic[attr_name]=dict()
            for tag,tag_data in self.data.items():
                try:
                    attr_dic[attr_name][tag] = tag_data[attr_name]
                except KeyError:
                    attr_dic[attr_name][tag] = Pdata._plot_attr_default[attr_name]
        return attr_dic

    def _get_scatter_marker_size_color(self):
        self.outdic={}
        self.outdic['color']=_get_attr_value_from_objins_dic(self.Scatter_PathC,'_facecolors').values()[0]
        self.outdic['size']=self._get_plot_attr_value_from_data('ssize').values()[0]
        self.outdic['marker']=self._get_plot_attr_value_from_data('smarker').values()[0]
        return self.outdic

    def _get_Scatter_PathC_marker_size_color(self):
        sc=self._get_scatter_marker_size_color()
        outdic={}
        for tag,ScatterP in self.Scatter_PathC.items():
            outdic[(tag,ScatterP)]=dict()
            for attr_name,attr_dic in sc.items():
                outdic[(tag,ScatterP)][attr_name]=attr_dic[tag]
        return outdic


    def _get_handler_label_by_artistdic(self,artist_dic,taglab=False,
                                        tag_seq=None):
        """
        Retrieve handler and label list from self.ScatterP/Line2D/
            Bar_Container for legend seting. When tag_seq is a list of tags,
            will return handler/label list as sorted by tags in
            tag_seq, note using tag_seq will restrict the scope of
            tags included in legend.
        """
        handler_list=[]
        label_list=[]
        if tag_seq is None:
            if sorted(self._taglist) == sorted(artist_dic.keys()):
                tag_seq = self._taglist[:]
            else:
                tag_seq=artist_dic.keys()
        for tag in tag_seq:
            artist=artist_dic[tag]
            handler_list.append(artist)
            if taglab==True:
                label_list.append(tag)
            else:
                label_list.append(self.data[tag]['label'])
        return handler_list,label_list

    def _get_axes_by_artistdic(self,artist_dic):
        """
        Get which axes the artisti_dic is in
        """
        artist=artist_dic.values()[0]
        try:
            return artist.axes
        except AttributeError:
            return artist[0].axes

    def set_legend_scatter(self,axes=None,taglab=False,tag_seq=None,**kwargs):
        print "DeprecatingWarning: set_legend_scatter"
        tag_seq = _replace_none_by_given(tag_seq,self._taglist)
        handler_list,label_list=self._get_handler_label_by_artistdic(self.Scatter_PathC,taglab=taglab,tag_seq=tag_seq)
        if axes is None:
            axes=self._get_axes_by_artistdic(self.Scatter_PathC)
        self.LegendScatter=axes.legend(handler_list,label_list,**kwargs)
        return self.LegendScatter

    def set_legend_line(self,axes=None,taglab=False,tag_seq=None,**kwargs):
        print "DeprecatingWarning: set_legend_line"
        handler_list,label_list=self._get_handler_label_by_artistdic(self.Line2D,taglab=taglab,tag_seq=tag_seq)
        if axes is None:
            axes=self._get_axes_by_artistdic(self.Line2D)
        self.LegendLine=axes.legend(handler_list,label_list,**kwargs)
        return self.LegendLine

    def set_legend_bar(self,axes=None,taglab=False,tag_seq=None,**kwargs):
        print "DeprecatingWarning: set_legend_bar"
        handler_list,label_list=self._get_handler_label_by_artistdic(self.Bar_Container,taglab=taglab,tag_seq=tag_seq)
        if axes is None:
            axes=self._get_axes_by_artistdic(self.Bar_Container)
        self.LegendBar=axes.legend(handler_list,label_list,**kwargs)

    def set_data_void(self):
        for tag, tag_value in self.data.items():
            tag_value['x'] = []
            tag_value['y'] = []

    def set_xdata_by_ydata_by_tag(self,tag):
        """
        Fill the xdata of all tags by the ydata of the specified tag.

        Notes:
        ------
        1. Original tag order will be maintained.

        Returns:
        --------
        Pdata object with the used tag being removed.
        """
        remain_taglist = StringListAnotB(self._taglist,[tag])
        targetdic=Dic_Extract_By_Subkeylist(self.data,remain_taglist)
        for temptag in remain_taglist:
            targetdic[temptag]['x'] = self.data[tag]['y']
        return Pdata(targetdic)

    def collapse_xdata_as_tag(self,tag,newxdata=None):
        """
        Collapse the xdata as a tag, use newxdata to set the newxdata
            for all tags.
        """
        dic = self.data.copy()
        if newxdata is None:
            newxdata = np.arange(len(dic[self.taglist[0]]['x']))
        else:
            pass

        for tag,subdic in dic.items():
            dic[tag]['x'] = newxdata
        dic[tag] = {'x':newxdata,'y':self.data[self.taglist[0]]['x']}
        return Pdata(dic)

    def get_handle_label(self,plottype='all',taglab=False,tag_seq=None):
        handler_list=[]
        label_list=[]
        artist_dic = self._get_artist_dic_by_plottype(plottype)
        if isinstance(artist_dic,dict):
            artist_dic_list = [artist_dic]
        else:
            artist_dic_list = artist_dic

        for artdic in artist_dic_list:
            try:
                sub_hand_list,sub_lab_list=\
                    self._get_handler_label_by_artistdic(artdic,
                        taglab=taglab,tag_seq=tag_seq)
                handler_list.extend(sub_hand_list)
                label_list.extend(sub_lab_list)
            except AttributeError:
                pass
        return (handler_list,label_list)

    def get_proleg(self,plottype='all',taglab=True,tag_seq=None):
        (handler_list,label_list) = self.get_handle_label(plottype=plottype,
                    taglab=taglab,tag_seq=tag_seq)
        if len(handler_list) > len(self.taglist):
            raise ValueError('''Warning! More hanlers are found than tgas. There might
            be more than one type of plots initiated, you msut provide
            explicitly the plottype.''')
        else:
            pleg = g.ProxyLegend(dict(zip(label_list,handler_list)))
            pleg.set_tag_order(label_list)
            return pleg

    def set_legend(self,plottype='all',axes=None,taglab=True,
                   twinx=False,
                       tag_seq=None,**kwargs):
        """
        plottype: could be 'line','sca'/'scatter','bar','stackline'
        """
        (handler_list,label_list) = self.get_handle_label(plottype=plottype,
                    taglab=taglab,tag_seq=tag_seq)
        if axes is None:
            handler = handler_list[0]
            if not twinx:
                try:
                    axes=handler.axes
                except AttributeError:
                    if isinstance(handler,mat.container.Container):
                        axes = handler[0].axes
                    else:
                        raise AttributeError("could not retrieve from handler")
            else:
                axes=handler.axes.twinx()
                axes.tick_params(labelright='off')
        self.LegendAll=axes.legend(handler_list,
                                   label_list,**kwargs)

    def set_legend_all(self,axes=None,taglab=False,
                       tag_seq=None,**kwargs):
        print "DeprecatingWarning: set_legend_all"
        self.set_legend('all',axes=axes,taglab=taglab,
                        tag_seq=tag_seq,**kwargs)

    def set_legend_select(self,axes=None,taglab=False,tag_seq=None,**kwargs):
        pass

    def _remove_artist_by_tag(self,artist_dic,taglist='all'):
        #this method from http://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot
        if taglist=='all':
            taglist=artist_dic.keys()
        for tag in taglist:
            if tag in artist_dic:
                art=artist_dic[tag]
                wl=weakref.ref(art)
                art.remove()
                del art
                del artist_dic[tag]

    def remove_line_by_tag(self,taglist='all'):
        self._remove_artist_by_tag(self.Line2D,taglist=taglist)

    def remove_scatter_by_tag(self,taglist='all'):
        self._remove_artist_by_tag(self.Scatter_PathC,taglist=taglist)

    def setp_by_tag(self,artist_dic,**nested_attr_tag_value_dic):
        """
        Set artist property by attr_name=single value, or
            attr_name={tag:attr_value} pairs. In case of single value provided,
            it's broadcast to apply on all artists. In case of a
            {tag:attr_value} dictionary, attr_value applied according to
            its tat key.  Note the {tag:attr_value}.keys() must be exactly
            the same as in artist_dic.keys().
        """
        print "DeprecatingWarning: setp_by_tag"
        for attr_name,tag_attr_value_dic in nested_attr_tag_value_dic.items():
            if not isinstance(tag_attr_value_dic,dict):
                for tag,art in artist_dic.items():
                    plt.setp(art,**{attr_name:tag_attr_value_dic})
            else:
                for tag,attr_value in tag_attr_value_dic.items():
                    plt.setp(artist_dic[tag],**{attr_name:attr_value})


    def _setp_by_tag(self,artist_dic,tagkw=False,**nested_attr_tag_value_dic):
        #retrieve the taglist that's used by the expand function.
        if sorted(artist_dic.keys()) == sorted(self._taglist):
            taglist = self._taglist[:]
        else:
            taglist = artist_dic.keys()

        for attr_name,tag_attr_value in nested_attr_tag_value_dic.items():
            final_dic = Pdata._expand_tag_value_to_dic(taglist,
                                                     tag_attr_value, tagkw)
            for tag,attr_value in final_dic.items():
                plt.setp(artist_dic[tag],**{attr_name:attr_value})

    def _get_artist_dic_by_plottype(self,plottype='all'):
        '''
        Notes:
        ------
        return a list if plottype == 'all'
        '''
        #print "plottype:{0}".format(plottype)
        if plottype == 'line':
            artist_dic = self.Line2D
        elif plottype in ['sca','scatter']:
            artist_dic = self.Scatter_PathC
        elif plottype == 'bar':
            artist_dic = self.Bar_Container
        elif plottype == 'stackline':
            artist_dic = self.stackline_proleg.data
        elif plottype == 'all':
            artist_dic = []
            artist_dic_list = []
            for plotatt in self._default_plottype_list:
                try:
                    artist_dic.append(self.__dict__[plotatt])
                except KeyError:
                    pass
        else:
            try:
                artist_dic = self.__getattribute__(plottype)
            except AttributeError:
                raise AttributeError('''plottype not present! please add new one''')
        return artist_dic


    def setp_tag(self,plottype='all',tagkw=False,**nested_attr_tag_value_dic):
        """
        Set the property by tag.

        Parameters:
        -----------
        plottype: could be 'line','sca'/'scatter','bar','all.',or directly put
            the attribute names that hold the artists.
        """
        artist_dic = self._get_artist_dic_by_plottype(plottype)
        if isinstance(artist_dic,dict):
            self._setp_by_tag(artist_dic,tagkw=tagkw,**nested_attr_tag_value_dic)
        else:
            for sub_artist_dic in artist_dic:
                self._setp_by_tag(sub_artist_dic,tagkw=tagkw,**nested_attr_tag_value_dic)



    def plot_split_axes(self,plotkw={},**kwargs):
        '''
        Plot for each tag a separate subplot.

        Parameters:
        -----------
        plotkw: the keyword used in plt.plot function.
        kwargs:
            force_axs: force the axes.
            tagseq: the tag sequence, default is self._taglist
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag; None or False to suppress.
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        '''
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                        default_tagseq=self._taglist,**kwargs)
        for tag,axt in axdic.items():
            pd_temp=self.regroup_data_by_tag([tag])
            pd_temp.plot(ax=axt,legend=False,**plotkw)
        self.axes = axdic
        self.axdic = axdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def bar_split_axes(self,barkw={},**kwargs):
        '''
        Plot for each tag a separate subplot.

        Parameters:
        -----------
        barkw: the keyword used in plt.plot function.
        kwargs:
            force_axs: force the axes.
            tagseq: the tag sequence, default is self._taglist
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag; None or False to suppress.
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            subkw: kwarg in plt.subplots function
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        '''
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                        default_tagseq=self._taglist,**kwargs)
        for tag,axt in axdic.items():
            pd_temp=self.regroup_data_by_tag([tag])
            pd_temp.bar(axes=axt,legend=False,**barkw)
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def set_text(self,textblock,pos='uc',ftdic={'size':12},**kwargs):
        """
        Set text based on the tags and their axes, by using g.Set_AxText

        Parameters:
        -----------
        textblock: could be  single_value/list/dict/list_of_2len_tuples,
            the same as the keyvalue in the function add_attr_by_tag.
        kwargs: for text method.
        """
        if not hasattr(self,'axdic'):
            raise AttributeError("This is not a seperate axes plot!")
        else:
            textdic = Pdata._expand_tag_value_to_dic(self._taglist,
                                                     textblock,False)
            for tag,text in textdic.items():
                g.Set_AxText(self.axdic[tag],text,pos=pos,ftdic=ftdic,**kwargs)


    def set_xlabel(self,xlabel=None,xunit=False,
                  ha='center',va='center',
                  **kwargs):
        if not hasattr(self,'axdic'):
            raise AttributeError("This is not a seperate axes plot!")
        else:
            self.xlabeldic = {}
            for tag,axt in self.axdic.items():
                try:
                    orlabel = self.data[tag]['xlabel']
                except KeyError:
                    if xlabel is None:
                        orlabel = tag
                    else:
                        orlabel = xlabel

                if xunit == True:
                    try:
                        xunit = self.data[tag]['xunit']
                    except AttributeError:
                        xunit = ''
                else:
                    xunit=''

                if xunit != '':
                    full_xlabel = orlabel+'('+xunit+')'
                else:
                    full_xlabel = orlabel

                self.xlabeldic[tag] = axt.set_xlabel(full_xlabel,
                                                     va=va,
                                                     ha=ha,
                                                     **kwargs)

    def set_ylabel(self,ylabel=None,yunit=False,ha='center',
                   va='center',**kwargs):
        if not hasattr(self,'axdic'):
            raise AttributeError("object has no axdic attribute!")
        else:
            self.ylabeldic = {}
            for tag,axt in self.axdic.items():
                try:
                    orlabel = self.data[tag]['ylabel']
                except KeyError:
                    if ylabel is None:
                        orlabel = tag
                    else:
                        orlabel = ylabel

                if yunit == True:
                    try:
                        yunit = self.data[tag]['yunit']
                    except AttributeError:
                        yunit = ''
                else:
                    yunit=''

                if yunit != '':
                    full_ylabel = orlabel+'('+yunit+')'
                else:
                    full_ylabel = orlabel

                self.ylabeldic[tag] = axt.set_ylabel(full_ylabel,
                                                     va=va,
                                                     ha=ha,
                                                     **kwargs)


    def plot_OLS_reg(self,taglist='all',color='k',ls='--',
                     PosEquation='uc',
                     precision_slope=3, precision_inter=3, **kwargs):
        """
        Add OLS regression line.
        """
        taglist = self._set_default_tag(taglist)
        self.OLSlinedic = {}
        OLSresultdic = {}
        for tag in taglist:
            line,OLSre_list = g.plot_OLS_reg(self.axdic[tag],
                                        self.data[tag]['x'],
                                        self.data[tag]['y'],
                                        c=color,ls=ls,
                                        PosEquation=PosEquation,
                                        precision_slope=precision_slope,
                                        precision_inter=precision_inter,
                                        **kwargs)
            self.OLSlinedic[tag] = line
            OLSresultdic[tag] = OLSre_list
        OLSre_colname = ['slope','intercept','r_value','p_value','stderr']
        dft1 = pa.DataFrame(OLSresultdic,index=OLSre_colname)
        dft2 = pa.DataFrame({'R2':dft1.ix['r_value']**2})
        self.OLSresult = pa.concat([dft1,dft2.transpose()])

    def OLS_add_pvalue_R2(self,pos='ul',pvalue=True,R2=True,
                          regdf='OLSresult',surfix='',
                          ftdic={'size':'12'},**kwargs):
        """
        Add the R2 and pvalue to the plot by using the self.lax.add_label
        method. It will try to use the Pdata generated regression dataframe,
        or could be forced by an external dataframe.

        Parameters:
        -----------
        regdf: in case of string, it's expected to be an attribute of the
            Pdata.Pdata object, otherwise it should be provided as a
            pandas dataframe, with the "R2" and 'p_value' in the dataframe
            index, and the Pdata.Pdata tags in the dataframe column.

        pvalue,R2: boolean variable.
        surfix: string type, used to be surfixed in the text.
        pos: string or tuple type, used in the g.Set_AxText method.
        """
        #we try to find the dataframe for the regression result.
        if isinstance(regdf,str):
            regdf = self.__getattribute__(regdf)
        elif isinstance(regdf,pa.DataFrame):
            pass
        else:
            raise TypeError("regdf could only be string or dataframe")

        #we try to build the dict for the R2.
        if R2 == True:
            if 'R2' not in regdf.index:
                raise ValueError("R2 not in the regdf index!")
            else:
                Rsquare_dic = regdf.ix['R2'].to_dict()
            Rsquare_anno_dic = {}
            for key,val in Rsquare_dic.items():
                Rsquare_anno_dic[key] = 'R2='+str(round(val,2))+surfix
        else:
            pass

        #we try to build the dict for the p-value.
        if pvalue == True:
            if 'p_value' not in regdf.index:
                raise ValueError("p_value not in the regdf index!")
            else:
                pvalue_dic = regdf.ix['p_value'].to_dict()
            pvalue_anno_dic = {}
            for key,val in pvalue_dic.items():
                pvalue_anno_dic[key] = 'p='+str(round(val,3))+surfix
        else:
            pass

        def get_combine_anno(Rsquare_anno_dic,pvalue_anno_dic):
           combine_anno_dic = {}
           for key in Rsquare_anno_dic.keys():
               combine_anno_dic[key] = pvalue_anno_dic[key] + ', '+\
                                       Rsquare_anno_dic[key]
           return combine_anno_dic

        if pvalue == True and R2 == True:
            combine_anno_dic = get_combine_anno(Rsquare_anno_dic,
                                                pvalue_anno_dic)
            self.lax.add_label(label=combine_anno_dic,pos=pos,
                               ftdic=ftdic,**kwargs)
        elif pvalue == True:
            self.lax.add_label(label=pvalue_anno_dic,pos=pos,
                               ftdic=ftdic,**kwargs)
        elif R2 == True:
            self.lax.add_label(label=Rsquare_anno_dic,pos=pos,
                               ftdic=ftdic,**kwargs)
        else:
            raise ValueError("Both pvalue and R2 are False!")

    def OLS_add_info(self,keyword,regdf='OLSresult',
                     pos='ul',prefix='',surfix='',
                     show_keyword=True,
                     ftdic={'size':'12'},**kwargs):
        """
        Add other information form the regression to the plot by using
        the self.lax.add_label method. It will try to use the Pdata
        generated regression dataframe, or could be forced by an
        external dataframe, with the keyword in the index of the dataframe.

        Parameters:
        -----------
        regdf: in case of string, it's expected to be an attribute of the
            Pdata.Pdata object, otherwise it should be provided as a
            pandas dataframe, with the keyword(str type) in the dataframe
            index, and the Pdata.Pdata tags in the dataframe column.

        keyword: string type, used to indicate what information is required.
        prefix/surfix: string type, used to be pre/surfixed in the text.
        pos: string or tuple type, used in the g.Set_AxText method.
        """
        #we try to find the dataframe for the regression result.
        if isinstance(regdf,str):
            regdf = self.__getattribute__(regdf)
        elif isinstance(regdf,pa.DataFrame):
            pass
        else:
            raise TypeError("regdf could only be string or dataframe")

        #we try to retrieve the information for the keyword.
        if isinstance(keyword,str):
            if keyword not in regdf.index:
                raise ValueError("{0} not in the regdf index".format(keyword))
            else:
                keyword_dic = regdf.ix[keyword].to_dict()
            keyword_anno_dic = {}
            for key,val in keyword_dic.items():
                if show_keyword:
                    keyword_anno_dic[key] = prefix+keyword+'='+str(round(val,3))+surfix
                else:
                    keyword_anno_dic[key] = prefix+str(round(val,3))+surfix
        else:
            raise ValueError("keyword must be a string")
        self.lax.add_label(label=keyword_anno_dic,pos=pos,
                               ftdic=ftdic,**kwargs)

    def add_regression_line(self,reg_df,color='k',ls='--',**kwargs):
        """
        Add regression line by specifying slope and intercept.

        Parameters:
        -----------
        1. reg_df: the regression result dataframe, which must have
            ['Intercept','Slope'] in its column names.
        """
        if not isinstance(reg_df,pa.DataFrame):
            raise TypeError("reg_df must be pandas DataFrame type!")
        else:
            if 'Intercept' not in reg_df.columns:
                raise ValueError("Intercept not in the columns of dataframe")
            elif 'Slope' not in reg_df.columns:
                raise ValueError("Slope not in the columns of dataframe")
            else:
                pass

        taglist = reg_df.index.tolist()
        self.regression_linedic = {}
        for tag in taglist:
            if tag not in self.taglist:
                raise ValueError("{0} not in the taglist")
            else:
                xdata = self.data[tag]['x']
                ax = self.axdic[tag]
                xdata_plot = pb.linspace_array(xdata)
                slope = reg_df.ix[tag]['Slope']
                intercept = reg_df.ix[tag]['Intercept']
                ydata_plot = xdata_plot * slope + intercept
                line = ax.plot(xdata_plot,ydata_plot,color=color,ls=ls,**kwargs)
                self.regression_linedic[tag] = line

    def plot_split_axes_byX(self,force_axs=None,sharex=True,tagpos='ul',unit=False,ylab=False,force_taglist=None,ncols=1,plotkw={},**fig_kw):
        """
        line plot with each tag in a separate axes, data with different tags have the same xdata.
        Now only unit is used, ylab is not used.
        Arguments:
            force_taglist: could be used to plot for only selected tags, or to force the sequence of tags.
        """
        print "Warning! This is deprecated! use plot_split_axes instead."
        if force_taglist is None:
            tag_list=self._taglist
        else:
            tag_list=force_taglist
        num=len(tag_list)
        axdic={}
        if force_axs is None:
            if ncols == 1:
                fig,axs=plt.subplots(nrows=num, ncols=1, sharex=sharex, sharey=False, squeeze=True, subplot_kw=None, **fig_kw)
            else:
                nrows=num/ncols+1
                fig,axt=plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=False, squeeze=True, subplot_kw=None, **fig_kw)
                axs=axt.flatten()[0:num]
        else:
            if num<=len(force_axs):
                axs=force_axs[0:num]
        for tag,axt in zip(tag_list,axs):
            pd_temp=self.regroup_data_by_tag([tag])
            pd_temp.plot(axt,**plotkw)
            axdic[tag]=axt
            g.Set_AxText(axt,tag,tagpos)
            if unit==True:
                if isinstance(unit,str):
                    print "forced unit is used"
                    axt.set_ylabel(unit)
                else:
                    try:
                        axt.set_ylabel(pd_temp.data[tag]['unit'])
                    except AttributeError:
                        pass
        if force_axs is None:
            return fig,axdic
        else:
            return axdic



    def to_dic(self,sharex=False):
        outdic={}
        if sharex==True:
            for tag,tag_dic in self.data.items():
                outdic[tag]=tag_dic['y']
        else:
            raise ValueError("This can only be use for sharex=True, please specify this explicitly or check data length for each tag.")
        return outdic


    def to_DataFrame(self,tagindex=False,col_name=None,df_index=None):
        """
        Change the data to DataFrame

        Parameters:
        ----------
        tagindex : True if the tags are used as index of the new DataFrame, in this case the col_name must be provided. False if the tags are used as DataFrame column names.
        """
        outdic=self.to_dic(sharex=True)
        if tagindex == False:
            if df_index is None:
                index=self.data.values()[0]['x'] #as sharex is True, so we can pick x data for any tag and use it as index of the DataFrame.
            else:
                index=df_index
            df = pa.DataFrame(outdic,index=index) #as sharex is True, so we can pick x data for any tag and use it as index of the DataFrame.

        else:
            if col_name is None:
                raise ValueError("In case of using tag as DataFrame index, new columns names must be provided! col_name could not be None!")
            else:
                df=tools.DataFrame_from_Dic_Key_as_Index(outdic, columns=col_name)
        return df

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
            self.data[newtag] = self.data[oldtag]
            if oldtag != newtag:
                del self.data[oldtag]
            self._taglist[self._taglist.index(oldtag)] = newtag

    def to_pd(self,filename):
        """
        pickle the self.data to filename; the surfix '.pd' could be used as indication as pdata
        """
        pb.pfdump(self.data,filename)


    @classmethod
    def merge(cls,*pdlist):
        """
        Merge Pdata.Pdata objects into a single Pdata.Pdata
        """
        newdata = OrderedDict()
        for pd in pdlist:
            newdata.update(pd.data)
        return Pdata(data=newdata)

    def duplicate_tag(self,oldtag,newtag):
        self.add_tag(tag=newtag)
        self.data[newtag] = self.data[oldtag].copy()

    def drop_tag(self,tag):
        del self.data[tag]
        self._taglist.remove(tag)

def pmerge(*pdlist):
    print "Deprecating Warning! pmerge"
    datadic = {}
    for pd in pdlist:
        datadic.update(pd.data)
    pdnew=Pdata(data=datadic)
    return pdnew

def open(pdfilename):
    """
    create a pdata from "*.pd" file
    """
    print "DeprecatingWarning! open"
    pdata=Pdata()
    data=pb.pfload(pdfilename)
    pdata = Pdata(data=data)
    return pdata

def dic_pdata_to_pd(pd_dic,filename):
    pd_data_dic={}
    for kw,pd in pd_dic.items():
        pd_data_dic[kw]=pd.data
    pb.pfdump(pd_data_dic,filename)

def dic_pdata_from_pd(pdfilename):
    pd_data_dic=pb.pfload(pdfilename)
    pd_dic={}
    for kw,pd_data in pd_data_dic.items():
        pd=Pdata()
        pd.data=pd_data
        pd_dic[kw]=pd
    return pd_dic


def plot_dic_as_pdata_sharex_noerror(indic,force_axs=None,sharex=True,tagpos='ul',unit=False,ylab=False,force_taglist=None,ncols=1,plotkw={},**fig_kw):
    pd = Pdata()
    pd.add_entry_sharex_noerror_by_dic(indic)
    pd.plot_split_axes_byX(force_axs=force_axs,sharex=sharex,tagpos=tagpos,unit=unit,ylab=ylab,force_taglist=force_taglist,ncols=ncols,plotkw={},**fig_kw)
    return pd

def read_csv(filepath_or_buffer,sep=',',header=0,index_col=None,
             index_func=None,names=None,date_parser=None,
             skiprows=None,na_values=None,df_func=None,
             force_sharex=None,**kwds):
    """
    A wrapper of pa.read_csv and Pdata.from_DataFrame,
        to read csv directly to Pdata.Pdata

    Parameters:
    -----------
    force_sharex: use force_sharex to overwrite the default sharex of
        Pdata that are the index column of the csv/DataFrame object.

    Notes
    -----
    1. the index_col (index of the pandas DataFrame) will be treated as
        shared xaxis of the Pdata.Pdata object
    2. the (column) names of the pandas.DataFrame object will be used
        as tags in the Pdata.Pdata object.

    See also
    --------
    from_DataFrame
    """
    df = pa.read_csv(filepath_or_buffer, sep=sep, dialect=None,
                     header=header, index_col=index_col, names=names,
                     skiprows=skiprows, na_values=na_values,
                     keep_default_na=True,
                     thousands=None, comment=None, parse_dates=False,
                     keep_date_col=False, dayfirst=False,
                     date_parser=date_parser, nrows=None, iterator=False,
                     chunksize=None, skip_footer=0, converters=None,
                     verbose=False, delimiter=None,
                     encoding=None, squeeze=False, **kwds)

    return from_DataFrame(df,df_func=df_func,index_func=index_func,
                          force_sharex=force_sharex)

def from_DataFrame(df,df_func=None,index_func=None,force_sharex=None):
    """
    Create a sharex Pdata.Pdata object from pandas DataFrame

    Parameters:
    -----------
    df_func: function that applies on DataFrame before feeding data into Pdata.
    index_func: index function that will be applied before using the
        DataFrame index as shared xaxis of the Pdata object, this is
        useful as sometimes DataFrame index could be a bit strange
        and not readily compatible with matplotlib functions.
    force_sharex: In case index_func could not achieve the object to
        transform the index to desired sharex xaxis, force_sharex
        is used to force write the Pdata shared xaxis.

    Notes:
    ------
    1. the column sequence of pa.DataFrame will be retained as taglist.
    """
    print """DeprecatingWarning Pdata.from_DataFrame! use 
             Pdata.Pdata.from_dataframe instead"""
    pd = Pdata()
    if df_func is not None:
        df = df_func(df)
    if force_sharex is None:
        if index_func is None:
            pd.add_entry_sharex_noerror_by_dic(df,x=df.index.values)
        else:
            pd.add_entry_sharex_noerror_by_dic(df,x=index_func(df.index.values))
    else:
        pd.add_entry_sharex_noerror_by_dic(df,x=force_sharex)
    pd.set_tag_order(list(df.columns))
    return pd



class NestPdata(object):
    """
    NestPdata will receive a dictionary of Pdata object.
    """
    def __init__(self,dic_pdata):
        self.child_pdata = dic_pdata
        self.parent_tags = dic_pdata.keys()
        self.child_tags = self.child_pdata.values()[0].list_tags()


    def __repr__(self):
        return """parent tags: {0}""".format(self.parent_tags) + '\n' +\
        "child tags: {0}".format(self.child_tags)

    @property
    def data(self):
        nestpdata_datadic = {}
        for parent_tag in self.parent_tags:
            pdtemp = self.child_pdata[parent_tag]
            nestpdata_datadic[parent_tag] = copy.deepcopy(pdtemp.data)
        return nestpdata_datadic

    def iteritems(self):
        for ptag in self.parent_tags:
            yield ptag,self.child_pdata[ptag]

    def data_ix(self,ls):
        """
        Use a list to retrieve the final data.

        ls: ['ptag','ctag','x/y/...']
        """
        return self.data[ls[0]][ls[1]][ls[2]]

    def set_new_tags(self,old_new_tag_tuple_list,mode='parent'):
        """
        Change the old tag to new tag according to old_new_tag_tuple_list

        Parameters:
        -----------
        old_new_tag_tuple_list: [(oldtag1,newtag1),(oldtag2,newtag2)]

        Notes:
        ------
        1. In-place operation.
        """
        if mode == 'parent':
            for (oldtag,newtag) in old_new_tag_tuple_list:
                self.child_pdata[newtag] = self.child_pdata[oldtag]
                del self.child_pdata[oldtag]
                self.parent_tags[self.parent_tags.index(oldtag)] = newtag
        elif mode == 'child':
            for pd in self.child_pdata.values():
                pd.set_new_tags(old_new_tag_tuple_list)
        else:
            raise ValueError("mode not understood.")

    def set_parent_tag_order(self,taglist):
        if sorted(self.parent_tags) != sorted(taglist):
            raise ValueError("new parent_tags not equal to old one")
        else:
            self.parent_tags = taglist[:]

    def set_child_tag_order(self,taglist):
        for ptag in self.parent_tags:
            self.child_pdata[ptag].set_tag_order(taglist)
        self.child_tags = self.child_pdata.values()[0].list_tags()

    def permuate_tag(self):
        '''
        Notes:
        ------
        1. the sequence of parent_tags and child_tags are reserved.
        '''
        ptags = copy.deepcopy(self.parent_tags)
        ctags = copy.deepcopy(self.child_tags)

        data = pb.Dic_Nested_Permuate_Key(self.data)
        self.parent_tags = ctags
        self.child_pdata = {}
        for parent_tag in self.parent_tags:
            pdata_temp = Pdata(data[parent_tag])
            self.child_pdata[parent_tag] = pdata_temp
        self.child_tags = ptags



    def apply(self,func=None,axis=None,taglist='all'):
        """
        Apply a function either 'x' or 'y' axis or 'both' or 'diff', if
            axis=='diff', func should be supplied with a dictionary by
            using ('x'/'y',x_func/y_func) pairs.

        Returns:
        --------
        npd object
        """
        dic = OrderedDict()
        for ptag,pd in self.iteritems():
            dic[ptag] = pd.apply_function(func=func,
                taglist=taglist,axis=axis,copy=True)
        return NestPdata(dic)


    def pool_data_by_tag(self,mode='child',tagkw=False,**group_dic):
        """
        Pool the data together by specifying new_tag=[old_tag_list] pairs;
            when tagkw==True, it's allowed to use new_tag=old_tag_keyword to
            specify what old tags will be pooled together which will
            include old_tag_keyword. Other attributes outside the
            _data_base_keylist will be copied from the first old_tag of
            the old_tag_list.
        Example:
            pool_data_by_tag(alldry=['1stdry','2nddry','3rddry'],
                allwet=['wet1','wet2'])
            pool_data_by_tag(tagkw=True,alldry='dry')
        """
        dic = OrderedDict()
        if mode == 'child':
            for ptag,cpd in self.iteritems():
                dic[ptag] = cpd.pool_data_by_tag(tagkw=tagkw,**group_dic)
            return NestPdata(dic)
        elif mode == 'parent':
            npdt = self.copy()
            npdt.permuate_tag()
            npdt2= npdt.pool_data_by_tag(self,mode='child',tagkw=tagkw,**group_dic)
            npdt2.permuate_tag()
            return npdt2



        if tagkw==True:
            group_dic_final={}
            tags=self._taglist
            for newtag,newtag_tagkw in group_dic.items():
                group_dic_final[newtag]=FilterStringList(newtag_tagkw,tags)
        else:
            group_dic_final=group_dic

        pdata=Pdata()
        for newtag in group_dic_final.keys():
            old_tag_list=group_dic_final[newtag]
            old_entry_list=[self._fill_errNone_with_Nan(tag) for
                                tag in old_tag_list]
            new_entry=Pdata.Hstack_Dic_By_Key(old_entry_list,
                                              Pdata._data_base_keylist)
            #for _extra_base_keylist attributes, their default value in
            #_new_entry will be supplied to the new tag (pooled) data.
            #all other ploting attributes in _plot_attr_dic.values(),
            #they're lost.
            first_old_tag=old_tag_list[0]
            new_entry.update(Dic_Remove_By_Subkeylist(self.data[first_old_tag],
                                                      Pdata._data_base_keylist))
            pdata.add_entry_by_dic(**{newtag:new_entry})
        return pdata


    def plot_split_parent_tag(self,plotkw={},legtag=None,
                              legtagseq=None,legkw={},
                              **kwargs):
        """
        Plot by using parent tags as label for subplots.

        Parameters:
        -----------
        legtag: the child tag for which legend will be shown, False to supress.
            'last' to use the empty unused subplot.
        legtagseq: the child tag sequence for the legtag.
        legkw: used in plt.legend for the legtag
        plotkw: the keyword used in plt.plot function.
        kwargs:
            force_axs: force the axes.
            tagseq: the sequence for parent tags.
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            plotkw: the keyword used in plt.plot function.
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        """
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                            default_tagseq=self.parent_tags,
                            default_tagpos=(0.02,0.83),
                            **kwargs)
        for tag,axt in axdic.items():
            pd_temp = self.child_pdata[tag]
            pd_temp.plot(ax=axt,legend=False,**plotkw)

        if legtag == False:
            pass
        elif legtag == 'last':
            axt = axdic.values()[0]
            legax = axt.figure.axes[-1]
            self.child_pdata[self.parent_tags[0]].set_legend(plottype='line',taglab=True,
                                                tag_seq=legtagseq,axes=legax,
                                                **legkw)
        elif isinstance(legtag,mat.axes.Axes):
            legax = legtag
            self.child_pdata[self.parent_tags[0]].set_legend(plottype='line',taglab=True,
                                                tag_seq=legtagseq,axes=legax,
                                                **legkw)

        else:
            legtag = _replace_none_by_given(legtag,self.parent_tags[0])
            legax = self.child_pdata[legtag]
            self.child_pdata[legtag].set_legend(plottype='line',taglab=True,
                                                tag_seq=legtagseq,
                                                **legkw)
        self.axes = axdic
        self.axdic = axdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def bar_split_parent_tag(self,barkw={},legtag=None,
                              legtagseq=None,legkw={},
                              **kwargs):
        """
        Plot by using parent tags as label for subplots.

        Parameters:
        -----------
        legtag: the child tag for which legend will be shown, False to supress.
        legtagseq: the child tag sequence for the legtag.
        legkw: used in plt.legend for the legtag
        barkw: the keyword used in plt.plot function.
        kwargs:
            force_axs: force the axes.
            tagseq: the sequence for parent tags.
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            barkw: the keyword used in plt.plot function.
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        """
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                            default_tagseq=self.parent_tags,
                            default_tagpos=(0.02,0.83),
                            **kwargs)
        for tag,axt in axdic.items():
            pd_temp = self.child_pdata[tag]
            pd_temp.bar(axes=axt,legend=False,**barkw)

        if legtag == False:
            pass
        else:
            legtag = _replace_none_by_given(legtag,self.parent_tags[0])
            self.child_pdata[legtag].set_legend(plottype='bar',taglab=True,
                                                tag_seq=legtagseq,
                                                **legkw)
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def plot_stackline_split_parent_tag(self,
                        fillkw={}, legdic={},
                        stackw={}, legtag=None,
                        **kwargs):
        """
        Plot for each parent tag a stackline plot in a subplot

        Parameters:
        -----------
        fillkw: kwargs in plt.fill_between function
        legdic: kwargs in plt.legend function
        stackw: parameters in Pdata.Pdata.plot_stackline function,
            including: tagseq(child tags),colors,bottom_fill,
                       legend(boolean)
        kwargs:
            force_axs: force the axes.
            tagseq: the sequence for parent tags.
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            subkw: kwarg in plt.subplots function
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        """
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                            default_tagseq=self.parent_tags, **kwargs)
        for tag,axt in axdic.items():
            pd_temp = self.child_pdata[tag]
            pd_temp.plot_stackline(axes=axt, fillkw=fillkw,
                                   legdic=legdic,
                                   legend=False,
                                   **stackw)

        if legtag is None:
            pass
        else:
            legtag = _replace_none_by_given(legtag,self.parent_tags[0])
            self.child_pdata[legtag].set_legend(plottype='stackline',
                                                taglab=True,
                                                tag_seq=legtagseq,
                                                **legkw)

        self.axdic = axdic
        self.axes = axdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def set_xdata_by_ydata_by_tag(self,child_tag):
        """
        For each child_pdata, call the Pdata.Pdata.set_xdata_by_ydata_by_tag
            to fill the xdata by the ydata of a speficified child_tag.
        """
        dic = OrderedDict()
        for ptag,pd in self.iteritems():
            dic[ptag] = pd.set_xdata_by_ydata_by_tag(child_tag)
        return NestPdata(dic)


    def get_proleg(self,childtag=None,plottype='all',taglab=True,tag_seq=None):
        '''
        Return the proxy legend for the NestPdata after plotting.

        Notes:
        ------
        1. If childtag is None, the proxy legend for an arbitrary one of
            the child Pdata will be returned, otherwise the one for the
            specified child Pdata will be returned.
        2. for other parameters, cf. Pdata.get_proleg
        '''
        if childtag is None:
            pd = self.child_pdata.values()[0]
        else:
            pd = self.child_pdata[childtag]
        return pd.get_proleg(plottype=plottype, taglab=taglab, tag_seq=tag_seq)


    def copy(self):
        original_parent_tags = self.parent_tags
        original_child_tags = self.child_tags
        nestpd_dic = {}
        for tag,child_pd in self.child_pdata.items():
            nestpd_dic[tag] = child_pd.copy()
        nestpd = NestPdata(nestpd_dic)
        nestpd.set_parent_tag_order(original_parent_tags)
        nestpd.set_child_tag_order(original_child_tags)
        return nestpd

    def _regroup_parent_tag(self,taglist):
        nestpd_dic = Dic_Extract_By_Subkeylist(self.child_pdata,taglist)
        if nestpd_dic == {}:
            raise AttributeError("the taglist does not include any original tag")
        else:
            return NestPdata(nestpd_dic)

    def regroup_data_by_tag(self,taglist,keyword=True,mode='parent'):
        if mode == 'parent':
            npd = self._regroup_parent_tag(taglist)
            npd.set_parent_tag_order(taglist)
            return npd
        elif mode == 'child':
            nestpd_dic_pre = self.copy()
            original_parent_tags = nestpd_dic_pre.parent_tags
            nestpd_dic_pre.permuate_tag()  #change the child to parent
            nestpd = nestpd_dic_pre._regroup_parent_tag(taglist)
            nestpd.permuate_tag()  #change back
            nestpd.set_parent_tag_order(original_parent_tags)
            nestpd.set_child_tag_order(taglist)
            return nestpd
        else:
            raise ValueError("unknown mode.")


    #def collapse_to_Pdata(self,mode='parent'):
        #if mode = 'parent'

    def __getitem__(self,taglist):
        """
        simple implementation of Pdata.NestPdata.regroup_data_by_tag, with
            slicing mode in parent tags.
        """
        if isinstance(taglist,str):
            taglist = [taglist]
        npd = self.regroup_data_by_tag(taglist,mode='parent')
        npd.set_parent_tag_order(taglist)
        return npd

    def child_xs(self,taglist):
        """
        simple implementation of Pdata.NestPdata.regroup_data_by_tag, with
            slicing mode in child tags.
        """
        if isinstance(taglist,str):
            taglist = [taglist]
        npd = self.regroup_data_by_tag(taglist,mode='child')
        npd.set_child_tag_order(taglist)
        return npd

    @classmethod
    def merge_npd(cls,*npdlist,**kwargs):
        '''
        Merge multiple NestPdata by either 'parent' or 'child'

        Parameters:
        -----------
        kwargs: mode='parent' or 'child'
        '''
        mode = kwargs.get('mode','parent')
        child_tags = npdlist[0].child_tags
        npddic = {}
        if mode == 'parent':
            ptaglist = []
            for npd in npdlist:
                if sorted(child_tags) != sorted(npd.child_tags):
                    raise ValueError('''child_tags not equal''')
                else:
                    npddic.update(npd.child_pdata)
                    ptaglist.extend(npd.parent_tags)
            npd_merge = NestPdata(npddic)
            npd_merge.set_parent_tag_order(ptaglist)
        elif mode == 'child':
            new_npdlist = []
            for npd in npdlist:
                npd.permuate_tag()
                new_npdlist.append(npd)
            npd_merge = NestPdata.merge_npd(*new_npdlist,mode='parent')
            npd_merge.permuate_tag()
        return npd_merge

    def setp_tag(self,plottype='all',tagkw=False,**nested_attr_tag_value_dic):
        '''
        Example: npd.setp_tag('line',lw=dict(wrong=1),alpha=dict(wrong=0.6),
                               color=dict(right='m'),zorder=dict(right=4))
        right and wrong are child_tags.
        '''
        for pdtemp in self.child_pdata.values():
            pdtemp.setp_tag(plottype, tagkw=tagkw,
                           **nested_attr_tag_value_dic)

    def set_legend_all(self,legtag=None,legtagseq=None,**kwargs):
        '''
        Set legend using one of the child_pdata
        '''
        legtag = _replace_none_by_given(legtag,self.parent_tags[0])
        self.child_pdata[legtag].set_legend_all(taglab=True,
                                            tag_seq=legtagseq,**kwargs)

    def to_Panel(self):
        dic = {}
        for subptag,pd in self.child_pdata.items():
            dic[subptag] = pd.to_DataFrame()
        return pa.Panel(dic)

    def to_npd(self,filename):
        """
        File name must end with ".npd"
        """
        if not filename.endswith('.npd'):
            raise ValueError('''filename not end with .npd''')
        else:
            pb.pfdump(self.data,filename)

    @classmethod
    def open_npd(cls,filename):
        data = pb.pfload(filename)
        dic = {}
        for subptag,subpdic in data.items():
            dic[subptag] = Pdata(data=subpdic)
        return NestPdata(dic)

    @classmethod
    def from_dict_of_dataframe(cls,dict_dataframe,
                               df_func=None,
                               index_func=None,
                               force_sharex=None):
        """
        Create a NestPdata object from a dictionary of dataframe or pandas
            panel data.

        Parameters:
        -----------
        dict_dataframe: could be a dict of dataframe or pandas panel data.

        Notes:
        ------
        1. In case that input is pandas panel object, the items of the panel
            will be the parent tags of the NestPdata; the minor axis will be
            the child tages of the NestPdata.
        """
        if isinstance(dict_dataframe,pa.Panel):
            dic = dict(dict_dataframe.iteritems())
        elif isinstance(dict_dataframe,dict):
            dic = dict_dataframe
        else:
            raise TypeError

        pddic = {}
        for parent_tag,dataframe in dic.items():
            pddic[parent_tag] = Pdata.from_dataframe(dataframe,
                                    df_func=df_func,
                                    index_func=index_func,
                                    force_sharex=force_sharex)
        return NestPdata(pddic)

    def add_attr_by_tag(self,tagkw=False,**nested_attr_tag_value_dic):
        """
        The Pdata.add_attr_by_tag applied on each child pdata.
        """
        for subptag,child_pd in self.child_pdata.items():
            child_pd.add_attr_by_tag(tagkw=tagkw,**nested_attr_tag_value_dic)

class Pdata3D(object):
    """
    Initialize with a dictionary of NestPdata
    """
    def __init__(self,dic_npd):
        if not isinstance(dic_npd,OrderedDict):
            raise TypeError("not a OrderedDict")
        else:
            self.child_npd = dic_npd
            cnpd1 = self.child_npd.values()[0]
            self.labels = dic_npd.keys()
            self.parent_tags = cnpd1.parent_tags
            self.child_tags = cnpd1.child_tags

    def __getitem__(self,key):
        if isinstance(key,str):
            return self.child_npd[key]
        elif isinstance(key,list):
            dic = OrderedDict()
            npds = [self.child_npd[label] for label in key]
            return Pdata3D(OrderedDict(zip(key,npds)))

    def iteritems(self):
        for label in self.labels:
            yield label,self.child_npd[label]

    @classmethod
    def from_Panel4D(cls,p4d):
        npdlist = []
        for label in p4d.labels:
            npd = NestPdata.from_dict_of_dataframe(p4d[label])
            npdlist.append(npd)
        dic = OrderedDict(zip(p4d.labels,npdlist))
        return Pdata3D(dic)

    def to_Panel4D(self):
        dic = {}
        for label,npd in self.iteritems():
            dic[label] = npd.to_Panel()
        return pa.Panel4D(dic)

    def __repr__(self):
        return "labels: {0}".format(self.labels)+ '\n' +\
        """parent tags: {0}""".format(self.parent_tags) + '\n' +\
        "child tags: {0}".format(self.child_tags)


    def plot_icecore(self,legkw={},add_label=True,legtag=False,plotkw={},**kwargs):
        """
        Parameters:
        -----------
        legtag: the child tag for which legend will be shown, False to supress.
        legtagseq: the child tag sequence for the legtag.
        legkw: used in plt.legend for the legtag
        plotkw: the keyword used in plt.plot function.
        kwargs:
            force_axs: force the axes.
            tagseq: the sequence for parent tags.
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            plotkw: the keyword used in plt.plot function.
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        """
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                            default_tagseq=self.parent_tags,
                            default_tagpos=(0.02,0.83),
                            **kwargs)

        lax = LabelAxes.LabelAxes(tags=axdic.keys(),axl=axdic.values())
        dic = lax.build_icecore(num=len(self.labels),keys=self.labels[::-1])
        lax2D = LabelAxes.LabelAxes2D(dic)
        for label,npd in self.iteritems():
            npd.plot_split_parent_tag(plotkw=plotkw,legtag=legtag,tagpos=False,
                              force_axdic=lax2D[label].data)

        self.lax2D = lax2D
        if add_label == True:
            lax2D.add_parent_label(pos=(0.02,0.8))
            lax2D.add_child_label()


    def get_proleg2D(self,childtag=None,plottype='all',taglab=True,tag_seq=None):
        plegs = []
        for label,npd in self.iteritems():
            plegs.append(npd.get_proleg(childtag=childtag,plottype=plottype,
                                        taglab=taglab,tag_seq=tag_seq))
        return g.ProxyLegend2D(OrderedDict(zip(self.labels,plegs)))


    def setp_tag(self,label=None,plottype='all',tagkw=False,**nested_attr_tag_value_dic):
        """
        """
        self.child_npd[label].setp_tag(plottype='all',tagkw=tagkw,**nested_attr_tag_value_dic)

    def set_label_order(self,labseq=None):
        """
        Set tag order and this order will be kept throughout all the class
        method when default taglist is used.
        """
        if sorted(self.labels) == sorted(labseq):
            self.labels = labseq[:]
        else:
            raise ValueError('ordered labels not equal to present one')

    def set_parent_tag_order(self,taglist):
        for ptag,npd in self.iteritems():
            npd.set_parent_tag_order(taglist)
        self.parent_tags = taglist[:]

    def set_child_tag_order(self,taglist):
        for ptag,npd in self.iteritems():
            npd.set_child_tag_order(taglist)
        self.child_tags = taglist[:]

class Mdata(Pdata):
    _data_base_keylist=['array','lat','lon']
    _new_entry=dict.fromkeys(_data_base_keylist,None)

    def __init__(self,data=None):
        if data is None:
            self.data = {}
        else:
            self.data = copy.deepcopy(data)

        if self.data == {}:
            self._taglist = []
        else:
            self._taglist = self.data.keys()
            self._data_complete_check_all()

    def add_tag(self,tag=None):
        self.data[tag]=Mdata._new_entry.copy()
        self._taglist.append(tag)

    def add_array(self,data,tag):
        self.data[tag]['array'] = data

    def add_lat_lon(self,tag=None,lat=None,lon=None):
        '''
        default lat,lon is half degree global coordinates
        '''
        if lat is None:
            lat = np.arange(89.5,-89.6,-1)
        if lon is None:
            lon = np.arange(-179.5,179.6,1)
        self.data[tag]['lat'] = lat
        self.data[tag]['lon'] = lon

    def add_array_lat_lon(self,tag=None,data=None,lat=None,lon=None):
        self.add_tag(tag)
        self.add_array(data,tag)
        self.add_lat_lon(tag=tag,lat=lat,lon=lon)

    def add_entry_array_bydic(self,ydic):
        for tag,ydata in ydic.items():
            self.add_tag(tag)
            self.add_array(ydata,tag)

    @classmethod
    def from_dict_of_array(cls,ydic,npindex=None,lat=None,lon=None):
        if npindex is None:
            ydicnew = ydic
        else:
            ydicnew = {}
            for k,v in ydic.items():
                ydicnew[k] = v[npindex]

        md = Mdata()
        md.add_entry_array_bydic(ydicnew)
        md.add_attr_by_tag(lat=lat,lon=lon)
        return md

    @classmethod
    def from_panel(cls,panel,lat=None,lon=None):
        ydic = OrderedDict(panel)
        ydicnew = pb.Dic_Apply_Func(lambda df:df.values,ydic)

        md = Mdata()
        md.add_entry_array_bydic(ydicnew)
        md.add_attr_by_tag(lat=lat,lon=lon)
        return md

    @classmethod
    def from_ndarray(cls,arr,tagaxis=0,taglist=None,lat=None,lon=None):
        if np.ndim(arr) != 3:
            raise ValueError('''array ndim is {0}, only 3 is valid'''.
                                format(arr.ndim))
        if tagaxis == 0:
            datalist = [arr[i] for i in range(arr.shape[0])]
        elif tagaxis == 1:
            datalist = [arr[:,i,:] for i in range(arr.shape[1])]
        elif tagaxis == 2:
            datalist = [arr[...,i] for i in range(arr.shape[2])]
        else:
            raise ValueError("unknown tagaxis!")
        taglist = _replace_none_by_given(taglist,['tag'+str(s) for s in np.arange(1,len(datalist)+1)])
        return Mdata.from_dict_of_array(OrderedDict(zip(taglist,datalist)),
                                        lat=lat,lon=lon)
    def __getitem__(self,key):
        if isinstance(key,str):
            return self.data[key]['array']
        elif isinstance(key,list):
            mdt = self.regroup_data_by_tag(key)
            mdt2 = Mdata(data=mdt.data)
            mdt2._taglist = key
            return mdt2

    def __setitem__(self,key,value):
        tag1 = self.taglist[0]
        self.add_tag(key)
        self.add_array(value,key)
        self.add_lat_lon(tag=key,lat=self.data[tag1]['lat'],lon=self.data[tag1]['lon'])

    def add_entry_share_latlon_bydic(self,ydic,lat=None,lon=None):
        for tag,ydata in ydic.items():
            self.add_tag(tag)
            self.add_array(ydata,tag)
            self.add_lat_lon(tag=tag,lat=lat,lon=lon)

    def imshow_split_axes(self, cmap=None, norm=None, aspect=None,
                          interpolation=None, alpha=None, vmin=None,
                          vmax=None, origin=None, extent=None,
                          imkw={}, cbar=False, cbarkw = {},
                          **kwargs):
        '''
        imshow each tag on a subplot.

        Parameters:
        -----------
        imkw: the keyword used in plt.imshow function.
        kwargs:
            force_axs: force the axes.
            tagseq: the tag sequence, default is self._taglist
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        '''
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                        default_tagseq=self._taglist,
                        default_tagpos='ouc',
                        **kwargs)
        imgdic={}
        for tag,axt in axdic.items():
            img = axt.imshow(self.data[tag]['array'], cmap=cmap, norm=norm,
                       aspect=aspect,
                       interpolation=interpolation, alpha=alpha,
                       vmin=vmin, vmax=vmax, origin=origin,
                       extent=extent, **imkw)
            imgdic[tag] = img
        self.axdic = axdic
        self.imgdic = imgdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

        if cbar == True:
            self.colorbar('img',**cbarkw)

    def colorbar(self,plottype='img',**kw):
        """
        Add colorbar.

        Parameters:
        -----------
        1. plottype: now accept only "img"
        """
        if plottype == 'img':
            mappable_dict = self.imgdic
        else:
            raise ValueError("plottype not known!")

        cbardic = {}
        for tag,mappable in mappable_dict.items():
            cbar = plt.colorbar(mappable,ax=mappable.axes,**kw)
            cbardic[tag] = cbar
        self.cbardic = cbardic

    def mapimshow_split_axes(self,projection='cyl',mapbound='all',
                             gridstep=None,shift=False,
                             map_threshold=None,
                             cmap=None,colorbarlabel=None,forcelabel=None,
                             levels=None,data_transform=False,
                             colorbardic={},cbarkw={},imgkw={},
                             *args,
                             **kwargs):
        '''
        bmap.mapimshow each tag on a subplot.

        Parameters:
        -----------
        imshowkw: bmap.mapcontourf kwargs.
        c.f. bmap.mapcontourf
        kwargs:
            force_axs: force the axes.
            tagseq: the tag sequence, default is self._taglist
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        '''
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                        default_tagseq=self._taglist,
                        default_tagpos='ouc',
                        sharex=True, sharey=True,
                        **kwargs)
        mapimgdic={}
        for tag,axt in axdic.items():
            mapimg = bmap.mapimshow(data=self.data[tag]['array'],
                                    lat=self.data[tag]['lat'],
                                    lon=self.data[tag]['lon'],
                                    ax=axt, projection=projection,
                                    mapbound=mapbound,
                                    gridstep=gridstep,
                                    shift=shift,
                                    cmap=cmap,
                                    colorbarlabel=colorbarlabel,
                                    forcelabel=forcelabel,
                                    levels=levels,
                                    data_transform=data_transform,
                                    map_threshold=map_threshold,
                                    colorbardic=colorbardic,
                                    cbarkw=cbarkw,
                                    *args,
                                    **imgkw)
            mapimgdic[tag] = mapimg

        self.axdic = axdic
        self.mapimgdic = mapimgdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())

    def mapcontourf_split_axes(self, projection='cyl',mapbound='all',
                               gridstep=None,shift=False,
                               map_threshold=None,
                               cmap=None,colorbarlabel=None,forcelabel=None,
                               show_colorbar=True,
                               levels=None,data_transform=False,
                               smartlevel=False,
                               colorbardic={},cbarkw={},
                               gmapkw={},
                               mconfkw={},
                               **kwargs):
        '''
        bmap.mapcontourf each tag on a subplot.

        Parameters:
        -----------
        mconfkw: bmap.mapcontourf kwargs.
        c.f. bmap.mapcontourf
        kwargs:
            force_axs: force the axes.
            tagseq: the tag sequence, default is self._taglist
            tagcolor: the tag color.
            force_axdic: force a dictionary of parent_tag/axes pairs.
            ncols: num of columns when force_axs is None
            sharex,sharey: the same as plt.subplots
            tagpos: the position of parent_tag
            column_major: True if parent tags are deployed in column-wise.
            unit: used as ylabel for each subplot
            xlim: xlim
            tagprefix:
                1. default value is empty string.
                2. in case of 'alpha', alphabetic values will be appended.
                2. in case of 'numeric', numbers will be appended.
            tagbracket:
                1. default value is 'normal', i.e., "()" will be used.
            tagcolor: the tag color
                subkw: kwarg in plt.subplots function
        '''
        axdic = _creat_dict_of_tagaxes_by_tagseq_g(
                        default_tagseq=self._taglist,
                        default_tagpos='ouc',
                        **kwargs)
        mapconfdic={}
        for tag,axt in axdic.items():
            mapconf = bmap.mapcontourf(data=self.data[tag]['array'],
                                       lat=self.data[tag]['lat'],
                                       lon=self.data[tag]['lon'],
                                       ax=axt, projection=projection,
                                       mapbound=mapbound,
                                       gridstep=gridstep,
                                       shift=shift,
                                       cmap=cmap,
                                       colorbarlabel=colorbarlabel,
                                       forcelabel=forcelabel,
                                       levels=levels,
                                       map_threshold=map_threshold,
                                       smartlevel=smartlevel,
                                       data_transform=data_transform,
                                       colorbardic=colorbardic,
                                       cbarkw=cbarkw,
                                       gmapkw=gmapkw,
                                       show_colorbar=show_colorbar,
                                       **mconfkw)

            mapconfdic[tag] = mapconf
        self.axdic = axdic
        self.mapconfdic = mapconfdic
        self.lax = LabelAxes.LabelAxes(axdic.keys(),axdic.values())


    def add_Rectangle(self,grid,container_name='auto',
                      **kwargs):
        """
        Add a rectangle of by specifing (lat1,lon1) and (lat2,lon2).

        Parameters:
        -----------
        grid: should be a tuple of (lat1,lon1,lat2,lon2)
        """
        if container_name == 'auto':
            if hasattr(self,'mapimgdic'):
                containter_dic = self.mapimgdic
            elif hasattr(self,'mapconfdic'):
                containter_dic = self.mapconfdic
            else:
                raise ValueError("Basemap container not found.")
        else:
            containter_dic = getattr(self,container_name)

        (lat1,lon1,lat2,lon2)= grid

        recdic = {}
        for tag in containter_dic.keys():
            m = getattr(containter_dic[tag],'m')
            (x1,x2),(y1,y2) = m([lon1,lon2],[lat1,lat2])
            rec = mat.patches.Rectangle((x1,y1),x2-x1,y2-y1,**kwargs)
            m.ax.add_patch(rec)
            recdic[tag] = rec
        self.recdic = recdic


    def apply_function(self,func=None,taglist=None,copy=False):
        '''
        Apply a function on the array for the tags as specified in taglist
        '''
        if copy == True:
            pdtemp = self.copy()
        else:
            pdtemp = self

        taglist=_replace_none_by_given(taglist,self._taglist)
        for tag in taglist:
            pdtemp.data[tag]['array']=func(pdtemp.data[tag]['array'])
        return pdtemp

    @classmethod
    def merge_mdata(cls,*mdlist):
        """
        Merge the mdata.
        """
        newdata = {}
        for md in mdlist:
            newdata.update(md.data)
        md = Mdata(data=newdata)
        return md

    def copy(self):
        data=copy.deepcopy(self.data)
        md = Mdata(data)
        md.set_tag_order(self._taglist)
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
        lat = self.data[tag]['lat']
        return pa.DataFrame(dic,index=lat)

class NestMdata(object):
    """
    NestMdata receives a dictionary of Mdata object.
    """
    def __init__(self,dic_mdata):
        self.child_mdata = dic_mdata
        self.parent_tags = dic_mdata.keys()
        self.child_tags = self.child_mdata.values()[0].list_tags()

    def __repr__(self):
        return """parent tags: {0}""".format(self.parent_tags) + '\n' +\
        "child tags: {0}".format(self.child_tags)

    def mapcontourf_split_parent_tag(self,column_major=True,
                               xlabelpad=None,
                               ylabelpad=None,
                               projection='cyl',mapbound='all',
                               gridstep=None,shift=False,
                               map_threshold=None,
                               cmap=None,colorbarlabel=None,forcelabel=None,
                               show_colorbar=True,
                               levels=None,data_transform=False,
                               smartlevel=False,
                               colorbardic={},cbarkw={},
                               gmapkw={},
                               mconfkw={},
                               xlabelkw={},
                               ylabelkw={},
                               **kwargs):


        num_ptag = len(self.parent_tags)
        num_ctag = len(self.child_tags)
        if column_major:
            fig,axs = plt.subplots(nrows=num_ctag,ncols=num_ptag)
            pdic = OrderedDict()
            for i,ptag in enumerate(self.parent_tags):
                pdic[ptag] = LabelAxes.LabelAxes(tags=self.child_tags,axl=axs[:,i])
            lax2D = LabelAxes.LabelAxes2D(pdic)
        else:
            fig,axs = plt.subplots(nrows=num_ptag,ncols=num_ctag)
            pdic = OrderedDict()
            for i,ptag in enumerate(self.parent_tags):
                pdic[ptag] = LabelAxes.LabelAxes(tags=self.child_tags,axl=axs[i,:])
            lax2D = LabelAxes.LabelAxes2D(pdic)

        self.lax2D = lax2D
        for ptag,cmd in self.child_mdata.items():
            cmd.mapcontourf_split_axes(force_axdic=lax2D[ptag].data,
                               tagpos=False,
                               projection=projection,
                               mapbound=mapbound,
                               gridstep=gridstep,shift=shift,
                               map_threshold=map_threshold,
                               cmap=cmap,colorbarlabel=colorbarlabel,
                               forcelabel=forcelabel,
                               show_colorbar=show_colorbar,
                               levels=levels,data_transform=data_transform,
                               smartlevel=smartlevel,
                               colorbardic={},cbarkw={},
                               gmapkw={},
                               mconfkw={})

        if column_major:
            self.lax2D[self.parent_tags[0]].set_ylabel(self.child_tags,labelpad=ylabelpad,**ylabelkw)
            dt = self.lax2D.child_ix(self.child_tags[0])
            dt.set_xlabel(self.parent_tags,labelpad=xlabelpad,**xlabelkw)
            dt.apply(lambda ax:ax.xaxis.set_label_position('top'))
        else:
            dt = self.lax2D[self.parent_tags[0]]
            dt.set_xlabel(self.child_tags,labelpad=xlabelpad,**xlabelkw)
            dt.apply(lambda ax:ax.xaxis.set_label_position('top'))
            self.lax2D.child_ix(self.child_tags[0]).set_ylabel(self.parent_tags,labelpad=ylabelpad,**ylabelkw)


class PolyData(Pdata):
    _data_base_keylist=['poly']
    _new_entry=dict.fromkeys(_data_base_keylist,None)

    def add_tag(self,tag=None):
        self.data[tag]=PolyData._new_entry.copy()
        self._taglist.append(tag)

    def add_poly(self,data,tag):
        self.data[tag]['poly'] = data

    def add_entry_bydic(self,ydic):
        for tag,ydata in ydic.items():
            self.add_tag(tag)
            self.add_poly(ydata,tag)

    @classmethod
    def from_dict_of_poly(cls,ydic):
        md = PolyData()
        md.add_entry_bydic(ydic)
        return md

    @classmethod
    def from_DataFrame(cls,df,column_name):
        pass


    def draw(self,axes=None,**kwargs):
        self.polydic={}
        axes=_replace_none_axes(axes)
        for tag in self._taglist:
            col = mat.collections.PolyCollection(self.data[tag]['poly'],**kwargs)
            axes.add_collection(col)
            self.polydic[tag] = col
        axes.autoscale_view()

    def setp_tag(self,tagkw=False,**nested_attr_tag_value_dic):
        artist_dic = self.polydic
        self._setp_by_tag(artist_dic,tagkw=tagkw,**nested_attr_tag_value_dic)


