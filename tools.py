#!/usr/bin/env python

import g
from collections import OrderedDict


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

def append_doc_of(fun):
    def decorator(f):
        f.__doc__ += fun.__doc__
        return f
    return decorator

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


def DataFrame_from_Dic_Key_as_Index(data,columns=None):
    """
    Purpose: to convert dictionary to pandas dataframe using dic.keys() as index and impose column names by setting columns=['var1','var2','var3']
    Note:
        1. data is a dictionary, columns is list of strings representing new dataframe column names.
        2. The original keys of input dictionary serve as index for new dataframe; len(columns) must be equal to len(data[key])
        3*. currently (2012/05/14) data[key]=(1D np.array -- or list -- or only one number) has been tested.
    Example:
        In [41]: print p2s2.keys()
        [1952, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]
        In [42]: p2s2[1952]
        Out[42]: array([ 16.57142857,   2.        ,  42.        ])
        In [51]: df=mathex.DataFrame_fromdic(p2s2,columns=['mean','min','max'])
        In [52]: df.head()
        Out[52]: 
              max       mean  min
              1952   42  16.571429    2
              1955   55  23.400000    5
              1956   35  16.714286    4
              1957   37  23.600000   11
              1958   71  39.666667   11
        In [53]: df.index
        Out[53]: 
        Int64Index([1952, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964,
               1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975,
               1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1987,
               1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
               1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006], dtype=int64)

    """
    temp=dict()
    if columns == None:
        raise ValueError('columns names must be provided!')
    else:
        for colname in columns:
            temp[colname]=[]
    for i in data.keys():
        if len(columns)==1:
            temp[colname].append(data[i])
        else:
            for j,colname in enumerate(columns): 
                temp[colname].append(data[i][j])
    tempdf=pa.DataFrame(temp,index=data.keys())
    return tempdf

def csv_concat(outfile,infile_list,index_col=None):
    """
    Purpose: Used to concat several csv files with the same column names.
    """
    if not isinstance(infile_list,list):
        raise TypeError("infile_list must be provided as a list")
    else:
        df_list=[]
        for filename in infile_list:
            dft = pa.read_csv(filename,index_col=index_col)
            df_list.append(dft)
        df_all = pa.concat(df_list)
        df_all.to_csv(outfile)
        return df_all

def expand_by_interval(firstPos,numrep,horizontal=None,vertical=None):
    """
    Expand the given first postioin (2-length tuple or 2-length list) to have
        numrep points by specifing horizontal or vertical interval.
    """
    x0,y0 = firstPos
    poslist = []
    if horizontal is not None:
        for num in range(numrep):
            xt,yt = (x0+horizontal*num,y0)
            poslist.append((xt,yt))
        return poslist
    elif vertical is not None:
        for num in range(numrep):
            xt,yt = (x0,y0+vertical*num)
            poslist.append((xt,yt))
        return poslist
    else:
        raise ValueError("both horizontal and vertical are None!")

def _propagate(tags,arg,itearg=False):
    """
    Propagate the arg input to a (ordered)dict. The behaviour varies
        according to input of arg and itearg.

    Parameters:
    -----------
    arg: the argument that's to be broadcasted, in case of:
        dict: will be returned directly.
        non-list: will be broadcast to form the output dict.
        list:
            itearg == True: each member of the list is an iterable object,
                currently could be tuple or list.
            Otherwise will be mapped to default tags to form output dict.
    itearg: True if the arg itself is expected to be an iterable object,
        for example, tuple or list.
    """
    tagnum = len(tags)
    if isinstance(arg,list):
        if itearg:
            if isinstance(arg[0],(tuple,list,np.ndarray)):
                if len(arg) != tagnum:
                    raise ValueError("""list length expected to be {0}"""
                                     .format(tagnum))
                else:
                    return OrderedDict(zip(tags,arg))
            else:
                return dict(zip(tags,[arg]*tagnum))
        else:
            if len(arg) != tagnum:
                raise ValueError("""list length expected to be {0}"""
                                 .format(tagnum))
            else:
                return OrderedDict(zip(tags,arg))
    elif isinstance(arg,dict):
        return arg
    else:
        return dict(zip(tags,[arg]*tagnum))





