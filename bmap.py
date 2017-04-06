#!/usr/bin/env python

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import mathex as mathex
import os as os
import re as re
import scipy as sp
import mpl_toolkits.basemap as bmp
from mpl_toolkits.basemap import cm
import pdb
import netCDF4 as nc
from matplotlib.backends.backend_pdf import PdfPages
import copy as pcopy
import g
import pb
import tools

rcParams={}
rcParams['Antarctica']=True
rcParams['gridstep'] = False



def _remove_antarctica(lcol):
    """
    lcol is the line collections returned by m.drawcoastlines
    """
    segs = lcol.get_segments()
    for i, seg in enumerate(segs):
     # The segments are lists of ordered pairs in spherical (lon, lat), coordinates.
     # We can filter out which ones correspond to Antarctica based on latitude using numpy.any()
     if np.any(seg[:, 1] < -60):
          segs.pop(i)
    lcol.set_segments(segs)

def near5even(datain):
    if datain%5==0:
        dataout=datain
    else:
        if datain/5<0:
            dataout=np.ceil(datain/5)*5
        else:
            dataout=np.floor(datain/5)*5
    return dataout

class gmap(object):
    """
    Purpose: plot the map used for later contour or image plot.

    Note:
        return m,lonpro,latpro,latind,lonind
        1. m --> map drawed;
           lonpro/latpro --> lat/lon transferred to projection coords;
           latind/lonind --> index used to select the final data to be
                mapped (contour, or image).
           latm/lonm --> the latitude/longitude used to generate latpro/lonpro.
                Noth this does not necessarily have to be the same as original
                input lat/lon in the data, but they should have the same length.
                And latm/lonm is only adjusted to cover exactly the whole extent
                of mapping, rather than input lat/lon, in many cases are either
                gridcell-center or left/right side of the grid extent.
        2. lat must be descending and lon must be ascending.

    Parameters:
    -----------
    ax,lat,lon: Default will set up an axes with global coverage at 0.5-degree.
    centerlatend,centerlonend: True if the two ends of the input lat/lon represents
        the center of the grid rather than the exact limit, in this case the real
        input of lat/lon to make map will be automatically adjusted. For data with
        Global coverage, this check is automatically done by verifying if the two
        ends of lat is close to (90,-90) and the two ends of lon are close to (-180,180),
        or the higher end of lon is close to 360 if it ranges within (0,360)
    kwargs: used for basemap.Basemap method.

    Returns:
    --------
    A gmap object.

    Example:
        >>> fig,ax=g.Create_1Axes()
        >>> gmap=bmap.gmap(ax,'cyl',mapbound='all',lat=np.arange(89.75,-89.8,-0.5),lon=np.arange(-179.75,179.8,0.5),gridstep=(30,30))
        >>> x,y=gmap.m(116,40)  #plot Beijing
        >>> m.scatter(x,y,s=30,marker='o',color='r')
    """
    def __init__(self,ax=None,projection='cyl',mapbound='all',lat=None,lon=None,
                 gridstep=None,centerlatend=True,centerlonend=True,
                 resolution='c',
                 rlat=None,rlon=None,
                 xticks=None,yticks=None,gridon=None,
                 lwcont=0.4,**kwargs):

        # Some basic check and setting
        ax = tools._replace_none_axes(ax)
        lat = tools._replace_none_by_given(lat,np.arange(89.75,-89.8,-0.5))
        lon = tools._replace_none_by_given(lon,np.arange(-179.75,179.8,0.5))

        if gridstep is None:
            gridstep = rcParams['gridstep']
            if gridstep is None:
                gridstep = (30,30)

        # Check lat/lon
        step_lat = lat[0] - lat[1]
        if step_lat <= 0:
            raise TypeError("lat input is increasing!")
        step_lon = lon[1] - lon[0]
        if step_lon <= 0:
            raise TypeError("lon input is decreasing!")

        if abs(lat[0]-90.)<1e-4 or lat[-1]+90.<1e-4:
            centerlatend = False
        if lon[0]+180.<1e-4 or abs(lon[-1]-180.)<1e-4 or abs(lon[-1]-360.)<1e-4:
            centerlatend = False

        ## Draw map for different projections
        if projection=='cyl':

            if rlat is not None or rlon is not None:
                mapbound = 'all'
                lat=np.linspace(rlat[0],rlat[1],num=len(lat))
                lon=np.linspace(rlon[0],rlon[1],num=len(lon))
                centerlatend = False
                centerlonend = False

            #Get the boundary for mapping
            if isinstance(mapbound,dict):
                raise ValueError('cannot use dict for cyl projection')
            elif mapbound=='all':
                lat1=lat[-1]
                lat2=lat[0]
                lon1=lon[0]
                lon2=lon[-1]
                #when the lat,lon input is of equal distance and the end of lat/lon
                #is the center of the grid, we have to adjust the end of the input
                #lat/lon to account for this.
                if centerlatend:
                    lat1 = lat1 - step_lat/2.
                    lat2 = lat2 + step_lat/2.
                if centerlonend:
                    lon1 = lon1 - step_lon/2.
                    lon2 = lon2 + step_lon/2.

            else:
                lat1=mapbound[0]
                lat2=mapbound[1]
                lon1=mapbound[2]
                lon2=mapbound[3]

            #draw the map, parallels and meridians
            m=bmp.Basemap(projection=projection,llcrnrlat=lat1,urcrnrlat=lat2,
                          llcrnrlon=lon1,urcrnrlon=lon2,resolution=resolution,ax=ax,
                          **kwargs)
            lcol = m.drawcoastlines(linewidth=lwcont)
            if not rcParams['Antarctica']:
                _remove_antarctica(lcol)

            if gridstep is not None and gridstep!=False:
                para_range=np.arange(near5even(lat1),near5even(lat2)+0.1,gridstep[0])
                meri_range=np.arange(near5even(lon1),near5even(lon2)+0.1,gridstep[1])
                m.drawparallels(para_range,labels=[1,0,0,0])
                m.drawmeridians(meri_range,labels=[0,0,0,1])

            #make the grid for mapping ndarray
            latind=np.nonzero((lat>=lat1)&(lat<=lat2))[0]
            lonind=np.nonzero((lon>=lon1)&(lon<=lon2))[0]
            numlat=len(latind)
            numlon=len(lonind)
            lonm,latm=m.makegrid(numlon,numlat)
            latm=np.flipud(latm)
            lonpro,latpro=m(lonm,latm)

        # npstere stands for north polar stereographic.
        elif projection=='npstere':
            if not isinstance(mapbound,dict):
                raise ValueError('please use dict to specify')
            else:
                if 'blat' in mapbound:
                    raise KeyError("Message from bmp.gmap: blat deprecated, use boundinglat instead.")

                m=bmp.Basemap(projection='npstere',boundinglat=mapbound['boundinglat'],
                              lon_0=mapbound['lon_0'],resolution=resolution,ax=ax,
                              **kwargs)
                m.drawcoastlines(linewidth=0.7)
                m.fillcontinents(color='0.8',zorder=0)

                if gridstep is not None and gridstep!=False:
                    m.drawparallels(np.arange(mapbound['para0'],90.01,gridstep[0]),
                                    labels=[1,0,0,0],fontsize=8,linewidth=0.5,color='0.7')
                    m.drawmeridians(np.arange(-180.,181.,gridstep[1]),
                                    labels=[0,0,0,0],fontsize=8,linewidth=0.5,color='0.7')
                #make the grid
                lat1=mapbound['boundinglat']
                latind=np.nonzero(lat>=lat1)[0]
                lonind=np.arange(len(lon))
                latnew=np.linspace(90, lat1, num=len(latind), endpoint=True) # endpoint should be True as we
                if lon[-1]>180:                                              # want to make the grid covering
                    lonnew=np.linspace(0,360,num=len(lonind),endpoint=True)  # the whole map extent no matter
                else:                                                        # how the original lat/lon in the
                    lonnew=np.linspace(-180,180,num=len(lonind),endpoint=True) # data are presented.
                lonm,latm=np.meshgrid(lonnew,latnew)
                lonpro,latpro=m(lonm,latm)

        elif projection=='kav7':
            if not isinstance(mapbound,dict):
                raise ValueError('please use dict to specify')
            else:
                m=bmp.Basemap(projection='kav7',
                              lon_0=mapbound['lon_0'],resolution=resolution,ax=ax,
                              **kwargs)
                m.drawcoastlines(linewidth=0.7)
                m.fillcontinents(color='0.8',zorder=0)
                if gridstep is not None and gridstep!=False:
                    m.drawparallels(np.arange(-90,91.,gridstep[0]),
                                    labels=[1,0,0,0],fontsize=10)
                    m.drawmeridians(np.arange(-180.,181.,gridstep[1]),
                                    labels=[0,0,0,0],fontsize=10)
                #make the grid
                lat1=lat[-1];lat2=lat[0]
                lon1=lon[0];lon2=lon[-1]
                latind=np.nonzero((lat>=lat1)&(lat<=lat2))[0]
                lonind=np.nonzero((lon>=lon1)&(lon<=lon2))[0]
                numlat=len(latind)
                numlon=len(lonind)
                lonm,latm=m.makegrid(numlon,numlat)
                latm=np.flipud(latm)
                lonpro,latpro=m(lonm,latm)
        else:
            raise ValueError('''projection '{0}' not supported'''
                             .format(projection))
        if xticks is not None:
            ax.set_xticks(xticks)
            if gridon is None:
                gridon = True

        if yticks is not None:
            ax.set_yticks(yticks)
            if gridon is None:
                gridon = True

        if gridon:
            ax.grid('on')

        self.m = m
        self.lonpro = lonpro
        self.latpro = latpro
        self.latind = latind
        self.lonind = lonind
        self.latm = latm
        self.lonm = lonm
        self.latorg_all = lat
        self.lonorg_all = lon
        self.latorg_used = lat[latind]
        self.lonorg_used = lon[lonind]

def _transform_data(pdata,levels,data_transform,extend='neither'):
    '''
    Return [pdata,plotlev,plotlab,extend,trans_base_list];
    if data_transform == False, trans_base_list = None.

    Notes:
    ------
    pdata: data used for contourf plotting.
    plotlev: the levels used in contourf plotting.
    extend: the value for parameter extand in contourf.
    trans_base_list: cf. mathex.plot_array_transg
    '''
    if levels is None:
        ftuple = (pdata,None,None,extend)
        if data_transform==True:
            raise Warning("Strange levels is None but data_transform is True")
    #level is given
    else:
        if data_transform==True:
            #make the data transform before plotting.
            pdata_trans,plotlev,plotlab,trans_base_list = \
                mathex.plot_array_transg(pdata, levels, copy=True)
            if np.isneginf(plotlab[0]) and np.isposinf(plotlab[-1]):
                ftuple = (pdata_trans,plotlev[1:-1],plotlab,'both')
            elif np.isneginf(plotlab[0]) or np.isposinf(plotlab[-1]):
                raise ValueError('''only one extreme set as infinitive, please
                    set both as infinitive if arrow colorbar is wanted.''')
            else:
                ftuple = (pdata_trans,plotlev,plotlab,extend)
        #data_transform==False
        else:
            plotlev = pb.iteflat(levels)
            plotlab = plotlev #label same as levels
            if np.isneginf(plotlab[0]) and np.isposinf(plotlab[-1]):
                #here the levels would be like [np.NINF,1,2,3,np.PINF]
                #in following contourf, all values <1 and all values>3 will be
                #automatically plotted in the color of two arrows.
                #easy to see in this example:
                #a=np.tile(np.arange(10),10).reshape(10,10);
                #fig,ax=g.Create_1Axes();
                #cs=ax.contourf(a,levels=np.arange(2,7),extend='both');
                #plt.colorbar(cs)
                ftuple = (pdata,plotlev[1:-1],plotlab,'both')
            elif np.isneginf(plotlab[0]) or np.isposinf(plotlab[-1]):
                raise ValueError('''only one extreme set as infinitive, please
                    set both as infinitive if arrow colorbar is wanted.''')
            else:
                ftuple = (pdata,plotlev,plotlab,extend)
    datalist = list(ftuple)

    if data_transform == True:
        datalist.append(trans_base_list)
    else:
        datalist.append(None)
    return datalist

def _generate_colorbar_ticks_label(data_transform=False,
                                   colorbarlabel=None,
                                   trans_base_list=None,
                                   forcelabel=None,
                                   plotlev=None,
                                   plotlab=None):
    '''
    Return (colorbar_ticks,colorbar_labels)
    '''
    #data_transform==True and levels is not None
    if data_transform==True:
        if colorbarlabel is not None:
            colorbarlabel=pb.iteflat(colorbarlabel)
            transformed_colorbarlabel_ticks,x,y,trans_base_list = \
                mathex.plot_array_transg(colorbarlabel, trans_base_list,
                                         copy=True)

        #Note if/else blocks are organized in 1st tire by check if the two
        #ends are -inf/inf and 2nd tire by check if colorbarlabel is None
        if np.isneginf(plotlab[0]) and np.isposinf(plotlab[-1]):
            if colorbarlabel is not None:
                ftuple = (transformed_colorbarlabel_ticks,colorbarlabel)
            else:
                ftuple = (plotlev,plotlab[1:-1])
        elif np.isneginf(plotlab[0]) or np.isposinf(plotlab[-1]):
            raise ValueError("It's strange to set only side as infitive")
        else:
            if colorbarlabel is not None:
                ftuple = (transformed_colorbarlabel_ticks,colorbarlabel)
            else:
                ftuple = (plotlev,plotlab)

    #data_transform==False
    else:
        if np.isneginf(plotlab[0]) and np.isposinf(plotlab[-1]):
            #if colorbarlabel is forced, then ticks and ticklabels will be forced.
            if colorbarlabel is not None:
                ftuple = (colorbarlabel,colorbarlabel)
            #This by default will be done, it's maintained here only for clarity.
            else:
                ftuple = (plotlab[1:-1],plotlab[1:-1])
        elif np.isneginf(plotlab[0]) or np.isposinf(plotlab[-1]):
            raise ValueError("It's strange to set only side as infitive")
        else:
            if colorbarlabel is not None:
                ftuple = (colorbarlabel,colorbarlabel)
            else:
                ftuple = (plotlab,plotlab)

    ftuple = list(ftuple)
    if forcelabel is not None:
        if len(forcelabel) != len(ftuple[1]):
            raise ValueError('''the length of the forcelabel and the
                length of labeled ticks is not equal!''')
        else:
            ftuple[1] = forcelabel

    return ftuple

def _generate_smartlevel(pdata):
    """
    generate smart levels by using the min, percentiles from 5th
        to 95th with every 5 as the step, and the max value.
    """

    def even_num(num):
        if num >= 10:
            return int(num)
        else:
            return round(num,4)

    def extract_percentile(array,per):
        return even_num(np.percentile(array,per))

    def generate_smartlevel_from_1Darray(array):
        vmax = even_num(np.max(array))
        vmin = even_num(np.min(array))
        per_level = map(lambda x:extract_percentile(array,x),
                        np.arange(5,96,5))
        return np.array([vmin]+per_level+[vmax])

    if np.isnan(np.sum(pdata)):
        pdata = np.ma.masked_invalid(pdata)

    if np.ma.isMA(pdata):
        array1D = pdata[np.nonzero(~pdata.mask)]
    else:
        array1D = pdata.flatten()

    return generate_smartlevel_from_1Darray(array1D)

def _generate_map_prepare_data(data=None,lat=None,lon=None,
                               projection='cyl',
                               mapbound='all',
                               rlat=None,rlon=None,
                               gridstep=(30,30),
                               shift=False,
                               map_threshold=None,
                               mask=None,
                               levels=None,
                               cmap=None,
                               smartlevel=None,
                               data_transform=False,
                               gmapkw={},
                               extend='neither',
                               ax=None):
    """
    This function makes the map, and transform data for ready
        use of m.contourf or m.imshow
    """
    if shift==True:
        data,lon=bmp.shiftgrid(180,data,lon,start=False)
    mgmap=gmap(ax,projection,mapbound,lat,lon,gridstep,rlat=rlat,rlon=rlon,**gmapkw)
    m,lonpro,latpro,latind,lonind = (mgmap.m, mgmap.lonpro, mgmap.latpro,
                                    mgmap.latind, mgmap.lonind)

    pdata = data[latind[0]:latind[-1]+1,lonind[0]:lonind[-1]+1]
    #mask by map_threshold
    pdata = mathex.ndarray_mask_by_threshold(pdata,map_threshold)
    #apply mask
    if mask is not None:
        pdata = np.ma.masked_array(pdata,mask=mask)
    #generate the smartlevel
    if smartlevel == True:
        if levels is not None:
            raise ValueError("levels must be None when smartlevel is True!")
        else:
            levels = _generate_smartlevel(pdata)
            data_transform = True
    #prepare the data for contourf
    pdata,plotlev,plotlab,extend,trans_base_list = \
        _transform_data(pdata,levels,data_transform,extend=extend)
    return (mgmap,pdata,plotlev,plotlab,extend,
            trans_base_list,data_transform)

def _set_colorbar(m,cs,colorbardic={},
                  levels=None,
                  data_transform=False,
                  colorbarlabel=None,
                  trans_base_list=None,
                  forcelabel=None,
                  show_colorbar=True,
                  plotlev=None,
                  plotlab=None,
                  cbarkw={}):
    """
    Wrap the process for setting colorbar.
    """
    #handle the colorbar attributes by using dictionary which flexibility.
    if show_colorbar == False:
        cbar = None
    else:
        location = colorbardic.get('location','right')
        size = colorbardic.get('size','3%')
        pad = colorbardic.get('pad','2%')
        cbar=m.colorbar(cs,location=location, size=size, pad=pad,**cbarkw)
        #set colorbar ticks and colorbar label
        if levels is None:
            pass
        else:
            ticks,labels = \
                _generate_colorbar_ticks_label(data_transform=data_transform,
                                               colorbarlabel=colorbarlabel,
                                               trans_base_list=trans_base_list,
                                               forcelabel=forcelabel,
                                               plotlev=plotlev,
                                               plotlab=plotlab)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)
    return cbar

class mapcontourf(object):
    """
    Purpose: plot a map on 'cyl' or 'npstere' projection.
    Arguments:
        ax --> An axes instance
        projection --> for now two projections have been added:
            related parameters: mapbound
            1. 'cyl' -- for global and regional mapping. In case of
                using mapbound, it should give the lat/lon values of
                not as the center of the grid cells but as the real
                boundary including the necessary shift of half of the
                resolution.
            2. 'npstere' -- for North Polar STEREographic map, needs to
                properly set mapbound keyword.
        lat,lon --> geographic coordinate variables; lat must be in
            desceding order and lon must be ascending.
        mapbound --> specify the bound for mapping;
            1. 'cyl'
                tuple containing (lat1,lat2,lon1,lon2); lat1 --> lower
                parallel; lat2 --> upper parallel; lon1 --> left meridian;
                lon2 --> right meridian; default 'all' means plot
                the extent of input lat, lon coordinate variables;
                for global mapping, set (-90,90,-180,180) or (-90,90,0,360).
            2. 'npstere'
                mapbound={'boundinglat':45,'lon_0':0,'para0':40}
                boundinglat --> boundinglat in the bmp.Basemap method.
                    The southern limit for mapping. This parallel is
                    tangent to the edge of the plot.
                lon_0 --> center of desired map domain, it's at 6-o' clock.
                para0 --> CustimizedParameter. souther boundary for parallel ticks,
                the default norther limit is 90; default longitude is 0-360
                (or -180-180)
        gridstep --> the step for parallel and meridian grid for the map,
            tuple containing (parallel_step, meridian_step).
        levels --> default None; levels=[-5,-2,-1,0,1,2,5] ;
            or levels=[(-10,-4,-2,-1,-0.4),(-0.2,-0.1,0,0.1,0.2),
                        (0.4,1,2,4,10)].
            1.  Anything that can work as input for function pb.iteflat()
                will work.
            2.  If the first and last element of pb.iteflat(levels) is
                np.NINF and np.PINF, the colorbar of contourf plot will
                use the 'two-arrow' shape.
            3.  If data_transform==True, the input data will be transformed
                from pb.iteflat(levels) to
                np.linspace(1,len(pb.iteflat(interval_original)). this can
                help to create arbitrary contrasting in the plot.
                cf. mathex.plot_array_transg
        smartlevel:
            1. when True, a "smart" level will be generated by
               using the min,max value and the [5th, 10th, ..., 95th]
               percentile of the input array.
            2. it will be applied after applying the mask_threshold.
        data_transform:
            1. set as True if increased contrast in the plot is desired.
                In this case the function mathex.plot_array_transg will
                be called and pb.iteflat(levels) will be used as original
                interval for data transformation.
            2. In case of data_transform==False, pb.iteflat(levels)
                will be used directly in the plt.contour function for
                ploting and hence no data transformation is made. The
                treatment by this way allows very flexible
                (in a mixed way) to set levels.
            3. In any case, if np.NINF and np.PINF as used as two
                extremes of levels, arrowed colorbar will be returned.

        colorbarlabel:
            1. used to put customized colorbar label and this will override
                using levels as colorbar. IF colorbarlabel is not None,
                colorbar ticks and labels will be set using colorbarlabel.
                so this means colorbarlabel could only be array or
                list of numbers.
            2. If data_transform==True, colorbar will also be transformed
                accordingly. In this case, the colorbar ticks will use
                transformed colorbarlabel data, but colorbar ticklables
                will use non-transformed colorbarlabel data. This means
                the actual ticks numbers and labels are not the same.

        forcelabel --> to force the colorbar label as specified by forcelabel.
            This is used in case to set the labels not in numbers but in
            other forms (eg. strings).
            In case of data_transform = True, levels will be used to
            specifiy levels for the original colorbar, colorbarlabel will
            be used to create ticks on colrobar which will be labeled,
            if forcelabel=None, then colorbarlabel will agined be used
            to label the ticks, otherwise forcelabel will be used to
            label the ticks on the colorbar. So this means forcelabel will
            mainly be list of strings.

        data --> numpy array with dimension of len(lat)Xlen(lon)
        map_threshold --> dictionary like {'lb':2000,'ub':5000}, data
            less than 2000 and greater than 5000 will be masked.
            Note this will be applied before data.
               transform.
        shift --> boolean value. False for longtitude data ranging [-180,180];
            for longtitude data ranging [0,360] set shift to True if a
            180 east shift is desired. if shift as True, the mapbound
            range should be set using shifted longtitude
            (use -180,180 rather than 0,360).
        colorbardic --> dictionary to specify the attributes for colorbar,
            translate all the keys in function bmp.Basemap.colorbar()
            into keys in colorbardic to manipulation.

    Note:
        1.  lat must be descending, and lon must be ascending.
        2*. NOTE use both data_transform=True and impose unequal
            colorbarlabel could be very confusing! Because normaly in
            case of data_transform as True the labels are ALREADY
            UNEQUALLY distributed!

            an example to use colorbarlabel and forcelabel:
            data_transform=True,
            levels=[0,1,2,3,4,5,6,7,8]
            colorbarlabel=[0,2,4,6,8]
            forcelabel=['extreme low','low','middle','high','extreme high']

            So colorbarlabel will set both ticks and labels, but forcelabel
            will further overwrite the labels.

        3. This function has been test using data, the script and
            generated PNG files are availabe at ~/python/bmaptest
    See also:
        mathex.plot_array_transg; gmap

    docstring from gmap:
    --------------------
    Purpose: plot the map used for later contour or image plot.

    Note:
        return m,lonpro,latpro,latind,lonind
        1. m --> map drawed;
           lonpro/latpro --> lat/lon transferred to projection coords;
           latind/lonind --> index used to select the final data to be
                mapped (contour, or image).
           latm/lonm --> the latitude/longitude used to generate latpro/lonpro.
                Noth this does not necessarily have to be the same as original
                input lat/lon in the data, but they should have the same length.
                And latm/lonm is only adjusted to cover exactly the whole extent
                of mapping, rather than input lat/lon, in many cases are either
                gridcell-center or left/right side of the grid extent.
        2. lat must be descending and lon must be ascending.

    Parameters:
    -----------
    ax,lat,lon: Default will set up an axes with global coverage at 0.5-degree.
    centerlatend,centerlonend: True if the two ends of the input lat/lon represents
        the center of the grid rather than the exact limit, in this case the real
        input of lat/lon to make map will be automatically adjusted. For data with
        Global coverage, this check is automatically done by verifying if the two
        ends of lat is close to (90,-90) and the two ends of lon are close to (-180,180),
        or the higher end of lon is close to 360 if it ranges within (0,360)
    kwargs: used for basemap.Basemap method.

    Returns:
    --------
    A gmap object.
    """
    def __init__(self,data=None,lat=None,lon=None,ax=None,
                 projection='cyl',mapbound='all',
                 rlat=None,rlon=None,
                 gridstep=None,shift=False,
                 map_threshold=None,mask=None,
                 cmap=None,colorbarlabel=None,forcelabel=None,
                 show_colorbar=True,
                 smartlevel=False,
                 levels=None,data_transform=False,
                 colorbardic={},
                 extend='neither',
                 cbarkw={},
                 gmapkw={},
                 contfkw={},
                 ):


        (mgmap,pdata,plotlev,plotlab,extend,
         trans_base_list,data_transform) = \
            _generate_map_prepare_data(data=data,lat=lat,lon=lon,
                                       projection=projection,
                                       mapbound=mapbound,
                                       rlat=rlat,rlon=rlon,
                                       gridstep=gridstep,
                                       shift=shift,
                                       map_threshold=map_threshold,
                                       mask=mask,
                                       levels=levels,
                                       cmap=cmap,
                                       smartlevel=smartlevel,
                                       data_transform=data_transform,
                                       gmapkw=gmapkw,
                                       ax=ax,
                                       extend=extend)

        #print extend
        #make the contourf plot
        cs=mgmap.m.contourf(mgmap.lonpro,mgmap.latpro,pdata,
                            levels=plotlev,extend=extend,cmap=cmap,**contfkw)
        ##handle colorbar
        cbar = _set_colorbar(mgmap.m,cs,
                             colorbardic=colorbardic,
                             levels=plotlev,
                             data_transform=data_transform,
                             colorbarlabel=colorbarlabel,
                             trans_base_list=trans_base_list,
                             forcelabel=forcelabel,
                             plotlev=plotlev,
                             plotlab=plotlab,
                             cbarkw=cbarkw,
                             show_colorbar=show_colorbar)
        #return
        self.m = mgmap.m
        self.cs = cs
        self.cbar = cbar
        self.plotlev = plotlev
        self.plotlab = plotlab
        self.ax = mgmap.m.ax
        self.trans_base_list = trans_base_list
        self.gmap = mgmap
        self.pdata = pdata

        if levels is None:
            pass
        else:
            cbar_ticks,cbar_labels = \
                _generate_colorbar_ticks_label(data_transform=data_transform,
                                               colorbarlabel=colorbarlabel,
                                               trans_base_list=trans_base_list,
                                               forcelabel=forcelabel,
                                               plotlev=plotlev,
                                               plotlab=plotlab)

            self.cbar_ticks = cbar_ticks
            self.cbar_labels = cbar_labels

    def colorbar(self,cax=None,**kwargs):
        """
        set colorbar on specified cax.

        kwargs applies for plt.colorbar
        """
        cbar = plt.colorbar(self.cs,cax=cax,**kwargs)
        cbar.set_ticks(self.cbar_ticks)
        cbar.set_ticklabels(self.cbar_labels)
        return cbar



class mapimshow(object):
    """
    Purpose: plot a map on cyl projection.
    Arguments:
        ax --> An axes instance
        lat,lon --> geographic coordinate variables;
        mapbound --> tuple containing (lat1,lat2,lon1,lon2);
                lat1 --> lower parallel; lat2 --> upper parallel;
                lon1 --> left meridian; lon2 --> right meridian;
                default 'all' means plot the extent of input lat, lon
                coordinate variables;
        gridstep --> the step for parallel and meridian grid for the map,
            tuple containing (parallel_step, meridian_step).
        vmin,vmax --> as in plt.imshow function
        data --> numpy array with dimension of len(lat)Xlen(lon)
        shift --> boolean value. False for longtitude data ranging [-180,180];
            for longtitude data ranging [0,360] set shift to True if
            a 180 east shift is desired.

    args,kwargs: for plt.imshow
    """
    def __init__(self,data=None,lat=None,lon=None,ax=None,
                 rlat=None,rlon=None,
                 projection='cyl',mapbound='all',
                 gridstep=(30,30),shift=False,map_threshold=None,
                 cmap=None,colorbarlabel=None,forcelabel=None,
                 show_colorbar=True,
                 smartlevel=False,
                 levels=None,data_transform=False,
                 interpolation='none',
                 extend='neither',
                 colorbardic={},
                 cbarkw={},
                 gmapkw={},
                 *args,
                 **kwargs):

        (mgmap,pdata,plotlev,plotlab,extend,
         trans_base_list,data_transform) = \
            _generate_map_prepare_data(data=data,lat=lat,lon=lon,
                                       rlat=rlat,rlon=rlon,
                                       projection=projection,
                                       mapbound=mapbound,
                                       gridstep=gridstep,
                                       shift=shift,
                                       map_threshold=map_threshold,
                                       levels=levels,
                                       cmap=cmap,
                                       smartlevel=smartlevel,
                                       data_transform=data_transform,
                                       gmapkw=gmapkw,
                                       ax=ax,
                                       extend=extend)

        # 2017-02-15
        # Here is to accommodate the case of data_transform=True, in
        # this case before of calculation error, the minimum value in
        # pdata is sometimes a little bigger than plotlev[0]. This makes
        # plotlev[0] is not displayed on the colorbar (because the
        # minimum value of pdata is bigger), and the display of plotlab
        # will shift by one tick in this case.
        if plotlev is not None:
            vmin = plotlev[0]
            vmax = plotlev[-1]
        else:
            vmin = None
            vmax = None

        cs=mgmap.m.imshow(pdata,cmap=cmap,origin='upper',
                          interpolation=interpolation,
                          vmin=vmin,vmax=vmax,
                          *args,**kwargs)

        cbar = _set_colorbar(mgmap.m,cs,
                             colorbardic=colorbardic,
                             levels=plotlev,
                             data_transform=data_transform,
                             colorbarlabel=colorbarlabel,
                             trans_base_list=trans_base_list,
                             forcelabel=forcelabel,
                             plotlev=plotlev,
                             plotlab=plotlab,
                             cbarkw=cbarkw,
                             show_colorbar=show_colorbar)

        self.m = mgmap.m
        self.cs = cs
        self.cbar = cbar
        self.plotlev = plotlev
        self.plotlab = plotlab
        self.ax = mgmap.m.ax
        self.trans_base_list = trans_base_list
        self.gmap = mgmap
        self.pdata = pdata

class mappcolormesh(object):
    """
    Purpose: plot a map on cyl projection.
    Arguments:
        ax --> An axes instance
        lat,lon --> geographic coordinate variables;
        mapbound --> tuple containing (lat1,lat2,lon1,lon2);
                lat1 --> lower parallel; lat2 --> upper parallel;
                lon1 --> left meridian; lon2 --> right meridian;
                default 'all' means plot the extent of input lat, lon
                coordinate variables;
        gridstep --> the step for parallel and meridian grid for the map,
            tuple containing (parallel_step, meridian_step).
        vmin,vmax --> as in plt.imshow function
        data --> numpy array with dimension of len(lat)Xlen(lon)
        shift --> boolean value. False for longtitude data ranging [-180,180];
            for longtitude data ranging [0,360] set shift to True if
            a 180 east shift is desired.

    args,kwargs: for plt.imshow
    """
    def __init__(self,data=None,lat=None,lon=None,ax=None,
                 rlat=None,rlon=None,
                 projection='cyl',mapbound='all',
                 gridstep=(30,30),shift=False,map_threshold=None,
                 cmap=None,colorbarlabel=None,forcelabel=None,
                 show_colorbar=True,
                 smartlevel=False,
                 levels=None,data_transform=False,
                 interpolation='none',
                 extend='neither',
                 colorbardic={},
                 cbarkw={},
                 gmapkw={},
                 *args,
                 **kwargs):

        (mgmap,pdata,plotlev,plotlab,extend,
         trans_base_list,data_transform) = \
            _generate_map_prepare_data(data=data,lat=lat,lon=lon,
                                       rlat=rlat,rlon=rlon,
                                       projection=projection,
                                       mapbound=mapbound,
                                       gridstep=gridstep,
                                       shift=shift,
                                       map_threshold=map_threshold,
                                       levels=levels,
                                       cmap=cmap,
                                       smartlevel=smartlevel,
                                       data_transform=data_transform,
                                       gmapkw=gmapkw,
                                       ax=ax,
                                       extend=extend)

        # 2017-02-15
        # Here is to accommodate the case of data_transform=True, in
        # this case before of calculation error, the minimum value in
        # pdata is sometimes a little bigger than plotlev[0]. This makes
        # plotlev[0] is not displayed on the colorbar (because the
        # minimum value of pdata is bigger), and the display of plotlab
        # will shift by one tick in this case.
        if plotlev is not None:
            vmin = plotlev[0]
            vmax = plotlev[-1]
        else:
            vmin = None
            vmax = None

        cs=mgmap.m.pcolormesh(mgmap.lonpro,mgmap.latpro,pdata,
                              cmap=cmap,
                              vmin=vmin,vmax=vmax,
                              *args,**kwargs)

        cbar = _set_colorbar(mgmap.m,cs,
                             colorbardic=colorbardic,
                             levels=plotlev,
                             data_transform=data_transform,
                             colorbarlabel=colorbarlabel,
                             trans_base_list=trans_base_list,
                             forcelabel=forcelabel,
                             plotlev=plotlev,
                             plotlab=plotlab,
                             cbarkw=cbarkw,
                             show_colorbar=show_colorbar)

        self.m = mgmap.m
        self.cs = cs
        self.cbar = cbar
        self.plotlev = plotlev
        self.plotlab = plotlab
        self.ax = mgmap.m.ax
        self.trans_base_list = trans_base_list
        self.gmap = mgmap
        self.pdata = pdata


def mcon_set_clim_color(mcon,over=(None,None),under=(None,None)):
    """
    Set the (value,color) for the over/under of colormap.

    Parameters:
    -----------
    over/under: tuple, first element as value, second as color.
    """
    if over[1] is not None:
        mcon.cs.cmap.set_over(over[1])
    if under[1] is not None:
        mcon.cs.cmap.set_under(under[1])
    mcon.cs.set_clim(vmin=under[0],vmax=over[0])
    mcon.colorbar()





