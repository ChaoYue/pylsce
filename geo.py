#!/usr/bin/env python

from osgeo import ogr
from osgeo import gdal
import matplotlib as mat
import pandas as pa
import numpy as np
import Pdata
import gnc
from collections import OrderedDict

def _check_df_inner_out_ring_validity(df):
    if not isinstance(df,pa.DataFrame):
        raise TypeError("first argument must be DataFrame")
    else:
        if 'inner_ring' not in df.columns or 'out_ring' not in df.columns:
            raise NameError("inner_ring/out_ring not dataframe column")
        else:
            pass


def get_layer_attribute_table(layer,feature_range=None):
    """
    Read the items(attributes) of the features in the layer as an attribure table.

    Parameters:
    -----------
    layer: the layer for which feature attribute tables to be read.
    feature_range: list type, used to indicate the range of the features to be read.
        Defual is the range(layer.GetFeatureCount()),i.e., all features.

    Notes:
    ------
    The result is like what has been show as Attribute Table in ArcGIS.
    """
    if feature_range is None:
        select_range = range(layer.GetFeatureCount())
    else:
        select_range = feature_range
    data_list = []
    index_list = []

    for i in select_range:
        feature = layer.GetFeature(i)
        data_list.append(feature.items())
        index_list.append(i)
    return pa.DataFrame(data_list,index=index_list)

def get_line_from_linearring(ring):
    """
    Get the list of vertices in the linearring.

    Returns:
    --------
    a list of tuples which represent the vertices.
    """
    geometry_name = ring.GetGeometryName()
    if geometry_name != 'LINEARRING':
        raise TypeError("The type is {0}".format(geometry_name))
    else:
        num_point = ring.GetPointCount()
        d = [ring.GetPoint(i) for i in range(num_point)]
        line = [(a,b) for (a,b,c) in d]
        return line


def get_linelist_from_polygon(polygon):
    """
    Get a "polygon" from the polygon object.

    Returns:
    --------
    A list of sublist, each sublist is a list of vertices from
        a linearring object.
    """
    geometry_name = polygon.GetGeometryName()
    if geometry_name != 'POLYGON':
        raise TypeError("the type is {0}".format(geometry_name))
    else:
        geocount = polygon.GetGeometryCount()
        linelist = []
        for i in range(geocount):
            ring = polygon.GetGeometryRef(i)
            line = get_line_from_linearring(ring)
            if line != []:
                linelist.append(line)
    if len(linelist) == 1:
        return (linelist,None)
    else:
        return (linelist[0:1],linelist[1:])

def get_lines_from_multipolygon(mpolygon):
    """
    Get lines from MultiPolygon object.

    Returns:
    --------
    """
    geometry_name = mpolygon.GetGeometryName()
    polygon_num = mpolygon.GetGeometryCount()
    if geometry_name != 'MULTIPOLYGON':
        raise TypeError("the type is {0}".format(geometry_name))
    else:
        out_ring_list,inner_ring_list = [],[]
        for i in range(polygon_num):
            polygon = mpolygon.GetGeometryRef(i)
            (out_ring,inner_ring) = get_linelist_from_polygon(polygon)
            for subring in out_ring:
                out_ring_list.append(subring)
            if inner_ring is not None:
                if len(inner_ring) == 1:
                    inner_ring_list.append(inner_ring[0])
                else:
                    for subring in inner_ring:
                        inner_ring_list.append(subring)
            else:
                pass
    if inner_ring_list == []:
        return (out_ring_list,None)
    else:
        return (out_ring_list,inner_ring_list)


def get_geometry_from_feature(feature):
    """
    Get geometry from feature.
    """
    georef = feature.GetGeometryRef()
    geometry_name = georef.GetGeometryName()
    if geometry_name == 'POLYGON':
        return get_linelist_from_polygon(georef)
    elif geometry_name == 'MULTIPOLYGON':
        return get_lines_from_multipolygon(georef)
    else:
        raise TypeError("input feature type is {0}".format(geometry_name))


def transform_layer_geometry_to_ring_dataframe(layer,feature_range=None):
    data_list = []
    index_list = []

    if feature_range is None:
        select_range = range(layer.GetFeatureCount())
    else:
        select_range = feature_range

    for i in select_range:
        feature = layer.GetFeature(i)
        out_ring_list,inner_ring_list = get_geometry_from_feature(feature)
        data_list.append({'out_ring':out_ring_list, 'inner_ring':inner_ring_list})
        index_list.append(i)
    return pa.DataFrame(data_list,index=index_list)

def dataframe_of_ring_change_projection(df,m):
    _check_df_inner_out_ring_validity(df)
    dfnew = df.copy()
    for name in ['inner_ring','out_ring']:
        for i in dfnew.index:
            if dfnew[name][i] is None:
                pass
            else:
                ddt = dfnew[name][i]
                dfnew[name][i] = map(lambda templist:map(lambda x:m(*x),templist),ddt)
    return dfnew

def group_dataframe_of_ring(df,groupby):
    """
    group the inner_ring,out_ring dataframe by key.
    """
    _check_df_inner_out_ring_validity(df)
    grp = df.groupby(groupby)

    def merge_list(inlist):
        outlist = []
        for first_level_list in inlist:
            if first_level_list is None:
                pass
            else:
                for sublist in first_level_list:
                    outlist.append(sublist)
        return outlist

    dfdic = {}
    for name in ['inner_ring','out_ring']:
        dfdic[name] = grp[name].apply(merge_list)
    return pa.DataFrame(dfdic)


def get_geometry_type_from_feature(feature):
    georef = feature.GetGeometryRef()
    geometry_name = georef.GetGeometryName()
    return geometry_name

def get_geometry_count_from_feature(feature):
    georef = feature.GetGeometryRef()
    geometry_count = georef.GetGeometryCount()
    return geometry_count

def Add_Linearring_to_Axes(ax,ring,facecolor='0.7',edgecolor='k',
                           transfunc=None,
                           **kwargs):
    """
    Add Linearring to Axes.

    Parameters:
    -----------
    transfunc: functions used for spatial transformation, they should receive
        tuple as parameter and return tuple.
    """
    if transfunc is None:
        ringnew = ring
    else:
        ringnew = [transfunc(t) for t in ring]

    poly = mat.collections.PolyCollection([ringnew],
                                          facecolor=facecolor,
                                          edgecolor=edgecolor,
                                          **kwargs)
    ax.add_collection(poly)

def Add_Polygon_to_Axes(ax,list_of_rings,facecolor='0.7',edgecolor='k',
                        inner_ring_facecolor='w',inner_ring_edgecolor='k',
                        inner_ring_kwargs={},
                        transfunc=None,
                        **kwargs):
    if len(list_of_rings) == 0:
        raise ValueError("input list_of_rings has length 0")
    elif len(list_of_rings) == 1:
        Add_Linearring_to_Axes(ax,list_of_rings[0],facecolor=facecolor,
                               edgecolor=edgecolor,
                               transfunc=transfunc,**kwargs)
    else:
        Add_Linearring_to_Axes(ax,list_of_rings[0],facecolor=facecolor,
                               edgecolor=edgecolor,
                               transfunc=transfunc,
                               **kwargs)
        for ring in list_of_rings[1:]:
            Add_Linearring_to_Axes(ax,ring,facecolor=inner_ring_facecolor,
                                   edgecolor=inner_ring_edgecolor,
                                   transfunc=transfunc,
                                   **inner_ring_kwargs)

def Add_MultiPolygon_to_Axes(ax,list_of_polygon,
                             facecolor='0.7',
                             edgecolor='k',
                             inner_ring_facecolor='w',
                             inner_ring_edgecolor='k',
                             inner_ring_kwargs={},
                             transfunc=None,
                             **kwargs):
    if len(list_of_polygon) == 0:
        raise ValueError("input list_of_polygon has length 0")
    else:
        for list_of_rings in list_of_polygon:
            Add_Polygon_to_Axes(ax,list_of_rings,
                                facecolor=facecolor,
                                edgecolor=edgecolor,
                                inner_ring_facecolor=inner_ring_facecolor,
                                inner_ring_edgecolor=inner_ring_edgecolor,
                                inner_ring_kwargs={},
                                transfunc=transfunc,
                                **kwargs)

def Add_Feature_to_Axes_Polygon(ax,feature,transfunc=None):
    geotype = get_geometry_type_from_feature(feature)
    geometry_vertices = get_geometry_from_feature(feature)
    if geotype == 'POLYGON':
        Add_Polygon_to_Axes(ax,geometry_vertices,inner_ring_facecolor='w',
                            transfunc=transfunc)
    elif geotype == 'MULTIPOLYGON':
        Add_MultiPolygon_to_Axes(ax,geometry_vertices,inner_ring_facecolor='w',
                                 transfunc=transfunc)
    else:
        raise ValueError("geometry type not polygon!")



def dataframe_column_from_array_by_geoindex(geoindex,arrdic):
    """
    Create dataframe from a dict of arrays using the geoindex input. The
        array value that correspond to a geoindex slice will be used
        to fill the column values.

    Parameters:
    -----------
    geoindex: an array (or iterable) containing tuples as its member,
        the tuples will be used to retrieve values from arrays.
    arrdic: a dict of arrays. The dict keys will be used as the
        output dataframe column names.
    """
    dic = {}
    for colname,arr in arrdic.items():
        if not np.ma.isMA(arr):
            dic[colname] = [arr[sl] for sl in geoindex]
        else:
            dic[colname] = [arr[sl] if arr.mask[sl] != True else None for sl in geoindex]
    return pa.DataFrame(dic,index=geoindex)

def dataframe_build_geoindex_from_lat_lon(df,lat_name='lat',
                                          lon_name='lon',
                                          lat=None,lon=None):
    """
    Build a geoindex column for the dataframe "df", by check each
        latitude/longitude pairs (lat_name/lon_name) falling in which
        grid cell of the grid as specified by the vectors of lat/lon.
        The latitude/longitude pairs falling outside the grid will
        have geoindex values as np.nan.

    Returns:
    --------
    A copy of input dataframe with in geoindex being added.

    Parameters:
    -----------
    df: input dataframe.
    lat_name/lon_name: the latitude/longitude field name of the dataframe.
    lat/lon: the latitude/longitude vectors used to compose the grid.
    """
    dft = df.copy()
    dft['geoindex'] = [(None,None)]*len(dft.index)
    for i in dft.index:
        vlat = dft[lat_name][i]
        vlon = dft[lon_name][i]
        try:
            dft['geoindex'][i] = gnc.find_index_by_point(lat,lon,(vlat,vlon))
        except ValueError:
            dft['geoindex'][i] = np.nan
    return dft

def mdata_by_geoindex_dataframe(df,shape=None,mask=None,empty_value=np.nan,
                                lat=None,lon=None):
    """
    Transfer the geoindexed dataframe into Pdata.Mdata for plotting
        or writing out ot NetCDF file.

    Parameters:
    ----------
    lat/lon: the lat/lon used for Mdata.
    shape: the shape of array to be constructed, limited to 2D array. Will be
        automatically derived if lat/lon is given.
    mask: the mask that's to be applied.
    empty_value: the value used to fill the empty gridcells, i.e., gridcells
        that do not appear in geoindex/index column.

    Notes:
    ------
    1. the df.index must be tuples.
    """
    if shape is None:
        if lat is None or lon is None:
            raise ValueError('shape must be provided if lat/lon is not provided')
        else:
            shape = (len(lat),len(lon))
    ydic = {}
    for name in df.columns.tolist():
        data = np.ones(shape)*empty_value
        for index,value in df[name].iterkv():
            if not isinstance(index,tuple):
                raise TypeError("index {0} not tuple".format(index))
            else:
                data[index]=value
        if mask is not None:
            data = np.ma.masked_array(data,mask=mask)
        ydic[name] = data
    return Pdata.Mdata.from_dict_of_array(ydic,lat=lat,lon=lon)


def HDF4_gdalopen(filename,layerlist,namelist):
    """
    To extract the hdf4 file layers into dictionary by using the gdal engine.

    Parameters:
    -----------
    layerlist: the layers to extract from the hdf4 file.
    namelist: the corresonding names for each layer.
    """
    if len(layerlist) != len(namelist):
        raise ValueError("Length of layerlist and namelist not equal")
    else:
        filename_list = ['HDF4_SDS:UNKNOWN:"{0}":{1}'.format(filename,num) for num in layerlist]
        dic = OrderedDict()
        for i,name in enumerate(namelist):
            dataset = gdal.Open(filename_list[i])
            data = dataset.ReadAsArray()
            dic[name] = data
            dataset = None
        return dic


