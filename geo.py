#!/usr/bin/env python

try:
    from osgeo import ogr
    from osgeo import gdal
except ImportError:
    print "osgeo not installed, ogr and gdal not imported!"

import matplotlib as mat
import pandas as pa
import numpy as np
import Pdata
import gnc
from collections import OrderedDict
import scipy as sp

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

#  Structure of shapefile:
#  shapefile > layer > feature (Polygon, MultiPolygon)
#    > rings/lines > linearring/line > vertex

def get_vertices_from_linearring(ring):
    """
    Get the list of vertices from the linearring. A LINEARRING is the geometry
    that comprises POLYGON or MULTIPOLYGON.

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
        vertices = [(a,b) for (a,b,c) in d]
        return vertices


def get_linelist_from_polygon(polygon):
    """
    Get the line list from a polygon object. A POLYGON is the geometry that
    comprises the Feature object.

    Returns:
    --------
    A list of sublist, each sublist is a list of vertices (2-len tuple) that
        comprise a linearring object.
    """
    geometry_name = polygon.GetGeometryName()
    if geometry_name != 'POLYGON':
        raise TypeError("the type is {0}".format(geometry_name))
    else:
        geocount = polygon.GetGeometryCount()
        linelist = []
        for i in range(geocount):
            ring = polygon.GetGeometryRef(i)
            line = get_vertices_from_linearring(ring)
            if line != []:
                linelist.append(line)
    return linelist


def add_one_linearring_to_axes(ax,ring,facecolor='0.7',edgecolor='k',
                               transfunc=None,
                               **kwargs):
    """
    Add ONE Linearring to Axes. a Linearring is one enclosed line, with its
    vertices being linearly connected to form an enclosed circle.

    Parameters:
    -----------
    ring: An enclosed line, provided as a list of 2-len tuples. Note this is
        different as the `verts` in mat.collections.PolyCollection. `verts` there
        is a list of rings or enclosed lines.
    transfunc: functions used for spatial transformation, they should receive
        tuple as parameter and return tuple.

    Notes:
    ------
    Actually mat.collections.PolyCollection could create more than one
        enclosed circles (polygon) in just a single call, but here we
        separate this function in order to set different colors for inner
        and outer circles.
    """
    if transfunc is None:
        ringnew = ring
    else:
        ringnew = [transfunc(t) for t in ring]

    # Note here we need to put ringnew in [] to conform to the `verts`
    # in the function of mat.collections.PolyCollection
    poly = mat.collections.PolyCollection([ringnew],
                                          facecolor=facecolor,
                                          edgecolor=edgecolor,
                                          **kwargs)
    ax.add_collection(poly)

def Add_Polygon_to_Axes(ax,linelist,
                        outer_ring_facecolor='0.7',outer_ring_edgecolor='k',
                        outer_ring_kwargs={},
                        inner_ring_facecolor='w',inner_ring_edgecolor='k',
                        inner_ring_kwargs={},
                        transfunc=None):
    """
    Notes:
    ------
    A polygon can have one or more lines, with each line shown as an eclosed
    circle by using mat.collections.PolyCollection. If one POLYGON has
    more than one ring, we treat the first one as outer circle, others as
    inner circles.
    """
    if len(linelist) == 0:
        raise ValueError("input linelist has length 0")
    else:
        add_one_linearring_to_axes(ax,linelist[0],
                               facecolor=outer_ring_facecolor,
                               edgecolor=outer_ring_edgecolor,
                               transfunc=transfunc,**outer_ring_kwargs)
        if len(linelist) > 1:
            for ring in linelist[1:]:
                add_one_linearring_to_axes(ax,ring,
                                       facecolor=inner_ring_facecolor,
                                       edgecolor=inner_ring_edgecolor,
                                       transfunc=transfunc,
                                       **inner_ring_kwargs)

def get_polygon_list_from_multipolygon(mpolygon):
    """
    Get polygon list from a MultiPolygon object.

    Returns:
    --------
    a polygon list, i.e., a list of lines, which is agian a nested list,
        whose member is a list of 2-len tuples.
    """
    geometry_name = mpolygon.GetGeometryName()
    polygon_num = mpolygon.GetGeometryCount()
    if geometry_name != 'MULTIPOLYGON':
        raise TypeError("the type is {0}".format(geometry_name))
    else:
        polygon_list = []
        for i in range(polygon_num):
            polygon = mpolygon.GetGeometryRef(i)
            linelist = get_linelist_from_polygon(polygon)
            polygon_list.append(linelist)

    return polygon_list

def Add_MultiPolygon_to_Axes(ax,polygon_list,
                             outer_ring_facecolor='0.7',
                             outer_ring_edgecolor='k',
                             outer_ring_kwargs={},
                             inner_ring_facecolor='w',
                             inner_ring_edgecolor='k',
                             inner_ring_kwargs={},
                             transfunc=None):
    """
    Parameters:
    -----------
    polygon_list: a nested list, i.e., list of lines. Note that lines are
    a list of line, which is a list of 2-len tuples.
    """
    if len(polygon_list) == 0:
        raise ValueError("input polygon_list has length 0")
    else:
        for list_of_rings in polygon_list:
            Add_Polygon_to_Axes(ax,list_of_rings,
                                outer_ring_facecolor=outer_ring_facecolor,
                                outer_ring_edgecolor=outer_ring_edgecolor,
                                outer_ring_kwargs={},
                                inner_ring_facecolor=inner_ring_facecolor,
                                inner_ring_edgecolor=inner_ring_edgecolor,
                                inner_ring_kwargs={},
                                transfunc=transfunc,
                                **kwargs)

def Add_Feature_to_Axes(ax,feature,
                        outer_ring_facecolor='0.7',
                        outer_ring_edgecolor='k',
                        outer_ring_kwargs={},
                        inner_ring_facecolor='w',
                        inner_ring_edgecolor='k',
                        inner_ring_kwargs={},
                        transfunc=None):

    georef = feature.GetGeometryRef()
    geometry_name = georef.GetGeometryName()


    if geometry_name == 'POLYGON':
        linelist = get_linelist_from_polygon(georef)
        Add_Polygon_to_Axes(ax,linelist,
                            outer_ring_facecolor=outer_ring_facecolor,
                            outer_ring_edgecolor=outer_ring_edgecolor,
                            outer_ring_kwargs=outer_ring_kwargs,
                            inner_ring_facecolor=inner_ring_facecolor,
                            inner_ring_edgecolor=inner_ring_edgecolor,
                            inner_ring_kwargs=inner_ring_kwargs,
                            transfunc=transfunc)

    elif geometry_name == 'MULTIPOLYGON':
        polygon_list = get_polygon_list_from_multipolygon(georef)
        Add_MultiPolygon_to_Axes(ax,polygon_list,
                                 outer_ring_facecolor=outer_ring_facecolor,
                                 outer_ring_edgecolor=outer_ring_edgecolor,
                                 outer_ring_kwargs=outer_ring_kwargs,
                                 inner_ring_facecolor=inner_ring_facecolor,
                                 inner_ring_edgecolor=inner_ring_edgecolor,
                                 inner_ring_kwargs=inner_ring_kwargs,
                                 transfunc=transfunc)
    else:
        raise ValueError("geometry type not polygon!")


def Add_Layer_to_Axes(ax,layer,
                      outer_ring_facecolor='0.7',
                      outer_ring_edgecolor='k',
                      outer_ring_kwargs={},
                      inner_ring_facecolor='w',
                      inner_ring_edgecolor='k',
                      inner_ring_kwargs={},
                      transfunc=None):

    feature_count = layer.GetFeatureCount()
    for i in range(feature_count):
        feature = layer.GetFeature(i)
        Add_Feature_to_Axes(ax,feature,
                            outer_ring_facecolor=outer_ring_facecolor,
                            outer_ring_edgecolor=outer_ring_edgecolor,
                            outer_ring_kwargs=outer_ring_kwargs,
                            inner_ring_facecolor=inner_ring_facecolor,
                            inner_ring_edgecolor=inner_ring_edgecolor,
                            inner_ring_kwargs=inner_ring_kwargs,
                            transfunc=transfunc)


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
    dft = pa.DataFrame(dic,index=np.arange(len(geoindex)))
    dft.insert(0,'geoindex',geoindex)
    return dft

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

def dataframe_build_geoindex_from_lat_lon_sp(df,lat_name='lat',
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

    Notes:
    ------
    This used a scipy function and is thus faster.
    """
    dft = df.copy()
    longrid,latgrid = np.meshgrid(lon,lat)
    grids = np.vstack([longrid.ravel(),latgrid.ravel()]).transpose()
    tree = sp.spatial.cKDTree(grids)
    points = np.vstack([df[lon_name].values,df[lat_name].values]).transpose()
    dist, indexes = tree.query(points)
    tindex = [np.unravel_index(num,latgrid.shape) for num in indexes]
    dft['geoindex'] = tindex
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



from math import radians, cos, sin, asin, sqrt
def distance_haversine(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees). Return distance in km.

    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def distance_haversine_dataframe(dft):
    """
    Calculate the cross distances of points given by a dataframe. The input
        dataframe must be 'lat','lon' as its columns.
    """
    adic = OrderedDict()
    for reg1 in dft.index.tolist():
        lat1,lon1 = dft.ix[reg1]['lat'],dft.ix[reg1]['lon']
        dic = OrderedDict()
        for reg2 in dft.index.tolist():
            lat2,lon2 = dft.ix[reg2]['lat'],dft.ix[reg2]['lon']
            dic[reg2] = distance_haversine(lon1,lat1,lon2,lat2)
        adic[reg1] = pa.Series(dic)

    return pa.DataFrame(adic)



