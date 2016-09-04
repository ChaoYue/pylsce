#!/usr/bin/env python

import numpy as np
import pandas as pa
import pb
from glccIndexTree6Herb4 import *
from collections import OrderedDict

def calc_area_from_LCCmatrix(veg5ini,list_LCCmatrix,area):
    """
    Caculate the time series of area from initial vegetation
        array (veg5ini) and a list of LCCmatrix.
    
    Parameters:
    -----------
    veg5ini: Initial vegetation fraction array, in the sequnce
        of baresoil, forest, grass, pasture, crop.
    list_LCCmatrix: list of LCC matrix, each matrix having 
        12 as the length of first dimension. In the sequence 
        of:
            f2g=1-1; f2p=2-1; f2c=3-1; 
            g2f=4-1; g2p=5-1; g2c=6-1; 
            p2f=7-1; p2g=8-1; p2c=9-1; 
            c2f=10-1; c2g=11-1; c2p=12-1
    area: area used to calcuate from the fraction to absolute area
    """

    #Initial arrays of veg fraction
    veget_1500 = veg5ini.copy()
    forest = veget_1500[1]
    grass = veget_1500[2]
    pasture = veget_1500[3]
    crop = veget_1500[4]

    list_forest = []
    list_grass = []
    list_pasture = []
    list_crop = []

    list_forest.append(forest)
    list_grass.append(grass)
    list_pasture.append(pasture)
    list_crop.append(crop)

    #indices
    f2g=1-1; f2p=2-1; f2c=3-1; g2f=4-1; g2p=5-1; g2c=6-1; p2f=7-1; p2g=8-1; p2c=9-1; c2f=10-1; c2g=11-1; c2p=12-1

    for ind in range(len(list_LCCmatrix)):
        arr = list_LCCmatrix[ind]

        forest_new = forest - arr[f2g] - arr[f2p] - arr[f2c] + arr[g2f] + arr[p2f] + arr[c2f]
        grass_new = grass - arr[g2f] - arr[g2p] - arr[g2c] + arr[f2g] + arr[p2g] + arr[c2g]
        pasture_new = pasture - arr[p2f] - arr[p2g] - arr[p2c] + arr[f2p] + arr[g2p] + arr[c2p]
        crop_new = crop - arr[c2f] - arr[c2p] - arr[c2g] + arr[f2c] + arr[p2c] + arr[g2c]

        list_forest.append(forest_new)
        list_grass.append(grass_new)
        list_pasture.append(pasture_new)
        list_crop.append(crop_new)

        forest = forest_new.copy()
        grass = grass_new.copy()
        pasture = pasture_new.copy()
        crop = crop_new.copy()

    baresoil = np.tile(veget_1500[0],(len(list_crop),1,1))
    series_arr = map(lambda list_arr:np.rollaxis(np.ma.dstack(list_arr),2,0),[list_forest,list_grass,list_pasture,list_crop])

    veg5type = dict(zip(['bareland', 'forest', 'grass', 'pasture', 'crop'],[baresoil] + series_arr))
    dic = pb.Dic_Apply_Func(lambda x:np.ma.sum(x*area,axis=(1,2)),veg5type)
    dft = pa.DataFrame(dic)
    return dft

glccmatrix_string = ['f2g','f2p','f2c','g2f','g2p','g2c','p2f','p2g','p2c','c2f','c2g','c2p']

f2g=1-1; f2p=2-1; f2c=3 -1; g2f=4-1; g2p=5-1; g2c=6-1; p2f=7-1; p2g=8-1; p2c=9-1; c2f=10-1; c2g=11-1; c2p=12-1

def lccm_SinglePoint_to_4X4matrix(lcc_matrix):
    """
    Convert the 12-length lcc_matrix to 4x4 matrix, return a dataframe:
    Column = recieving; Row = giving
        f   g   p   c
    f   -   -  f2p f2c
    g  g2f  -  g2p g2c
    p  p2f p2g  -  p2c
    c  c2f c2g c2p  -
    """
    f2g=1-1; f2p=2-1; f2c=3 -1; g2f=4-1; g2p=5-1; g2c=6-1; p2f=7-1; p2g=8-1; p2c=9-1; c2f=10-1; c2g=11-1; c2p=12-1
    arr = np.zeros((4,4))
    arr[0,1] = lcc_matrix[f2g]
    arr[0,2] = lcc_matrix[f2p]
    arr[0,3] = lcc_matrix[f2c]
    #arr[0,0] = 1-arr[0,1:].sum()
    
    arr[1,0] = lcc_matrix[g2f]
    arr[1,2] = lcc_matrix[g2p]
    arr[1,3] = lcc_matrix[g2c]
    #arr[1,1] = 1-lcc_matrix[[g2f,g2p,g2c]].sum()

    arr[2,0] = lcc_matrix[p2f]
    arr[2,1] = lcc_matrix[p2g]
    arr[2,3] = lcc_matrix[p2c]
    #arr[2,2] = 1-lcc_matrix[[p2f,p2g,p2c]].sum()
    
    arr[3,0] = lcc_matrix[c2f]
    arr[3,1] = lcc_matrix[c2g]
    arr[3,2] = lcc_matrix[c2p]
    #arr[3,3] = 1-lcc_matrix[[c2f,c2g,c2p]].sum()
    
    dft = pa.DataFrame(arr,columns=['f','g','p','c'],index=['f','g','p','c'])
    return dft

def vegmax_SinglePoint_to_veg4type(vegmax):
    """
    Convert a single-point 65-length veget_max into fractions of
    [f,g,p,c]
    """
    f = vegmax[ia2_1:ia9_6+1].sum()
    g = vegmax[ia10_1:ia10_4+1].sum() + vegmax[ia12_1:ia12_4+1].sum()
    p = vegmax[ia11_1:ia11_4+1].sum() + vegmax[ia13_1:ia13_4+1].sum()
    c = vegmax[ia14_1:ia15_4+1].sum()
    return np.array([f,g,p,c])

def glccpftmtc_SinglePoint_to_4X4matrix(glcc_pftmtc):
    """
    Convert the glcc_pftmtc(65x15) to 4x4 matrix, in a dataframe:
    Column = recieving; Row = giving
        f   g   p   c
    f   -   -  f2p f2c
    g  g2f  -  g2p g2c
    p  p2f p2g  -  p2c
    c  c2f c2g c2p  -
    """
    glccReal_from_glccpftmtc = np.zeros((4,4))

    glccReal_from_glccpftmtc[0,0] = glcc_pftmtc[ia2_1:ia9_6+1,1:9].sum()
    glccReal_from_glccpftmtc[0,1] = glcc_pftmtc[ia2_1:ia9_6+1,[9,11]].sum()
    glccReal_from_glccpftmtc[0,2] = glcc_pftmtc[ia2_1:ia9_6+1,[10,12]].sum()
    glccReal_from_glccpftmtc[0,3] = glcc_pftmtc[ia2_1:ia9_6+1,[13,14]].sum()

    glccReal_from_glccpftmtc[1,0] = glcc_pftmtc[ia10_1:ia10_4+1,1:9].sum() + glcc_pftmtc[ia12_1:ia12_4+1,1:9].sum()
    glccReal_from_glccpftmtc[1,2] = glcc_pftmtc[ia10_1:ia10_4+1,[10,12]].sum() + glcc_pftmtc[ia12_1:ia12_4+1,[10,12]].sum()
    glccReal_from_glccpftmtc[1,3] = glcc_pftmtc[ia10_1:ia10_4+1,[13,14]].sum() + glcc_pftmtc[ia12_1:ia12_4+1,[13,14]].sum()

    glccReal_from_glccpftmtc[2,0] = glcc_pftmtc[ia11_1:ia11_4+1,1:9].sum() + glcc_pftmtc[ia13_1:ia13_4+1,1:9].sum()
    glccReal_from_glccpftmtc[2,1] = glcc_pftmtc[ia11_1:ia11_4+1,[9,11]].sum() + glcc_pftmtc[ia13_1:ia13_4+1,[9,11]].sum()
    glccReal_from_glccpftmtc[2,3] = glcc_pftmtc[ia11_1:ia11_4+1,[13,14]].sum() + glcc_pftmtc[ia13_1:ia13_4+1,[13,14]].sum()

    glccReal_from_glccpftmtc[3,0] = glcc_pftmtc[ia14_1:ia15_4+1,1:9].sum()
    glccReal_from_glccpftmtc[3,1] = glcc_pftmtc[ia14_1:ia15_4+1,[9,11]].sum()
    glccReal_from_glccpftmtc[3,2] = glcc_pftmtc[ia14_1:ia15_4+1,[10,12]].sum()

    dft = pa.DataFrame(glccReal_from_glccpftmtc,columns=['f','g','p','c'],index=['f','g','p','c'])

    return dft


def get_age_fraction_vegetmax(vegmax):
    """
    Return a dictionary of fractions of different age classes
        for Bareland,Forest,Grass,Pasture and Crop.

    Parameters:
    ----------
    vegmax: The 65-length PFT should be the second dimension
    """
    frac_F = OrderedDict()

    frac_F['Bareland'] = vegmax[:,0:1,...].sum(axis=1)

    frac_F['Forest_Age_1'] = vegmax[:,np.arange(ia2_1,ia9_6,6),...].sum(axis=1)
    frac_F['Forest_Age_2'] = vegmax[:,np.arange(ia2_2,ia9_6,6),...].sum(axis=1)
    frac_F['Forest_Age_3'] = vegmax[:,np.arange(ia2_3,ia9_6,6),...].sum(axis=1)
    frac_F['Forest_Age_4'] = vegmax[:,np.arange(ia2_4,ia9_6,6),...].sum(axis=1)
    frac_F['Forest_Age_5'] = vegmax[:,np.arange(ia2_5,ia9_6,6),...].sum(axis=1)
    frac_F['Forest_Age_6'] = vegmax[:,np.arange(ia2_6,ia9_6+1,6),...].sum(axis=1)

    frac_F['Grassland_Age_1'] = vegmax[:,[ia10_1,ia12_1],...].sum(axis=1)
    frac_F['Grassland_Age_2'] = vegmax[:,[ia10_2,ia12_2],...].sum(axis=1)
    frac_F['Grassland_Age_3'] = vegmax[:,[ia10_3,ia12_3],...].sum(axis=1)
    frac_F['Grassland_Age_4'] = vegmax[:,[ia10_4,ia12_4],...].sum(axis=1)

    frac_F['Pasture_Age_1'] = vegmax[:,[ia11_1,ia13_1],...].sum(axis=1)
    frac_F['Pasture_Age_2'] = vegmax[:,[ia11_2,ia13_2],...].sum(axis=1)
    frac_F['Pasture_Age_3'] = vegmax[:,[ia11_3,ia13_3],...].sum(axis=1)
    frac_F['Pasture_Age_4'] = vegmax[:,[ia11_4,ia13_4],...].sum(axis=1)

    frac_F['Crop_Age_1'] = vegmax[:,[ia14_1,ia15_1],...].sum(axis=1)
    frac_F['Crop_Age_2'] = vegmax[:,[ia14_2,ia15_2],...].sum(axis=1)
    frac_F['Crop_Age_3'] = vegmax[:,[ia14_3,ia15_3],...].sum(axis=1)
    frac_F['Crop_Age_4'] = vegmax[:,[ia14_4,ia15_4],...].sum(axis=1)

    return frac_F

def get_veget_fraction_vegetmax(vegmax):
    """
    Return a dictionary of fractions of bareland,forest,grass,pasture and crop.
        This function calls the function "get_age_fraction_vegetmax" and sum
        together the fractions of different age classes.

    Parameters:
    ----------
    vegmax: The 65-length PFT should be the second dimension
    """
    dic = get_age_fraction_vegetmax(vegmax)
    newdic = OrderedDict()
    newdic['bareland'] = dic['Bareland']
    newdic['forest'] = dic['Forest_Age_1'] + dic['Forest_Age_2'] + dic['Forest_Age_3'] + dic['Forest_Age_4'] + dic['Forest_Age_5'] \
                       + dic['Forest_Age_6']
    newdic['grass'] = dic['Grassland_Age_1'] + dic['Grassland_Age_2'] + dic['Grassland_Age_3'] + dic['Grassland_Age_4']
    newdic['pasture'] = dic['Pasture_Age_1'] + dic['Pasture_Age_2'] + dic['Pasture_Age_3'] + dic['Pasture_Age_4']
    newdic['crop'] = dic['Crop_Age_1'] + dic['Crop_Age_2'] + dic['Crop_Age_3'] + dic['Crop_Age_4']
    return newdic








