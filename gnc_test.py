#!/usr/bin/env python

import gnc
import os
import g
import pdb
import sys
import numpy as np
import Pdata
#from Pdata import Pdata
import unittest
import pb
from nose import tools as ntools
import __builtin__


wd = os.getenv('PYDIR')

def open_Ncdata():
    """
    """
    d = gnc.Ncdata(wd+'testdata/stomate_history_TOTAL_M_LITTER_CONSUMP_4dim.nc')
    return d

class test_Add_Vars_to_Dict_by_RegSum():
    """
    test Add_Vars_to_Dict_by_RegSum
    """
    d = open_Ncdata()
    grid = (35,70,60,110) #(lat1,lon1,lat2,lon2)
    area = d.d1.Areas
    gird = None
    litter_consump = d.d1.LITTER_CONSUMP


    def test_grid_None(self):

        d = open_Ncdata()
        area = d.d1.Areas
        litter_consump = d.d1.LITTER_CONSUMP
        gridcell_num = litter_consump.shape[-2] * litter_consump.shape[-1]

        #testing sum; area_weight as None
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='sum')
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump,axis=-1),axis=-1),rtol=1E-5)
        #testing mean; area_weight as None
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='mean')
            #we set rtol as high as the mean values are all small ones.
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump,axis=-1),axis=-1)/gridcell_num,rtol=1E-3)

        #testing sum; area_weight as True
        litter_consump_area = area*litter_consump
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='sum',area_weight=True)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump_area,axis=-1),axis=-1))
        #testing sum; area_weight as True
        total_area = np.ma.sum(area)
        litter_consump_area = area*litter_consump
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='mean',area_weight=True)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump_area,axis=-1),axis=-1)/total_area,rtol=1E-3)

    def test_grid(self):

        d = open_Ncdata()
        grid = (35,70,60,110) #(lat1,lon1,lat2,lon2)
        (vlat1,vlat2) = (35,60)
        (vlon1,vlon2) = (70,110)
        area = d.Get_GridValue('Areas',(vlat1, vlat2), (vlon1, vlon2))
        litter_consump = d.Get_GridValue('LITTER_CONSUMP',(vlat1, vlat2), (vlon1, vlon2))
        gridcell_num = litter_consump.shape[-2] * litter_consump.shape[-1]

        #testing sum; area_weight as None
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='sum',grid=grid)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump,axis=-1),axis=-1),rtol=1E-5)
        #testing mean; area_weight as None
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='mean',grid=grid)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump,axis=-1),axis=-1)/gridcell_num,rtol=1E-3)

        #testing sum; area_weight as True
        litter_consump_area = area*litter_consump
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='sum',area_weight=True,grid=grid)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump_area,axis=-1),axis=-1))
        #testing sum; area_weight as True
        total_area = np.ma.sum(area)
        litter_consump_area = area*litter_consump
        litcon1 = d.Add_Vars_to_Dict_by_RegSum(['LITTER_CONSUMP'],mode='mean',area_weight=True,grid=grid)
        np.testing.assert_allclose(litcon1['LITTER_CONSUMP'],np.ma.sum(np.ma.sum(litter_consump_area,axis=-1),axis=-1)/total_area,rtol=1E-5)



class test_find_index_by_point():
    lat = np.arange(4.75,0,-0.5)
    lon = np.arange(0.25,5,0.5)
    @ntools.raises(ValueError)
    def test_decrease_lat(self):
        """
        increasing lat input should raise ValueError.
        """
        lat_increase = np.arange(0.25,5,0.5)
        lon = np.arange(0.25,5,0.5)
        gnc.find_index_by_point(lat_increase,lon,(2,2))
    @ntools.raises(ValueError)
    def test_big_value_lat(self):
        """
        vlat/vlon bigger than range should raise ValueError.
        """
        gnc.find_index_by_point(self.lat,self.lon,(5.01,2))

    @ntools.raises(ValueError)
    def test_small_value_lat(self):
        """
        vlat/vlon smaller than range should raise ValueError.
        """
        gnc.find_index_by_point(self.lat,self.lon,(-0.01,2))

    @ntools.raises(ValueError)
    def test_big_value_lon(self):
        """
        vlat/vlon bigger than range should raise ValueError.
        """
        gnc.find_index_by_point(self.lat,self.lon,(2,5.01))

    @ntools.raises(ValueError)
    def test_small_value_lon(self):
        """
        vlat/vlon smaller than range should raise ValueError.
        """
        gnc.find_index_by_point(self.lat,self.lon,(2,-0.01))

    def test_return_correct_value(self):
        ntools.assert_tuple_equal((0,0),gnc.find_index_by_point(self.lat,self.lon,(5,0)))
        ntools.assert_tuple_equal((9,0),gnc.find_index_by_point(self.lat,self.lon,(0,0)))
        ntools.assert_tuple_equal((9,9),gnc.find_index_by_point(self.lat,self.lon,(0,5)))
        ntools.assert_tuple_equal((0,9),gnc.find_index_by_point(self.lat,self.lon,(5,5)))
        ntools.assert_tuple_equal((2,7),gnc.find_index_by_point(self.lat,self.lon,(3.99,3.99)))







