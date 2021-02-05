#!/usr/bin/env python

import pb
import numpy as np
from collections import OrderedDict


#PFTs in ORCHIDEE
#    1 - Bare soil
#    2 - tropical  broad-leaved evergreen
#    3 - tropical  broad-leaved raingreen
#    4 - temperate needleleaf   evergreen
#    5 - temperate broad-leaved evergreen
#    6 - temperate broad-leaved summergreen
#    7 - boreal    needleleaf   evergreen
#    8 - boreal    broad-leaved summergreen
#    9 - boreal    needleleaf   summergreen
#   10 -           C3           grass
#   11 -           C4           grass
#   12 -           C3           agriculture
#   13 -           C4           agriculture

pftdic= \
{\
1: 'Bare soil',\
2: 'tropical broad-leaved evergreen',\
3: 'tropical broad-leaved raingreen',\
4: 'temperate needleleaf evergreen',\
5: 'temperate broad-leaved evergreen',\
6: 'temperate broad-leaved summergreen',\
7: 'boreal needleleaf evergreen',\
8: 'boreal broad-leaved summergreen',\
9: 'boreal needleleaf summergreen',\
10: 'C3 grass',\
11: 'C4 grass',\
12: 'C3 agriculture',\
13: 'C4 agriculture',\
}

pftdic15= \
{\
1: 'Bare soil',\
2: 'tropical broad-leaved evergreen',\
3: 'tropical broad-leaved raingreen',\
4: 'temperate needleleaf evergreen',\
5: 'temperate broad-leaved evergreen',\
6: 'temperate broad-leaved summergreen',\
7: 'boreal needleleaf evergreen',\
8: 'boreal broad-leaved summergreen',\
9: 'boreal needleleaf summergreen',\
10: 'C3 grass',\
11: 'C3 pasture',\
12: 'C4 grass',\
13: 'C4 pasture',\
14: 'C3 agriculture',\
15: 'C4 agriculture'
}


# pft15 in the trunk4
dicpft15_PFTtrunkDefault = OrderedDict()
dicpft15_PFTtrunkDefault[1]='SoilBareGlobal'
dicpft15_PFTtrunkDefault[2]='BroadLeavedEvergreenTropical'
dicpft15_PFTtrunkDefault[3]='BroadLeavedRaingreenTropical'
dicpft15_PFTtrunkDefault[4]='NeedleleafEvergreenTemperate'
dicpft15_PFTtrunkDefault[5]='BroadLeavedEvergreenTemperate'
dicpft15_PFTtrunkDefault[6]='BroadLeavedSummergreenTemperate'
dicpft15_PFTtrunkDefault[7]='NeedleleafEvergreenBoreal'
dicpft15_PFTtrunkDefault[8]='BroadLeavedSummergreenBoreal'
dicpft15_PFTtrunkDefault[9]='LarixSpBoreal'
dicpft15_PFTtrunkDefault[10]='C3GrassTemperate'
dicpft15_PFTtrunkDefault[11]='C4GrassTemperate'
dicpft15_PFTtrunkDefault[12]='C3AgricultureTemperate'
dicpft15_PFTtrunkDefault[13]='C4AgricultureTemperate'
dicpft15_PFTtrunkDefault[14]='C3GrassTropical'
dicpft15_PFTtrunkDefault[15]='C3GrassBoreal'

pftlist = [pftdic[num+1] for num in range(13)]
pftlist15 = [pftdic15[num+1] for num in range(len(pftdic15))]
pftlistgross = ['Bare soil','Tropical forest','Tropical forest','Temperate forest','Temperate forest','Temperate forest','Boreal forest','Boreal forest','Boreal forest','Natural grassland','Natural grassland','Cropland','Cropland']

pftlistgross15 = ['Bare soil','Tropical forest','Tropical forest','Temperate forest','Temperate forest','Temperate forest','Boreal forest','Boreal forest','Boreal forest','Natural grassland','Pasture','Natural grassland','Pasture','Cropland','Cropland']
cal = pb.calendar()
cal.get_month_doy()
index_first_day_month_noleap = cal.index_first_day_month_noleap
index_first_day_month_leap = cal.index_first_day_month_leap

latHD = np.arange(89.75,-90,-0.5)
lonHD = np.arange(-179.75,180,0.5)

class levels(object):
    biomass_gC_m2 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 35000, 40000]
    biomass_kgC_m2 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 35.0, 40.0]
    biomass_tC_ha = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 350, 400]
    litter_gC_m2 = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000]
    BA_percentage = [0.1,1,4,8,12,16,20,30,45,60]
    BA_percentage2 = [0.1,0.5,1,2,5,10,20,30,45,60,80,100]
    FireEmi_gC_m2 = [1,10,20,50,100,200,500,1000,2000]
    GPP_gC_m2 = [0,100,200,400,600,800,1200,1600,2000,2600,3200,3800,4400]
    precip_mm_month = [0,1,10,25,50,75,100,150,200,300,400,1000]
    precip_mm_year = [10,200,400,600,800,1000,1400,1800,2200,2800,3600,4200]
    temperature_Celsuis = [-55,-50,-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45]

vars_biomass = [u'LEAF_M', u'RESERVE_M', u'ROOT_M', u'SAP_M_BE', u'SAP_M_AB',u'HEART_M_BE', u'HEART_M_AB', u'FRUIT_M','TOTAL_M']
vars_litter = [u'LITTER_MET_AB',u'LITTER_MET_BE',u'LITTER_STR_AB',u'LITTER_STR_BE']

vars_balance = ['NBP_npp','GPP','NPP','CO2_TAKEN','LUCsource','CO2_FIRE','HARVEST_ABOVE','HET_RESP','GMsource']
vars_LUCsource = ['CONVFLUX','CFLUX_PROD10','CFLUX_PROD100']
vars_LUC = ['CONVFLUX_LCC','CONVFLUX_HAR','CFLUX_PROD10_LCC','CFLUX_PROD10_HAR','CFLUX_PROD100_LCC','CFLUX_PROD100_HAR']
vars_prod = ['PROD10_LCC','PROD10_HAR','PROD100_LCC','PROD100_HAR']

vars_diagose_fire = ['D_AREA_BURNT','D_NUMFIRE','D_FDI','ROS_F','LITTER_CONSUMP','CROWN_CONSUMP','CO2_FIRE',
                     'mean_fire_size','mean_fire_size_or','char_dens_fuel_ave','dead_fuel','dead_fuel_all']



vars_biomass_c_trunk = [u'LEAF_M_c', u'RESERVE_M_c', u'ROOT_M_c', u'SAP_M_BE_c', u'SAP_M_AB_c',u'HEART_M_BE_c', u'HEART_M_AB_c', u'FRUIT_M_c','LABILE_M_c','TOTAL_M_c']
vars_litter = [u'LITTER_MET_AB',u'LITTER_MET_BE',u'LITTER_STR_AB',u'LITTER_STR_BE']

