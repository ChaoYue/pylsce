#!/usr/bin/env python

import numpy as np
import Pdata
from collections import OrderedDict
import gnc
import pandas as pa



"""
This module stores functions only meaningful to ORCHIDEE
"""

veget4type=['bareland','forest','grass','crop']
veget5type=['bareland','forest','grass','pasture','crop']
forest4type = ['BrEv','BrDe','NeEv','NeDe']
forest4type_longname = ['BrEv','BrDe','NeEv','NeDe']

def VEGET_MAX_index_list(typenum=4):
    """
    Return the np.s_ index to slice the vegetmax related variables
        into different classes.

    Parameters:
    -----------
    typenum:
        4: recast into bareland,forest,grass,and crop.
        5: recast into bareland,forest,grass,pasture,and crop.
        forest: extract only the fractions for forest into four types:
            BrEv,BrDe,NeEv,NeDe
    """
    if typenum == 4:
        index_list = [np.s_[...,0:1,:,:],
                      np.s_[...,1:9,:,:],
                      np.s_[...,9:11,:,:],
                      np.s_[...,11:13,:,:]
                     ]
        namelist = ['bareland','forest','grass','crop']

    elif typenum == 'forest':
        index_list = [np.s_[...,[1,4],:,:],
                      np.s_[...,[2,5,7],:,:],
                      np.s_[...,[3,6],:,:],
                      np.s_[...,[8],:,:]
                     ]
        namelist = ['BrEv','BrDe','NeEv','NeDe']

    elif typenum == 5:
        index_list = [np.s_[...,0:1,:,:],
                      np.s_[...,1:9,:,:],
                      np.s_[...,[9,11],:,:],
                      np.s_[...,[10,12],:,:],
                      np.s_[...,[13,14],:,:]
                     ]
        namelist = ['bareland','forest','grass','pasture','crop']

    else:
        raise ValueError("typenum not equal to 4!")
    return (index_list,namelist)

def VEGET_MAX_recast(arr,typenum=4,return_dic=False):
    """
    change the VEGET_MAX variable into another several types.

    Parameters:
    -----------
    arr: input VEGET_MAX or veget ndarray.
    typenum:
        4: recast into bareland,forest,grass,and agriculture.
        5: recast into bareland,forest,grass,pasture and crop.
        forest: extract only the fractions for forest into four types:
            BrEv,BrDe,NeEv,NeDe
    return_dic: boolean, True to return a dictionary with the keys as the types.
    """
    if arr.ndim < 3:
        raise ValueError("input array dimension less than 3!")
    else:
        if arr.shape[-3] not in [13,15]:
            raise ValueError("the third last dim is not PFT!")
        else:
            index_list,namelist = VEGET_MAX_index_list(typenum)
            arrlist = [np.ma.sum(arr[ind],axis=-3)[...,np.newaxis,:,:]
                        for ind in index_list]
            if not return_dic:
                return np.ma.concatenate(arrlist,axis=-3)
            else:
                arrlist = [arr[...,0,:,:] for arr in arrlist]
                return OrderedDict(zip(namelist,arrlist))


def dataframe_from_stomate(filepattern,largefile=True,multifile=True,
                           dgvmadj=False,spamask=None,
                           veget_npindex=np.s_[:],areaind=np.s_[:],
                           out_timestep='annual',version=1,
                           replace_nan=False):
    """
    Parameters:
    -----------
    filepattern: could be a single filename, or a file pattern
    out_timestep: the timestep of output file, used to provide information
        to properly scale the variable values, could be 'annual' or 'daily'.
    dgvmadj: use DGVM adjustment, in this case tBIOMASS rathern than TOTAL_M
        is used.
    veget_npindex: passed to the function of get_pftsum:
        1. could be used to restrict for example the PFT
        weighted average only among natural PFTs by setting
        veget_npindex=np.s_[:,0:11,:,:]. It will be used to slice
        VEGET_MAX variable.
        2. could also be used to slice only for some subgrid
        of the whole grid, eg., veget_npindex=np.s_[...,140:300,140:290].

    Notes:
    ------
    1. This function could handle automatically the case of a single-point
       file or a regional file. When a single-point file (pattern) is given,
       PFT-weighted carbon density will be used rather than the total C over
       the spatial area.
    """
    gnc_sto = gnc.Ncdata(filepattern,largefile=largefile,multifile=multifile,
                         replace_nan=replace_nan)

    if version == 1:
        # list all pools and fluxes
        list_flux_pft = ['GPP','NPP','HET_RESP','CO2_FIRE','CO2FLUX','CO2_TAKEN']
        list_flux_pftsum = ['CONVFLUX','CFLUX_PROD10','CFLUX_PROD100','HARVEST_ABOVE']
        list_flux = list_flux_pft+list_flux_pftsum

        list_pool = ['TOTAL_M','TOTAL_SOIL_CARB']
        list_all = list_flux_pft+list_flux_pftsum+list_pool
        nlist_var = [list_flux_pft, list_flux_pftsum, list_pool]

        for varlist in nlist_var:
            gnc_sto.retrieve_variables(varlist)
            gnc_sto.get_pftsum(print_info=False,veget_npindex=veget_npindex)
            gnc_sto.remove_variables(varlist)

        #handle adjustment of different variables
        if dgvmadj:
            gnc_sto.retrieve_variables(['tGPP','tRESP_GROWTH','tRESP_MAINT','tRESP_HETERO','tCO2_FIRE'])
            gnc_sto.pftsum.__dict__['NPP'] = gnc_sto.d1.tGPP - gnc_sto.d1.tRESP_MAINT - gnc_sto.d1.tRESP_GROWTH
            gnc_sto.pftsum.__dict__['HET_RESP'] = gnc_sto.d1.tRESP_HETERO
            gnc_sto.pftsum.__dict__['CO2_FIRE'] = gnc_sto.d1.tCO2_FIRE
            gnc_sto.remove_variables(['tGPP','tRESP_GROWTH','tRESP_MAINT','tRESP_HETERO','tCO2_FIRE'])

            gnc_sto.retrieve_variables(['tBIOMASS','tLITTER','tSOILC'])
            gnc_sto.pftsum.__dict__['TOTAL_M'] = gnc_sto.d1.tBIOMASS
            gnc_sto.pftsum.__dict__['TOTAL_SOIL_CARB'] = gnc_sto.d1.tLITTER + gnc_sto.d1.tSOILC
            gnc_sto.remove_variables(['tBIOMASS','tLITTER','tSOILC'])

        # we have to treat product pool independently
        try:
            gnc_sto.retrieve_variables(['PROD10','PROD100'])
            gnc_sto.pftsum.PROD10 = gnc_sto.d1.PROD10.sum(axis=1)
            gnc_sto.pftsum.PROD100 = gnc_sto.d1.PROD100.sum(axis=1)
            gnc_sto.remove_variables(['PROD10','PROD100'])
        except KeyError:
            gnc_sto.pftsum.PROD10 = gnc_sto.pftsum.NPP * 0.
            gnc_sto.pftsum.PROD100 = gnc_sto.pftsum.NPP * 0.

        # get the spatial operation and pass them into dataframe
        if not gnc_sto._SinglePoint:
            gnc_sto.get_spa()
            dft = pa.DataFrame(gnc_sto.spasum.__dict__)
        else:
            dft = pa.DataFrame(gnc_sto.pftsum.__dict__)

        # treat the output time step
        if out_timestep == 'annual':
            flux_scale_factor = 365.
            dft['CO2FLUX'] = dft['CO2FLUX']/30.  #CO2FLUX is monthly output
        elif out_timestep == 'daily':
            flux_scale_factor = 1
        dft[list_flux] = dft[list_flux]*flux_scale_factor

        # get total carbon pool
        dft['PROD'] = dft['PROD10'] + dft['PROD100']
        dft['CarbonPool'] = dft['TOTAL_M'] + dft['TOTAL_SOIL_CARB'] + dft['PROD']

        # calcate NBP
        dft['NBP_npp'] = dft['NPP']+dft['CO2_TAKEN']-dft['CONVFLUX']-dft['CFLUX_PROD10']-dft['CFLUX_PROD100']-dft['CO2_FIRE']-dft['HARVEST_ABOVE']-dft['HET_RESP']
        dft['NBP_co2flux'] = -1*(dft['CO2FLUX']+dft['HARVEST_ABOVE']+dft['CONVFLUX']+dft['CFLUX_PROD10']+dft['CFLUX_PROD100'])

    elif version == 2:
        # list all pools and fluxes
        list_flux_pft = ['GPP','NPP','HET_RESP','CO2_FIRE','CO2FLUX','CO2_TAKEN','METHANE','RANIMAL']
        list_flux_pftsum = ['CONVFLUX_LCC','CONVFLUX_HAR','CFLUX_PROD10_LCC','CFLUX_PROD10_HAR','CFLUX_PROD100_LCC','CFLUX_PROD100_HAR','HARVEST_ABOVE']
        list_flux = list_flux_pft+list_flux_pftsum

        list_pool = ['TOTAL_M','TOTAL_SOIL_CARB','LEAF_M']
        list_all = list_flux_pft+list_flux_pftsum+list_pool
        nlist_var = [list_flux_pft, list_flux_pftsum, list_pool]

        for varlist in nlist_var:
            gnc_sto.retrieve_variables(varlist,mask=spamask)
            gnc_sto.get_pftsum(print_info=False,veget_npindex=veget_npindex)
            gnc_sto.remove_variables(varlist)

        #handle adjustment of different variables
        if dgvmadj:
            if veget_npindex != np.s_[:]:
                raise ValueError("dgvmadj is not handled when veget_npindex does not include all")
            else:
                gnc_sto.retrieve_variables(['tGPP','tRESP_GROWTH','tRESP_MAINT','tRESP_HETERO','tCO2_FIRE'],mask=spamask)
                gnc_sto.pftsum.__dict__['NPP'] = gnc_sto.d1.tGPP - gnc_sto.d1.tRESP_MAINT - gnc_sto.d1.tRESP_GROWTH
                gnc_sto.pftsum.__dict__['HET_RESP'] = gnc_sto.d1.tRESP_HETERO
                gnc_sto.pftsum.__dict__['CO2_FIRE'] = gnc_sto.d1.tCO2_FIRE
                gnc_sto.remove_variables(['tGPP','tRESP_GROWTH','tRESP_MAINT','tRESP_HETERO','tCO2_FIRE'])

                gnc_sto.retrieve_variables(['tBIOMASS','tLITTER','tSOILC'],mask=spamask)
                gnc_sto.pftsum.__dict__['TOTAL_M'] = gnc_sto.d1.tBIOMASS
                gnc_sto.pftsum.__dict__['TOTAL_SOIL_CARB'] = gnc_sto.d1.tLITTER + gnc_sto.d1.tSOILC
                gnc_sto.remove_variables(['tBIOMASS','tLITTER','tSOILC'])

        # we have to treat product pool independently
        list_prod = ['PROD10_LCC','PROD10_HAR','PROD100_LCC','PROD100_HAR']
        gnc_sto.retrieve_variables(list_prod,mask=spamask)
        for var in list_prod:
            gnc_sto.pftsum.__dict__[var] = gnc_sto.d1.__dict__[var][veget_npindex].sum(axis=1)
        print gnc_sto.d1.__dict__['PROD10_LCC'][veget_npindex].shape
        print gnc_sto.d1.__dict__['PROD10_LCC'].shape
        print gnc_sto.pftsum.__dict__['PROD10_LCC'].shape
        gnc_sto.remove_variables(list_prod)

        # get the spatial operation and pass them into dataframe
        if not gnc_sto._SinglePoint:
            gnc_sto.get_spa(areaind=areaind)
            dft = pa.DataFrame(gnc_sto.spasum.__dict__)
        else:
            dft = pa.DataFrame(gnc_sto.pftsum.__dict__)

        #  2016-03-30: the shape of gnc_sto.d1.ContAreas could be
        #  (nlat,nlon) when there is no "CONTFRAC" or "NONBIOFRAC" in
        #  the history file, but could be (ntime,nlat,nlon) when they're
        #  present.

        #  # [++temporary++] treat CO2_TAKEN
        #  # In case of shifting cultivation is simulated, the CO2_TAKEN
        #  # could be big at the last day. However the veget_max is kept
        #  # the same as the old one over the year, so we have to use
        #  # last-year CO2_TAKEN multiply with the next-year veget_max.
        #  gnc_sto.retrieve_variables(['CO2_TAKEN'])
        #  co2taken_pftsum = np.ma.sum(gnc_sto.d1.CO2_TAKEN[:-1] * gnc_sto.d1.VEGET_MAX[1:],axis=1)
        #  if not gnc_sto._SinglePoint:
        #      dt = np.sum(co2taken_pftsum*gnc_sto.d1.ContAreas,axis=(1,2))
        #  else:
        #      dt = co2taken_pftsum
        #  dft['CO2_TAKEN'].iloc[:-1] = dt

        # treat the output time step
        if out_timestep == 'annual':
            flux_scale_factor = 365.
            dft['CO2FLUX'] = dft['CO2FLUX']/30.  #CO2FLUX is monthly output
        elif out_timestep == 'daily':
            flux_scale_factor = 1
        dft[list_flux] = dft[list_flux]*flux_scale_factor

        # get total carbon pool
        dft['PROD'] = dft['PROD10_LCC'] + dft['PROD10_HAR'] + dft['PROD100_LCC'] + dft['PROD100_HAR']
        dft['CarbonPool'] = dft['TOTAL_M'] + dft['TOTAL_SOIL_CARB'] + dft['PROD']

        # treat GM
        dft['RANIMAL'] = dft['RANIMAL']*1000
        dft['METHANE'] = dft['METHANE']*1000
        dft['GMsource'] = dft['RANIMAL'] + dft['METHANE']

        # treat LUC
        dft['CONVFLUX'] = dft['CONVFLUX_LCC'] + dft['CONVFLUX_HAR']
        dft['CFLUX_PROD10'] = dft['CFLUX_PROD10_LCC'] + dft['CFLUX_PROD10_HAR']
        dft['CFLUX_PROD100'] = dft['CFLUX_PROD100_LCC'] + dft['CFLUX_PROD100_HAR']
        dft['LUCsource'] = dft['CONVFLUX'] + dft['CFLUX_PROD10'] + dft['CFLUX_PROD100']

        # calcate NBP
        dft['NBP_npp'] = dft['NPP']+dft['CO2_TAKEN']-dft['CONVFLUX']-dft['CFLUX_PROD10']-dft['CFLUX_PROD100']-dft['CO2_FIRE'] \
                         -dft['HARVEST_ABOVE']-dft['HET_RESP']-dft['RANIMAL']-dft['METHANE']
        dft['NBP_co2flux'] = -1*(dft['CO2FLUX']+dft['HARVEST_ABOVE']+dft['CONVFLUX']+dft['CFLUX_PROD10']+dft['CFLUX_PROD100'])

    else:
        raise ValueError("Unknown version!")

    gnc_sto.close()

    return dft

def panel_summary_stomate(dic):
    """
    Return a panel of summary dataframe for stomate_history.nc using
    dataframe_from_stomate

    Parameters:
    -----------
    dic: a dictionary of (tag,filename) pairs.
    """
    pdic = OrderedDict()
    for k,f in dic.items():
        pdic[k] = dataframe_from_stomate(f)
    return pa.Panel(pdic)

def write_PFTmap(filename,data,resolution='05deg'):
    if resolution=='05deg':
        ### Build the dimensions
        ncfile = gnc.NcWrite(filename)
        ncfile.add_dim_lat(bounds = "bounds_lat",latvar=np.arange(89.75,-90,-0.5),units = "degrees_north",valid_min = -90.,valid_max = 90.,long_name = "Latitude",axis = "Y")
        ncfile.add_dim_lon(bounds = "bounds_lon",lonvar=np.arange(-179.75,180,0.5),units = "degrees_east",valid_min = -180.,valid_max = 180.,long_name = "Longitude",axis = "X")
        timedim_info = ['time_counter','time_counter','time_counter','f4',np.array([1850]),'no unit',True]
        ncfile.add_dim(timedim_info,units = "years since 0-1-1",Calendar = "gregorian",axis = "T")
        vegetdim_info = ['veget','veget','Vegetation Classes','i4',np.arange(1,16),"-",False]
        ncfile.add_dim(vegetdim_info,validmax = 15.,validmin = 1.)
        import cons
        from collections import OrderedDict
        pftdic = OrderedDict()
        for i in range(1,16):
            pftdic['PFT{0:0>2}'.format(i)] = cons.pftdic15[i]
            ### Write the vegetmax
        varinfo_value = ['maxvegetfrac',('time_counter', 'veget', 'lat', 'lon',),'f4',data]
        ncfile.add_var(varinfo_value,name = "maxvegetfrac",long_name = "Vegetation types",units = "-")
        ncfile.add_global_attributes(pftdic)
        ncfile.add_history_attr(histtxt="PFT map generated using data from FireMIP")
        ncfile.close()


