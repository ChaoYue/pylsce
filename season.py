#!/usr/bin/env python

import numpy as np
from collections import OrderedDict

seasons=['Spring','Summer','Autumn','Winter']

def monthly_data_to_season(arr,pyfunc=None,annual_mean=False):
    """
    Transform the continuous monthly data into by seasons.

    Parameters:
    -----------
    arr: The first dimension of arr must be the product of 12.
    pyfunc: python function used to integrate the three months of the season
        into the season-integrated value.

    Return:
        A dictionary of arrays, with the keys as the four seasons.
    """
    if arr.shape[0]%12 != 0:
        raise ValueError("The first dim length of input array is not a product of 12")
    else:
        arr_new = arr.reshape(12,-1,*arr.shape[1:],order='F')
        arr_season = np.split(arr_new,[3,6,9,12],axis=0)[:-1]
        arr_season_integrated = map(lambda x:pyfunc(x,axis=0),arr_season)
        if annual_mean == True:
            arr_season_integrated.append(pyfunc(arr_new,axis=0))
            taglist = seasons + ['Annual']
        else:
            taglist = seasons
        return OrderedDict(zip(taglist,arr_season_integrated))

