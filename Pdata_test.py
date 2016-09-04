#!/usr/bin/env python

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

#easy typing
#test_Hstack_Dic_By_Key test_Vstack_By_Tag test__class__ test__delattr__ test__dict__ test__doc__ test__format__ test__getattribute__ test__hash__ test__init__ test__module__ test__new__ test__reduce__ test__reduce_ex__ test__repr__ test__setattr__ test__sizeof__ test__str__ test__subclasshook__ test__weakref__ test_add_indata_by_tag_and_column test_all_nonplot_keylist test_bar_attr_base_keylist test_bar_attr_default test_broadcast_scalar_byx test_data_base_keylist test_data_complete_check_all test_data_complete_check_by_tag test_error_attr_base_keylist test_error_attr_default test_extra_base_keylist test_extra_base_keylist_default test_extra_nonplot_keylist test_fill_errNone_with_Nan test_gbar test_gerrorbar test_get_Scatter_PathC_marker_size_color test_get_attr_value_from_handle test_get_axes_by_artistdic test_get_err_by_tag test_get_handler_label_by_artistdic test_get_plot_attr_list_except test_get_plot_attr_value_from_data test_get_scatter_marker_size_color test_get_single_attr_by_check_similar_attrname_and_condition test_gscatter test_new_entry test_plot_attr_default test_plot_attr_dic test_plot_attr_keylist_all test_plot_attr_keylist_check test_remove_artist_by_tag test_scatter_attr_base_keylist test_scatter_attr_default test_set_default_tag test_set_ecolor_by_bar test_set_ecolor_by_scatter test_add_attr_by_tag test_add_entry_by_dic test_add_entry_df_groupby_column test_add_entry_doubleYerror test_add_entry_noerror test_add_entry_noerror_by_dic_default_xindex test_add_entry_sharex_noerror_by_dic test_add_entry_singleYerror test_add_entry_singleYerror3 test_add_tag test_addx test_addxerrh test_addxerrl test_addy test_addyerrh test_addyerrl test_apply_function test_bar test_copy test_creat_list_of_axes_by_tagseq test_errorbar test_get_data_as_dic test_imshow test_leftshift test_list_attr test_list_attr_extra_base test_list_attr_plot test_list_keys_for_tag test_list_tags test_plot test_plot_split_axes test_plot_split_axes_byX test_plot_stackline test_pool_data_by_tag test_regroup_data_by_tag test_regroup_data_by_tag_keyword test_remove_line_by_tag test_remove_scatter_by_tag test_scatter test_scattertag test_set_data_void test_set_default_plot_attr test_set_legend_all test_set_legend_bar test_set_legend_line test_set_legend_scatter test_set_legend_select test_set_new_tags test_set_tag_order test_setp_by_tag test_shift_ydata test_subset_begin test_subset_end test_to_DataFrame test_to_dic test_to_pd

test_taglist = ['wet1', 'wet2', 'wet3', 'dry1', 'dry2', 'dry3']
cwet=['red','magenta','orange']
cdry=['blue','cyan','green']
c_wet_dry = cwet+cdry
ssize_list = range(20,171,30)

def assert_equal_unsorted_pdtag_list(pd,taglist):
    '''
    Assert if the taglists of the Pdata.Pdata object is equal to the
    given taglist, no sequence is considered.
    '''
    ntools.assert_list_equal(sorted(pd.list_tags()),sorted(taglist))

def assert_equal_taglist(pd1,pd2):
    """
    assert taglist equal for two Pdata objects
    """
    data1 = pd1.data
    data2 = pd2.data
    taglist1 = sorted(data1.keys())
    taglist2 = sorted(data2.keys())
    ntools.assert_list_equal(taglist1,taglist2,'''the two Pdata object do not
        have the same tags''')

def assert_equal_keylist_by_tag(pd1,pd2,tag):
    """
    assert keylist equal for the same tag for two Pdata objects
    """
    data1 = pd1.data
    data2 = pd2.data
    try:
        keylist1 = sorted(data1[tag].keys())
        keylist2 = sorted(data2[tag].keys())
        ntools.assert_list_equal(keylist1,keylist2,'''tag value of the
            two Pdata objects do not have the same keys''')
    except KeyError:
        raise KeyError('''specified tag does not appear in one of the
            tow Pdata objects''')


def assert_equal_by_tag_key(pd1,pd2,tag,key):
    """
    assert equal value for specified tag/key combination for two Pdata objects
    """
    data1 = pd1.data
    data2 = pd2.data
    try:
        value1 = data1[tag][key]
        value2 = data2[tag][key]
        if isinstance(value1,np.ndarray) and isinstance(value2,np.ndarray):
            np.testing.assert_allclose(value1,value2,err_msg=''' the values
            for tag '{0}' and key '{1}' are not equal'''.format(tag,key))
        else:
            ntools.assert_equal(value1,value2)
    except KeyError:
        raise KeyError('''specified tag/key combination not found in one
            of the two Pdata objects''')

def assert_equal_Pdata(pd1,pd2):
    """
    Assert if tags, keys for each tag, and values for each tag/key combination
        for the two Pdata objects are all equal. i.e., the 'data' field of the
        two Pdata objects are completely equal.
    """
    assert_equal_taglist(pd1,pd2)
    for tag in pd1.list_tags():
        assert_equal_keylist_by_tag(pd1,pd2,tag)
        for key in pd1.list_keys_for_tag(tag):
            assert_equal_by_tag_key(pd1,pd2,tag,key)

def create_pddic(taglist):
    """
    Creat dictionary for building Pdata object. the dictionary has values for
        all the keys in Pdata.Pdata._data_base_keylist
    """
    pddic = dict((tag,{})for tag in taglist)
    for i,tag in enumerate(taglist):
        pddic[tag]['x'] = np.arange(1,11) + i
        pddic[tag]['y'] = np.arange(1,11) + 10*i
        pddic[tag]['xerrl'] = np.round(np.random.uniform(0,0.5,10),2)
        pddic[tag]['xerrh'] = np.round(np.random.uniform(0,0.5,10),2)
        pddic[tag]['yerrl'] = np.round(np.random.uniform(0,2,10),2)
        pddic[tag]['yerrh'] = np.round(np.random.uniform(0,2,10),2)
    return pddic

def create_Pdata_by_pddic_basic_method(pddic):
    """
    Create a Pdata object by using "basic" method, i.e., all the key fiels are
        added by using method "add*".
    """
    pd = Pdata.Pdata()
    for tag,data_dic in pddic.items():
        pd.add_tag(tag)
        pd.addx(pddic[tag]['x'],tag)
        pd.addy(pddic[tag]['y'],tag)
        pd.addxerrl(pddic[tag]['xerrl'],tag)
        pd.addxerrh(pddic[tag]['xerrh'],tag)
        pd.addyerrl(pddic[tag]['yerrl'],tag)
        pd.addyerrh(pddic[tag]['yerrh'],tag)
    return pd

def create_Pdata_by_taglist(taglist):
    pddic = create_pddic(taglist)
    pd = create_Pdata_by_pddic_basic_method(pddic)
    return pd

def create_pd_for_nestpd(num):
    pd = create_Pdata_by_taglist(test_taglist)
    pd.set_tag_order(test_taglist)
    pd.apply_function(func=lambda x:x+num*10,axis='y')
    return pd

def create_nestpd(ptaglist=None):
    if ptaglist is None:
        parent_taglist = ['test'+str(i) for i in range(1,5)]
    else:
        parent_taglist = ptaglist[:]

    nestpd = {}
    for ptag,num in zip(parent_taglist,range(2,1+2*len(parent_taglist),2)):
        nestpd[ptag] = create_pd_for_nestpd(num)
    #nested Pdata receives a dictionary of Pdata
    npd = Pdata.NestPdata(nestpd)
    return npd


class test_add_entry_by_dic:
    """
    Add data to Pdata object. The "add*" method and "add_entry_by_dic" method
        should give the equal Pdata object
    """

    def test_single_tag(self):
        pddic = create_pddic('tag')
        pd1 = create_Pdata_by_pddic_basic_method(pddic)
        pd2 = Pdata.Pdata()
        pd2.add_entry_by_dic(**pddic)
        assert_equal_Pdata(pd1,pd2)

    def test_multiple_tag(self):
        pd1 = create_Pdata_by_taglist(['a','b','c'])
        pd2 = create_Pdata_by_taglist(['l','m','n'])
        ntools.assert_list_equal(sorted(pd1.list_tags()),sorted(['a','b','c']))
        ntools.assert_list_equal(sorted(pd2.list_tags()),sorted(['l','m','n']))


class test_add_entry_noerror:
    '''Test Pdata.Pdata.add_entry_noerror '''
    def test_x_None(self):
        pd = Pdata.Pdata()
        y = np.arange(11,21)
        pd.add_entry_noerror(x=None,y=y,tag='tag')
        np.testing.assert_array_equal(pd.data['tag']['x'],np.arange(len(y))+1)
        np.testing.assert_array_equal(pd.data['tag']['y'],y)

    def test_x_not_None(self):
        x = np.arange(1,11)
        y = np.arange(11,21)
        pd = Pdata.Pdata()
        pd.add_entry_noerror(x=x,y=y,tag='tag')
        np.testing.assert_array_equal(pd.data['tag']['x'],x)
        np.testing.assert_array_equal(pd.data['tag']['y'],y)

    @ntools.raises(ValueError)
    def test_raise_x_lenth_error(self):
        '''
        Value error when lenth of x is not equal to y
        '''
        x = np.arange(1,12)
        y = np.arange(11,21)
        pd = Pdata.Pdata()
        pd.add_entry_noerror(x=x,y=y,tag='tag')
        #ntools.assert_raises(ValueError,pd.add_entry_noerror,x,y,'tag')

def test_expand_by_keyword():
    expected = [('dry1', 'r'), ('dry2', 'r'), ('dry3', 'r'), ('wet1', 'b'),
        ('wet2', 'b'), ('wet3', 'b')]
    output = Pdata.Pdata._expand_by_keyword(test_taglist,
                                            [('dry','r'),('wet','b')])
    ntools.assert_list_equal(expected,output)

class test_expand_tag_value_to_dic():
    '''
    Test _expand_tag_value_to_dic
    '''
    def test_nested_dic_and_pure_list(self):
        '''
        tag_attr_value_dic is a full dict or equal-length list or listoftuples
        '''
        test_colorlist = ['r','g','b','k','m','y']
        value_tuple_list = zip(test_taglist,test_colorlist)
        tag_color_dic = dict(value_tuple_list)
        #test equal-length list
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,
                                                      test_colorlist)
        ntools.assert_dict_equal(output,tag_color_dic)
        #test tuple_list
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,
                                                      value_tuple_list)
        ntools.assert_dict_equal(output,tag_color_dic)
        #test full dict
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,
                                                      tag_color_dic)
        ntools.assert_dict_equal(output,tag_color_dic)

    def test_single_value(self):
        '''
        single value broadcast by length of taglist
        '''
        exdic = dict(zip(test_taglist,['rgb']*6))
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,'rgb')
        ntools.assert_dict_equal(output,exdic)

    @ntools.raises(TypeError)
    def test_listoftuple_tagkw_nonboolean(self):
        '''
        TypeError expected when tagkw is nonboolean and a listoftuples provided.
        '''
        value_tuple_list = [('dry','r')]
        #test tuple_list
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,
                                           value_tuple_list,tagkw='dry')

    def test_single_value_tagkw_string(self):
        '''
        use tagkw as a string to broadcast single value to only few tags.
        '''
        tag_color_dic = dict.fromkeys(['dry1','dry2','dry3'],'r')
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,'r',
                                                      tagkw='dry')
        ntools.assert_dict_equal(output,tag_color_dic)

    @ntools.raises(TypeError)
    def test_single_value_tagkw_True(self):
        '''
        TypeError be raised tag_attr_value as single string and tagkw is True
        '''
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,'r',
                                                      tagkw=True)

    @ntools.raises(ValueError)
    def test_unequal_list_error(self):
        '''
        Unequal valuelist and taglist length should cause ValueError
        '''
        test_colorlist = ['r','g','b','k','m','y','lightblue']
        Pdata.Pdata._expand_tag_value_to_dic(test_taglist, test_colorlist)

    def test_tagkw(self):
        '''
        Test keyword for taglist is used.
        '''
        tv_tuple_list = [('dry','r'),('wet','b')]
        exlist = [('dry1', 'r'), ('dry2', 'r'), ('dry3', 'r'), ('wet1', 'b'),
            ('wet2', 'b'), ('wet3', 'b')]

        exdic = dict(exlist)
        output = Pdata.Pdata._expand_tag_value_to_dic(test_taglist,
                                                      tv_tuple_list,True)
        ntools.assert_dict_equal(output,exdic)

class test_pool_data_by_tag:
    def test_taglist(self):
        pd = create_Pdata_by_taglist(test_taglist)
        pd2 = pd.pool_data_by_tag(dry=['dry1','dry2','dry3'],
                                  wet=['wet1','wet2','wet3'])
        assert_equal_unsorted_pdtag_list(pd2,['dry','wet'])

    def test_ydata(self):
        pd = create_Pdata_by_taglist(test_taglist)
        pd2 = pd.pool_data_by_tag(dry=['dry1','dry2','dry3'],
                                  wet=['wet1','wet2','wet3'])
        np.testing.assert_array_equal(pd2.data['dry']['y'],np.arange(31,61))
        np.testing.assert_array_equal(pd2.data['wet']['y'],np.arange(1,31))


class test_set_tag_order:
    @ntools.raises(AssertionError)
    def test_tag_change(self):
        '''
        Test that the taglist have really been changed.
        '''
        pd = create_Pdata_by_taglist(test_taglist[::-1])
        pd2 = pd.copy()
        pd.set_tag_order(test_taglist)
        ntools.assert_list_equal(pd.list_tags(),pd2.list_tags())
    def test_tag_order(self):
        '''
        Test the taglist is changed into expected order.
        '''
        pd = create_Pdata_by_taglist(test_taglist[::-1])
        pd.set_tag_order(test_taglist)
        ntools.assert_list_equal(pd.list_tags(),test_taglist)
    def test_data_not_changed(self):
        '''
        Test data not changed when changing the taglist order.
        '''
        pd = create_Pdata_by_taglist(test_taglist[::-1])
        pd2 = pd.copy()
        pd.set_tag_order(test_taglist)
        assert_equal_Pdata(pd,pd2)


class test_add_attr_by_tag():
    def test_with_ordered_taglist(self):
        '''
        test with ordered taglist; two attributes tested.
        '''
        pd = create_Pdata_by_taglist(test_taglist[::-1])
        pd.set_tag_order(test_taglist)
        pd.add_attr_by_tag(scolor=c_wet_dry)
        pd.add_attr_by_tag(ssize=ssize_list)
        color_dict = dict(zip(test_taglist,c_wet_dry))
        ssize_dict = dict(zip(test_taglist,ssize_list))
        pd_color_dict = pd.get_data_as_dic('scolor')
        pd_ssize_dict = pd.get_data_as_dic('ssize')
        ntools.assert_dict_equal(pd_color_dict,color_dict)
        ntools.assert_dict_equal(pd_ssize_dict,ssize_dict)


class test_get_attr_sequential_check:
    def test_scatter_color(self):
        fig, ax = g.Create_1Axes()
        a = np.arange(10)
        col = ax.scatter(a,a,facecolor=np.array([ 0.,  0.,  1.,  1.]),
                         edgecolor=np.array([ 1.,  0.,  0.,  1.]))
        fc = Pdata.Pdata._get_attr_sequential_check(col,
            [(__builtin__.len,'_facecolors',0),
             (__builtin__.len,'_edgecolors',0)])
        np.testing.assert_array_equal(fc[0],np.array([ 0.,  0.,  1.,  1.]))
        ec = Pdata.Pdata._get_attr_sequential_check(col,
            [(__builtin__.len,'_facecolors',1),
             (__builtin__.len,'_edgecolors',0)])
        np.testing.assert_array_equal(ec[0],np.array([ 1.,  0.,  0.,  1.]))


class test_npd_set_tag_order():
    def test_npd_parent_tag(self):
        '''
        test set parent tag order.
        '''
        npd = create_nestpd()
        ptags = ['test'+str(i) for i in range(1,5)[::-1]]
        npd.set_parent_tag_order(ptags)
        ntools.assert_list_equal(npd.parent_tags,ptags)

    def test_npd_parent_tag(self):
        '''
        test set child tag order.
        '''
        npd = create_nestpd()
        npd.set_child_tag_order(test_taglist[::-1])
        ntools.assert_list_equal(npd.child_tags,test_taglist[::-1])



