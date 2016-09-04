#!/usr/bin/env python

from collections import OrderedDict
import pandas as pa
import matplotlib as mat
import numpy as np
import g
import pb
import tools

def _replace_none_by_given(orinput,default):
    if orinput is None:
        return default
    else:
        return orinput

class LabelAxes(object):
    def __init__(self,tags=None,axl=None):
        if not isinstance(tags,(list,tuple,np.ndarray)):
            raise TypeError("tags could only be list or tuple numpy array")
        else:
            if not isinstance(axl[0],mat.axes.Axes):
                raise TypeError('value must be mat.axes.Axes type!')
            else:
                if len(tags) > len(axl):
                    raise ValueError("length of axl less than tags")
                else:
                    self.data = OrderedDict(zip(tags,axl))
                    self.tags = tags
                    self.tagnum = len(self.tags)
                    self.axl = list(axl)[0:self.tagnum]


    def _get_tags(self,key):
        #normal key:
        if isinstance(key,(str,unicode)):
            return [key]
        elif isinstance(key,int):
            return [self.tags[key]]
        else:
            if isinstance(key, slice):
                return self.tags[key]
            elif isinstance(key, list):
                if len(np.unique(map(type,key))) > 1:
                    raise TypeError("input list must be single type")
                else:
                    if isinstance(key[0],str):
                        return key
                    elif isinstance(key[0],int):
                        return [self.tags[index-1] for index in key]
                    else:
                        raise TypeError("slice not understood.")
            else:
                raise TypeError("slice not understood.")

    def __getitem__(self,key):
        subtags = self._get_tags(key)
        subvals = [self.data[stag] for stag in subtags]
        if len(subtags) == 1 and not isinstance(key,list):
            return subvals[0]
        else:
            return LabelAxes(tags=subtags,axl=subvals)

    def __len__(self):
        return self.tagnum

    def __repr__(self):
        return '\n'.join([repr(self.__class__),"tags:",','.join(self.tags)])

    def _propagate(self,arg,itearg=False):
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
        return tools._propagate(self.tags,arg,itearg=itearg)

    @property
    def figure(self):
        return self.axl[0].figure

    def to_dict(self):
        return OrderedDict(zip(self.tags,self.axl))

    def iteritems(self):
        for tag in self.tags:
            yield tag,self.data[tag]

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, **kwargs):
        """
        """
        ylabdic = self._propagate(ylabel)
        for tag,ylabel in ylabdic.items():
            self.data[tag].set_ylabel(ylabel,
                                       fontdict=fontdict,
                                       labelpad=labelpad,
                                       **kwargs)
    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, **kwargs):
        """
        """
        xlabdic = self._propagate(xlabel)
        for tag,xlabel in xlabdic.items():
            self.data[tag].set_xlabel(xlabel,
                                       fontdict=fontdict,
                                       labelpad=labelpad,
                                       **kwargs)

    def set_xlim(self,xlim,**kwargs):
        xlimdict = self._propagate(xlim,itearg=True)
        for tag,xlim in xlimdict.items():
            self.data[tag].set_xlim(xlim,**kwargs)

    def set_ylim(self,ylim,**kwargs):
        ylimdict = self._propagate(ylim,itearg=True)
        for tag,ylim in ylimdict.items():
            self.data[tag].set_ylim(ylim,**kwargs)

    def set_axis_bgcolor(self,color):
        colordict = self._propagate(color,itearg=False)
        for tag,c in colordict.items():
            self.data[tag].set_axis_bgcolor(c)

    def add_label(self,label=None,pos='ul',color='k',ftdic={'size':12},**kwargs):
        """
        Use g.Set_AxText to add labels for the axes.

        Parameters:
        -----------
        label: tags will be used when it is None, properly broadcast when
            label is given.

        pos:
            - a dictionary of {'first':(x0,y0),'horizontal':hshift,'vertical':vshift},
                positions will be internally calculated.
            - other types will be properly broadcast.
        color: will be properly broadcast.
        """
        labdic=OrderedDict()

        if label is None:
            label = self.tags
        labeldic = self._propagate(label)

        if isinstance(pos,dict):
            dic = pos.copy()
            if 'horizontal' in dic:
                pos = tools.expand_by_interval(dic['first'],
                            len(taglist),horizontal=dic['horizontal'])
            elif 'vertical' in dic:
                pos = tools.expand_by_interval(dic['first'],
                            len(taglist),vertical=dic['vertical'])
        posdic = self._propagate(pos)

        if isinstance(color,list) and isinstance(color[0],(tuple,list,np.ndarray)):
            colordic = self._propagate(color,itearg=True)
        else:
            colordic = self._propagate(color)

        print colordic

        for tag,ax in self.iteritems():
            labdic[tag] = g.Set_AxText(ax,labeldic[tag],pos=posdic[tag],
                                       color=colordic[tag],
                                       ftdic=ftdic,**kwargs)

        return labdic

    def add_info_dataframe(self,dft,keyword,
                     pos='ul',color='k',
                     prefix='',surfix='',
                     show_keyword=True,
                     numdigit=3,
                     ftdic={'size':'12'},**kwargs):
        """
        Add information from the dataframe using add_label method.

        Parameters:
        -----------
        dft: Input dataframe
        keyword: string or list of strings, used to indicate what information
            is required.
        prefix/surfix: string type, used to be pre/surfixed in the text.
        numdigit: Number of digit to show when convert float into string
        show_keyword: whether the keyword is shown, True when keyword is
            a string.

        (Parameters from add_label method)
        pos:
            string or tuple type, used in the g.Set_AxText method.
            - a dictionary of {'first':(x0,y0),'horizontal':hshift,'vertical':vshift},
                positions will be internally calculated.
            - other types will be properly broadcast.
        color: will be properly broadcast.
        """

        ## Major part of codes are for constructing final annotation dict.
        # A temporary function to convert float numbers to string for
        # display
        def convert_to_str(keyword_dic,show_keyword,keyword,numdigit):
            dic = OrderedDict()
            for key,val in keyword_dic.items():
                if show_keyword:
                    dic[key] = keyword+'='+str(round(val,numdigit))
                else:
                    dic[key] = str(round(val,numdigit))
            return dic


        # retrieve the composite information by combing all keywords
        if isinstance(keyword,str):
            if keyword not in dft.index:
                raise ValueError("{0} not in the dft index".format(keyword))
            else:
                keyword_dic = dft.ix[keyword].to_dict()
                final_keyword_dic = convert_to_str(keyword_dic,show_keyword,keyword,numdigit)

        elif isinstance(keyword,list):

            # creat an empty dict, we cannot use
            # mulkwdic =
            # OrderedDict.fromkeys(dft.columns,[None]*len(keyword))
            # because all values of the created dic share the same
            # memory.
            emptylist = [None]*len(keyword)
            mulkwdic = OrderedDict()
            for tag in dft.columns:
                mulkwdic[tag] = emptylist[:]

            for i,kw in enumerate(keyword):
                keyword_dic = dft.ix[kw].to_dict()
                tmp_kwdic = convert_to_str(keyword_dic,True,kw,numdigit)
                for tag in dft.columns:
                    mulkwdic[tag][i] = tmp_kwdic[tag]

            final_keyword_dic = OrderedDict()
            for tag in mulkwdic.keys():
                final_keyword_dic[tag] = ', '.join(mulkwdic[tag])
        else:
            raise ValueError("keyword must be a string or list of string")


        # add prefix and surfix
        keyword_anno_dic = OrderedDict()
        for key,val in final_keyword_dic.items():
            keyword_anno_dic[key] = prefix+val+surfix


        ## display the information using lax.add_label method.
        self.add_label(label=keyword_anno_dic,pos=pos,color=color,
                               ftdic=ftdic,**kwargs)


    def call(self,funcname,*args,**kwargs):
        """
        Call a mat.axes.Axes method with identical args and kwargs for
            all tags.
        """
        for tag,ax in self.data.items():
            ax.__getattribute__(funcname)(*args,**kwargs)

    def apply(self,func,copy=False):
        """
        Apply function that applies on axes object.
        """
        if not copy:
            for tag,ax in self.data.items():
                func(ax)
        else:
            newaxs = [func(ax) for ax in self.data.values()]
            return LabelAxes(self.tags,newaxs)

    def duplicate_twinx(self):
        """
        Return another LabelAxes object
        """
        twinaxl = [host.twinx() for host in self.axl]
        return LabelAxes(tags=self.tags,axl=twinaxl)

    def build_icecore(self,num=3,keys=None,direction='vertical'):
        """
        Build an icecore like axes by num.

        Returns:
        --------
        A dictionary of LabelAxes object with each LabelAxes as (keys,subplots)

        Parameters:
        -----------
        num: the number of new axes in each original axes
        keys: used as the keys for the output dictionay, otherwise will
            be sequential numbers (i.e., [1,2,3,...]).
        direction: currently could only be vertical.

        Notes:
        ------
        1. The sequence is always from bottom to top.
        """
        newaxs = [g.Axes_Replace_by_IceCore(ax,num,direction=direction)
                    for ax in self.axl]
        newaxs = np.array(newaxs)
        print newaxs.shape
        if keys is None:
            keys = range(newaxs.shape[1])

        dic = OrderedDict()
        for i,tag in enumerate(self.tags):
            dic[tag] = LabelAxes(tags=keys,axl=newaxs[i,:][::-1])

        return dic


    def reduce_ticks_by_half(self,axis='y'):
        """
        Reduce the x/y/both axis ticks by half.
        """
        for ax in self.axl:
            if axis == 'y':
                ortick = ax.get_yticks()
                ax.set_yticks(ortick[::2])
            elif axis == 'x':
                ortick = ax.get_xticks()
                ax.set_xticks(ortick[::2])
            elif axis == 'both':
                ortick = ax.get_yticks()
                ax.set_yticks(ortick[::2])
                ortick = ax.get_xticks()
                ax.set_xticks(ortick[::2])
            else:
                raise ValueError("wrong axis value!")

    def remove_attribute(self,attrname,index=slice(None)):
        """
        Remove some attributes of the mat.axes.Axes object allowing the
        resetting.

        Parameters:
        -----------
        attrname: the attribute name used to get the attributes to be
            removed. For instance, to remove the texts attributes, set
            attrname as 'texts'
        index: a python slice object used to indicate which slices of the
            attributes (list) is to be removed.
        """
        for ax in self.axl:
            attr_list = ax.__getattribute__(attrname)
            attr_list_remove = attr_list[index]
            for attr in attr_list_remove:
                attr.remove()

    def add_axes(self,pos='right',offset=None,
                 width=None,height=None,middle=True,**kwargs):
        """
        Add axes relative to each of the lax axes member, by calling
            g.Axes_add_axes. Return a lax instance.

        Parameters:
        -----------
        relative: currently could only be a list of ndarray of axes.
        pos: the relative position of new axes to the relative, could be
            'left','right','above','below'
        offset: offset in unit if fraction of figure of the new added axes
            in the direction of `pos` to the relative.
        width,height: the width and height of the new axes. In case of pos
            ='right/left', height will be calculated as the same of
            the relative unless it's being forced. In case of the pos=
            'above/below', width will be calculated as the same of
            the relative unless it's being forced.
        middle: boolean value. True to put the axes right in the middle.

        kwargs: kwargs used in mat.figure.Figure.add_axes
        """
        dic = OrderedDict()
        for tag in self.tags:
            dic[tag] = g.Axes_add_axes(relative=[self.data[tag]],
                         pos=pos,offset=offset,width=width,
                         height=height,middle=middle,**kwargs)
        lax = LabelAxes(tags=dic.keys(),axl=dic.values())
        return lax

def apply_list_lax(laxlist,func,copy=False):
    """
    Apply function to list of LabelAxes object
    """
    if not copy:
        for lax in laxlist:
            lax.apply(func)
    else:
        return [lax.apply(func,True) for lax in laxlist]


class LabelAxes2D(object):
    """
    Initiate by a OrderedDict of LabelAxes object.
    """
    def __init__(self,laxdic):
        if not isinstance(laxdic,OrderedDict):
            raise TypeError("must be OrderedDict of LabelAxes objects")
        else:
            if not isinstance(laxdic.values()[0],LabelAxes):
                raise TypeError('dict values must be LabelAxes objects')
            else:
                self.child_lax = laxdic
                self.parent_tags = laxdic.keys()
                self.child_tags = laxdic.values()[0].tags

    @property
    def data(self):
        dic = OrderedDict()
        for ptag in self.parent_tags:
            dic[ptag] = self.child_lax[ptag].data
        return dic

    def child_ix(self,ctag):
        """
        Return directly a LabelAxes.LabelAxes project by selecting a child tag.
        """
        data = pb.Dic_Nested_Permuate_Key(self.data)
        dic = data[ctag]
        return LabelAxes(tags=dic.keys(),axl=dic.values())


    def __repr__(self):
        return '\n'.join([repr(self.__class__),
                          "parent_tags:",','.join(self.parent_tags),
                          "child_tags:",','.join(self.child_tags)])

    def __getitem__(self,key):
        return self.child_lax[key]

    def iteritems(self):
        for ptag in self.parent_tags:
            yield ptag,self.child_lax[ptag]

    def set_xlim(self,xlim,**kwargs):
        for ptag,plax in self.child_lax.items():
            plax.set_xlim(xlim,**kwargs)

    def set_ylim(self,ylim,**kwargs):
        for ptag,plax in self.child_lax.items():
            plax.set_ylim(ylim,**kwargs)

    def set_axis_bgcolor(self,color):
        colordict = tools._propagate(self.parent_tags,color,itearg=False)
        for ptag in self.parent_tags:
            self.child_lax[ptag].set_axis_bgcolor(colordict[ptag])

    def add_parent_label(self,pos='ouc',ftdic={'size':12},**kwargs):
        for ptag,lax in self.iteritems():
            lax.add_label(label=ptag,pos=pos,ftdic=ftdic,**kwargs)

    def add_child_label(self,ptag=None,pos='ouc',color='m',ftdic={'size':12},**kwargs):
        ptag = _replace_none_by_given(ptag,self.parent_tags[-1])
        self.child_lax[ptag].add_label(pos=pos,ftdic=ftdic,color=color,**kwargs)

    def apply(self,func,copy=False):
        """
        Apply function that applies on axes object.
        """
        if not copy:
            for ptag,clax in self.iteritems():
                clax.apply(func)
        else:
            raise ValueError("Not implemented")

def label_axl(axl,label,pos='ul',forcelabel=None,**kwargs):
    """
    Parameters:
    -----------
    forcelabel: the 'label' parameter in LabelAxes.add_label
    """
    lax = LabelAxes(tags=label,axl=axl)
    lax.add_label(label=forcelabel,pos=pos,**kwargs)
    return lax


