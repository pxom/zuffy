"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
Functions to handle the display of a FPT
"""

class ObjectColor():

    def_operator_colors = [ # default list of operator colors (pale pastels)
        '#ff999922',
        '#99ff9922',
        '#9999ff22',
        '#99ffff22',
        '#ff99ff22',
        '#ffff9922',
        ]
    
    def_feature_colors = [ # default list of feature colors (strong)
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf',
        ]

    def __init__(self, color_list=None):
        self.object_colors  = color_list
        self.used_colors    = {}

    def getColor(self, object_name):
        """
        This returns a color from the list of Operator or Feature colors.
        If the named object (Feature or Operator) does not already exist on the list of seen 
        objects then the next available color on the list is assigned and returned otherwise
        it returns the same color as when the object was first seen.
        """
        cmap = self.object_colors
        if object_name in self.used_colors:
            return cmap[self.used_colors[object_name]]
        else:
            next_color_id = len(self.used_colors) % len(cmap) # wrap around if at end of the list
            self.used_colors[object_name] = next_color_id
            return cmap[next_color_id]

class FeatureColor(ObjectColor):
    """
    Initialises the list of Feature colors.
    """

    def __init__(self, color_list=None):
        if color_list == None:
            color_list = self.def_feature_colors
        else:
            color_list.extend(self.def_feature_colors)
        super().__init__(color_list)


class OperatorColor(ObjectColor):
    """
    Initialises the list of Operator colors.
    """
    def __init__(self, color_list=None):
        if color_list == None:
            color_list = self.def_operator_colors
        else:
            color_list.extend(self.def_operator_colors)
        super().__init__(color_list)

