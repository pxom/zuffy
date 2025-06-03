"""
@author: POM <zuffy@mahoonium.ie>
License: BSD 3 clause
Functions to handle the display of a FPT
"""

# standardised by gemini

from typing import List, Dict, Optional

class ColorAssigner:
    """
    A base class for assigning cyclical colors to named objects (e.g., features, operators).

    This class manages a pool of colors and assigns a unique color from the pool
    to each new `object_name` it encounters. If an `object_name` has been seen
    before, it returns the previously assigned color. If the color pool is exhausted,
    it cycles back to the beginning of the pool.

    Attributes
    ----------
    _DEFAULT_OPERATOR_COLORS : list[str]
        Default hexadecimal color codes for operators, typically pale pastels.
    _DEFAULT_FEATURE_COLORS : list[str]
        Default hexadecimal color codes for features, typically strong, distinct colors.
    """

    _DEFAULT_OPERATOR_COLORS: List[str] = [ # default list of operator colors (pale pastels)
        '#ff999922', '#99ff9922', '#9999ff22', '#99ffff22',
        '#ff99ff22', '#ffff9922',
    ]
    
    _DEFAULT_FEATURE_COLORS: List[str] = [ # default list of feature colors (strong)
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    def __init__(self, color_pool: Optional[List[str]] = None) -> None:
        """
        Initializes the ColorAssigner with a specified or default color pool.

        Parameters
        ----------
        color_pool : list[str], optional
            A list of hexadecimal color codes to use as the color pool.
            If None, subclasses should provide their specific default.
        """
        if color_pool is None:
            raise ValueError("`color_pool` must be provided during initialization or by a subclass.")
        
        self.colors: List[str] = color_pool
        self.assigned_colors: Dict[str, int] = {} # Maps object_name to index in self.colors

    def get_color(self, object_name: str) -> str:
        """
        Retrieves a color for a given object name.

        If the `object_name` has been assigned a color before, that same color is returned.
        Otherwise, a new color is assigned from the internal color pool, cycling through
        the pool if necessary, and then returned.

        Parameters
        ----------
        object_name : str
            The unique name of the object (e.g., a feature name, an operator string)
            for which to retrieve a color.

        Returns
        -------
        str
            The hexadecimal color code assigned to the `object_name`.
        """
        if object_name in self.assigned_colors:
            color_index = self.assigned_colors[object_name]
        else:
            # Calculate the next available color index, wrapping around the list
            color_index = len(self.assigned_colors) % len(self.colors)
            self.assigned_colors[object_name] = color_index
        
        return self.colors[color_index]


class FeatureColorAssigner(ColorAssigner):
    """
    Manages color assignments specifically for feature objects.

    Uses a distinct set of default colors suitable for features, which can be
    extended with user-provided colors.
    """
    def __init__(self, custom_colors: Optional[List[str]] = None) -> None:
        """
        Initializes the FeatureColorAssigner.

        Parameters
        ----------
        custom_colors : list[str], optional
            A list of custom hexadecimal color codes to use. These colors will
            be prioritized before the default feature colors are used.
            If None, only the default feature colors will be used.
        """
        if custom_colors is None:
            color_pool = self._DEFAULT_FEATURE_COLORS
        else:
            # Combine custom colors with default colors. Use list() to create a new list
            # from custom_colors to avoid modifying the original list passed by the user.
            color_pool = list(custom_colors) + self._DEFAULT_FEATURE_COLORS
        
        super().__init__(color_pool)


class OperatorColorAssigner(ColorAssigner):
    """
    Manages color assignments specifically for operator objects.

    Uses a distinct set of default colors suitable for operators, which can be
    extended with user-provided colors.
    """
    def __init__(self, custom_colors: Optional[List[str]] = None) -> None:
        """
        Initializes the OperatorColorAssigner.

        Parameters
        ----------
        custom_colors : list[str], optional
            A list of custom hexadecimal color codes to use. These colors will
            be prioritized before the default operator colors are used.
            If None, only the default operator colors will be used.
        """
        if custom_colors is None:
            color_pool = self._DEFAULT_OPERATOR_COLORS
        else:
            # Combine custom colors with default colors.
            color_pool = list(custom_colors) + self._DEFAULT_OPERATOR_COLORS
            
        super().__init__(color_pool)