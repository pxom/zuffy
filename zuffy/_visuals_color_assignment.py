"""
Functions to manage the selection of colors for the display of a Fuzzy Pattern Tree.

This module provides a flexible way to assign consistent colors to different entities
(like features and operators) within a visualization, ensuring unique colors are
assigned sequentially from a predefined or custom pool.

"""

from typing import List, Dict, Optional


class ColorAssigner: # pylint: disable=too-few-public-methods
    """
    A base class for assigning cyclical colors to named objects (e.g., features, operators).

    This class manages a pool of colors and assigns a unique color from the pool
    to each new `object_name` it encounters. If an `object_name` has been seen
    before, it returns the previously assigned color. If the color pool is exhausted,
    it cycles back to the beginning of the pool.

    Subclasses are expected to provide their specific default color pools.
    """

    def __init__(self, color_pool: List[str]) -> None:
        """
        Initializes the ColorAssigner with a specified color pool.

        Parameters
        ----------
        color_pool : list of str
            A list of hexadecimal color codes to use as the color pool.
            This list must not be empty.

        Raises
        ------
        ValueError
            If the `color_pool` is empty.
        """
        if not color_pool:
            raise ValueError("`color_pool` cannot be empty.")

        self.colors: List[str] = color_pool
        # Maps object_name to its assigned index in self.colors to ensure consistency
        self.assigned_colors: Dict[str, int] = {}


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
            # `len(self.colors)` is guaranteed to be > 0 due to __init__ check.
            color_index = len(self.assigned_colors) % len(self.colors)
            self.assigned_colors[object_name] = color_index

        return self.colors[color_index]


class FeatureColorAssigner(ColorAssigner): # pylint: disable=too-few-public-methods
    """
    Manages color assignments specifically for feature objects.

    Uses a distinct set of default colors suitable for features, which can be
    extended with user-provided colors.
    """

    # Default hexadecimal color codes for features, typically strong, distinct colors.
    _DEFAULT_FEATURE_COLORS: List[str] = [
        '#4f77d4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    def __init__(self, custom_colors: Optional[List[str]] = None) -> None:
        """
        Initializes the FeatureColorAssigner.

        Parameters
        ----------
        custom_colors : list of str, optional
            A list of custom hexadecimal color codes to use. These colors will
            be prioritized before the default feature colors are used.
            If None, only the default feature colors will be used.
        """
        if custom_colors is None:
            color_pool = self._DEFAULT_FEATURE_COLORS
        else:
            # Combine custom colors with default colors, ensuring `custom_colors` come first.
            # `list(custom_colors)` creates a shallow copy to avoid modifying the original list.
            color_pool = list(custom_colors) + self._DEFAULT_FEATURE_COLORS

        super().__init__(color_pool)


class OperatorColorAssigner(ColorAssigner): # pylint: disable=too-few-public-methods
    """
    Manages color assignments specifically for operator objects.

    Uses a distinct set of default colors suitable for operators, which can be
    extended with user-provided colors.
    """
    # Default hexadecimal color codes for operators, typically pale pastels.
    _DEFAULT_OPERATOR_COLORS: List[str] = [
        '#f8ffef', '#f8ffff', '#fff8f8', '#fffff8',
        '#f8f8ff', '#eff8ff',
    ]

    def __init__(self, custom_colors: Optional[List[str]] = None) -> None:
        """
        Initializes the OperatorColorAssigner.

        Parameters
        ----------
        custom_colors : list of str, optional
            A list of custom hexadecimal color codes to use. These colors will
            be prioritized before the default operator colors are used.
            If None, only the default operator colors will be used.
        """
        if custom_colors is None:
            color_pool = self._DEFAULT_OPERATOR_COLORS
        else:
            # Combine custom colors with default colors, ensuring `custom_colors` come first.
            # `list(custom_colors)` creates a shallow copy to avoid modifying the original list.
            color_pool = list(custom_colors) + self._DEFAULT_OPERATOR_COLORS

        super().__init__(color_pool)
