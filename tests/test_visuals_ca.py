"""
Pytest script for the zuffy._visuals_color_assignment module.

This script tests the functionality of ColorAssigner, FeatureColorAssigner,
and OperatorColorAssigner classes, ensuring correct color assignment,
cycling, and error handling.
"""

import pytest
from zuffy._visuals_color_assignment import ColorAssigner, FeatureColorAssigner, OperatorColorAssigner
from typing import List, Dict, Optional

# --- Test Cases for ColorAssigner ---

def test_color_assigner_init_valid_pool():
    """Test ColorAssigner initialization with a valid color pool."""
    colors = ['#FF0000', '#00FF00']
    assigner = ColorAssigner(colors)
    assert assigner.colors == colors
    assert assigner.assigned_colors == {}

def test_color_assigner_init_empty_pool():
    """Test ColorAssigner initialization with an empty color pool raises ValueError."""
    with pytest.raises(ValueError, match="`color_pool` cannot be empty."):
        ColorAssigner([])

def test_color_assigner_get_color_new_objects():
    """Test ColorAssigner assigns new colors sequentially to new objects."""
    colors = ['#FF0000', '#00FF00', '#0000FF']
    assigner = ColorAssigner(colors)

    assert assigner.get_color("obj1") == '#FF0000'
    assert assigner.assigned_colors == {"obj1": 0}

    assert assigner.get_color("obj2") == '#00FF00'
    assert assigner.assigned_colors == {"obj1": 0, "obj2": 1}

    assert assigner.get_color("obj3") == '#0000FF'
    assert assigner.assigned_colors == {"obj1": 0, "obj2": 1, "obj3": 2}

def test_color_assigner_get_color_existing_objects():
    """Test ColorAssigner returns the same color for existing objects."""
    colors = ['#FF0000', '#00FF00']
    assigner = ColorAssigner(colors)

    assigner.get_color("obj1") # Assigns #FF0000
    assigner.get_color("obj2") # Assigns #00FF00

    assert assigner.get_color("obj1") == '#FF0000' # Should return previously assigned color
    assert assigner.get_color("obj2") == '#00FF00' # Should return previously assigned color

def test_color_assigner_get_color_cycles():
    """Test ColorAssigner cycles through the color pool when all colors are assigned."""
    colors = ['#FF0000', '#00FF00'] # Pool of 2 colors
    assigner = ColorAssigner(colors)

    assert assigner.get_color("obj1") == '#FF0000'
    assert assigner.get_color("obj2") == '#00FF00'
    assert assigner.get_color("obj3") == '#FF0000' # Should cycle back to first color
    assert assigner.get_color("obj4") == '#00FF00' # Should cycle back to second color
    assert assigner.get_color("obj1") == '#FF0000' # Existing object should still get its original color

def test_color_assigner_mixed_calls():
    """Test ColorAssigner with a mix of new and existing objects, including cycling."""
    colors = ['c1', 'c2', 'c3']
    assigner = ColorAssigner(colors)

    assert assigner.get_color('A') == 'c1'
    assert assigner.get_color('B') == 'c2'
    assert assigner.get_color('A') == 'c1'  # Repeated 'A'
    assert assigner.get_color('C') == 'c3'
    assert assigner.get_color('D') == 'c1'  # Cycle
    assert assigner.get_color('B') == 'c2'  # Repeated 'B'
    assert assigner.get_color('E') == 'c2'  # Cycle again


# --- Test Cases for FeatureColorAssigner ---

def test_feature_color_assigner_init_default():
    """Test FeatureColorAssigner initialization with default colors."""
    assigner = FeatureColorAssigner()
    assert assigner.colors == FeatureColorAssigner._DEFAULT_FEATURE_COLORS
    assert assigner.assigned_colors == {}

def test_feature_color_assigner_init_custom_colors():
    """Test FeatureColorAssigner initialization with custom colors."""
    custom = ['#AAA', '#BBB']
    assigner = FeatureColorAssigner(custom_colors=custom)
    expected_colors = custom + FeatureColorAssigner._DEFAULT_FEATURE_COLORS
    assert assigner.colors == expected_colors
    assert assigner.assigned_colors == {}

def test_feature_color_assigner_get_color_behavior():
    """Test FeatureColorAssigner's get_color behavior (new, existing, cycling)."""
    assigner = FeatureColorAssigner(custom_colors=['#CUSTOM1', '#CUSTOM2'])
    
    # Custom colors used first
    assert assigner.get_color("feat1") == '#CUSTOM1'
    assert assigner.get_color("feat2") == '#CUSTOM2'
    
    # Default colors used next
    assert assigner.get_color("feat3") == FeatureColorAssigner._DEFAULT_FEATURE_COLORS[0]
    assert assigner.get_color("feat4") == FeatureColorAssigner._DEFAULT_FEATURE_COLORS[1]

    # Existing object should get its original color
    assert assigner.get_color("feat1") == '#CUSTOM1'

    # Cycling behavior (should cycle through custom then default)
    # Get enough unique features to exceed combined pool size for full cycle test
    assigner = FeatureColorAssigner(custom_colors=['C1', 'C2'])
    all_expected_colors = ['C1', 'C2'] + FeatureColorAssigner._DEFAULT_FEATURE_COLORS

    for i in range(len(all_expected_colors) + 3): # Go beyond the pool size
        obj_name = f"obj_{i}"
        expected_color = all_expected_colors[i % len(all_expected_colors)]
        assert assigner.get_color(obj_name) == expected_color

    # Test re-getting a very early object after cycling
    assert assigner.get_color("obj_0") == 'C1'


# --- Test Cases for OperatorColorAssigner ---

def test_operator_color_assigner_init_default():
    """Test OperatorColorAssigner initialization with default colors."""
    assigner = OperatorColorAssigner()
    assert assigner.colors == OperatorColorAssigner._DEFAULT_OPERATOR_COLORS
    assert assigner.assigned_colors == {}

def test_operator_color_assigner_init_custom_colors():
    """Test OperatorColorAssigner initialization with custom colors."""
    custom = ['#CCC', '#DDD']
    assigner = OperatorColorAssigner(custom_colors=custom)
    expected_colors = custom + OperatorColorAssigner._DEFAULT_OPERATOR_COLORS
    assert assigner.colors == expected_colors
    assert assigner.assigned_colors == {}

def test_operator_color_assigner_get_color_behavior():
    """Test OperatorColorAssigner's get_color behavior (new, existing, cycling)."""
    assigner = OperatorColorAssigner(custom_colors=['#OP_CUSTOM1', '#OP_CUSTOM2'])
    
    # Custom colors used first
    assert assigner.get_color("add") == '#OP_CUSTOM1'
    assert assigner.get_color("sub") == '#OP_CUSTOM2'
    
    # Default colors used next
    assert assigner.get_color("mul") == OperatorColorAssigner._DEFAULT_OPERATOR_COLORS[0]
    assert assigner.get_color("div") == OperatorColorAssigner._DEFAULT_OPERATOR_COLORS[1]

    # Existing object should get its original color
    assert assigner.get_color("add") == '#OP_CUSTOM1'

    # Cycling behavior
    assigner = OperatorColorAssigner(custom_colors=['O1', 'O2'])
    all_expected_colors = ['O1', 'O2'] + OperatorColorAssigner._DEFAULT_OPERATOR_COLORS

    for i in range(len(all_expected_colors) + 3):
        op_name = f"op_{i}"
        expected_color = all_expected_colors[i % len(all_expected_colors)]
        assert assigner.get_color(op_name) == expected_color

    assert assigner.get_color("op_0") == 'O1'
