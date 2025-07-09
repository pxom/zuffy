"""
Functions for visualizing Fuzzy Pattern Trees (FPTs) and related models.

This module provides utilities to:
- Generate Graphviz DOT scripts for FPT models.
- Plot the evolutionary metrics of the GP sub-estimators.
- Calculate and display permutation feature importances.
- Plot the performance of the experiments (iterations)

"""

import html
import numbers
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import graphviz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.inspection import permutation_importance
from sklearn.utils._param_validation import HasMethods, Interval, validate_params

from gplearn.functions import _Function

from zuffy._visuals_color_assignment import FeatureColorAssigner, OperatorColorAssigner

# Constants for graphviz_tree function
DEFAULT_GRAPH_FONT = "Helvetica"
DEFAULT_GRAPH_FONT_SIZE = "22"
DEFAULT_ROOT_TEXT = "WTA"
DEFAULT_OUTPUT_FILENAME = "zuffy_output"
DEFAULT_CONSTANT_COLOR = '#D3D3D3' # Light grey for constant nodes
# Use a large scale to ensure unique node IDs across multiple sub-trees within the same graph.
NODE_ID_OFFSET = 1000

def _add_importance(feature_metrics: List[Union[float, int]]) -> str:
    """
    Generates an HTML string to display feature importance metrics.

    This function formats the mean importance, standard deviation, and rank
    of a feature into an HTML table row suitable for embedding in Graphviz
    node labels.

    Parameters
    ----------
    feature_metrics : list of float or int, length 3
        A list containing exactly three elements:
        `[mean_importance, std_dev_importance, rank]`.
        `mean_importance` and `std_dev_importance` should be numeric (float/int),
        and `rank` should be an integer.

    Returns
    -------
    extra_html : str
        An HTML string representing the feature's importance metrics,
        formatted as a table row to be inserted into a Graphviz HTML-like label.

    Raises
    ------
    ValueError
        If the input `feature_metrics` list does not contain exactly three elements
        or if the mean importance is not numeric.
    """
    if not isinstance(feature_metrics, list) or len(feature_metrics) != 3:
        raise ValueError(f"The `feature_metrics` list must contain exactly three elements "
                         f"(mean, stdev, rank) for this feature: {feature_metrics}.")

    f_mean, f_std, f_rank = feature_metrics
    if not isinstance(f_mean, (float, int)):
        raise ValueError(f"The mean importance value must be numeric, but found: "
                         f"{f_mean} (type: {type(f_mean)}).")

    # Use f-strings for direct rounding and HTML entity for ±
    extra_html = (
        f'<tr><td bgColor="white" color="red">'
        f'<font point-size="14"><b>{f_rank}</b>: {f_mean:.3f} &plusmn; {f_std:.3f}</font>'
        f'</td></tr>'
    )
    return extra_html

def _output_node(i: int, node: Union[int, float], feature_names: Optional[List[str]],
                 feature_color_assigner: FeatureColorAssigner,
                 imp_feat: Optional[Dict[str, List[Union[float, int]]]] = None) -> str:
    """
    Generates the Graphviz DOT string for a single node in the tree.

    This function distinguishes between feature (terminal) nodes, which are
    integers corresponding to feature indices, and constant (terminal) nodes,
    which are floats. It assigns colors and optionally includes importance metrics.

    Parameters
    ----------
    i : int
        The unique integer identifier for the node within the Graphviz DOT script.
    node : int or float
        The value of the node. If an `int`, it represents a feature index.
        If a `float`, it represents a constant value.
    feature_names : list of str or None
        A list of string names for the input features. If `None`, generic
        'X0', 'X1', etc., names will be used for feature nodes.
    feature_color_assigner : FeatureColorAssigner
        An instance of `FeatureColorAssigner` used to retrieve specific colors
        for feature nodes based on their names.
    imp_feat : dict, optional
        A dictionary where keys are feature names and values are lists of
        importance metrics `[mean_importance, std_dev_importance, rank]`.
        If provided, importance information will be added to feature nodes.
        Defaults to `None`.

    Returns
    -------
    dot_string : str
        The Graphviz DOT string defining the node's properties (label, shape, color).

    Raises
    ------
    ValueError
        If `node` is an integer (feature index) but is out of bounds for the
        provided `feature_names` list.
    """
    dot_string = ''
    if isinstance(node, int): # This indicates a feature node (terminal)
        if feature_names is None:
            feature_name = f'X{node}'
        else:
            if node < len(feature_names):
                feature_name = feature_names[node]
            else:
                raise ValueError(
                    f"Feature index {node} is out of bounds for the provided "
                    f"feature names (expected index < {len(feature_names)})."
                )

        # Get color for the feature node using the assigner
        fill_color = feature_color_assigner.get_color(feature_name)

        # Optionally add feature importance metrics
        extra_html = ""
        if imp_feat and feature_name in imp_feat:
            # _add_importance returns an HTML string for a table row
            extra_html = _add_importance(imp_feat[feature_name])

        # Escape HTML special characters in the feature name and replace '|' with <br/>
        # for proper display in Graphviz HTML-like labels.
        display_feature_name = html.escape(feature_name).replace('|', '<br/>')

        dot_string += (
            f'{i} [label=<\n'
            f'  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" '
            f'   bgColor="{fill_color}">\n'
            f'    <tr><td>{display_feature_name}</td></tr>\n'
            f'    {extra_html}\n' # Empty string if no importance, otherwise a <tr>
            f'  </table>>,\n'
            f'  color="black", shape=none\n'
            f'] ;\n'
        )
    else: # This indicates a constant node (float terminal)
        fill_color = DEFAULT_CONSTANT_COLOR
        dot_string += (f'{i} [label="{node:.3f}", style=filled, fillcolor="{fill_color}"] ;\n')
    return dot_string

def sanitise_names(names: Optional[List[Any]]) -> Optional[List[str]]:
    """
    Sanitises a list of names for safe use in HTML contexts.

    Each element in the input list is converted to a string, and then HTML
    special characters are escaped to prevent rendering issues in Graphviz
    labels.

    Parameters
    ----------
    names : list of Any or None
        A list of names (can be any object convertible to string).
        If `None`, the function returns `None`.

    Returns
    -------
    sanitised_list : list of str or None
        A new list with sanitised string versions of the names.
        Returns `None` if the input `names` was `None`.
    """
    if names is None:
        return None

    sanitised_list = []
    for name in names:
        # Convert to string and then escape HTML special characters
        sanitised_list.append(html.escape(str(name)))
    return sanitised_list

@validate_params(
    {
        #"program": [HasMethods(["_program", "n_features"])], # More specific methods check
        "feature_names": [None, list],
        "start": [Interval(numbers.Integral, 0, None, closed='left')],
        "operator_col_fn": [None, HasMethods(["get_color"])],
        "feature_col_fn": [None, HasMethods(["get_color"])],
        "imp_feat": [None, dict],
    },
    prefer_skip_nested_validation=True
)
def graph_tree_class(program, feature_names: Optional[List[str]] = None, start: int = 0,
                    operator_col_fn: Optional[OperatorColorAssigner] = None,
                    feature_col_fn: Optional[FeatureColorAssigner] = None,
                    imp_feat: Optional[Dict[str, List[Union[float, int]]]] = None) -> str:
    """
    Returns a string Graphviz script for visualizing a genetic programming program
    representing a class of a Fuzzy Pattern Tree.

    This function generates a DOT language script that can be rendered by Graphviz
    to produce a visual representation of the class for inclusion in a FPT. It 
    handles operators, features (terminals), and can optionally display feature 
    importances.

    Parameters
    ----------
    program : object
        The genetic programming program object (e.g., from `gplearn`'s `_Program`).
        Expected to have a `program` attribute (the FPT structure as a list of nodes)
        and an `n_features` attribute (number of input features).
    feature_names : list of str, optional
        A list of string names for the features used in the program. If `None`,
        generic 'X0', 'X1', etc., names are used. Defaults to `None`.
    start : int, optional
        An integer offset to add to node indices. This is useful when combining
        multiple sub-graphs into a larger Graphviz visualization to ensure unique
        node IDs. Defaults to 0.
    operator_col_fn : OperatorColorAssigner, optional
        An instance of `OperatorColorAssigner` to determine colors for operator nodes.
        If `None`, a default `OperatorColorAssigner` instance will be used.
    feature_col_fn : FeatureColorAssigner, optional
        An instance of `FeatureColorAssigner` to determine colors for feature nodes.
        If `None`, a default `FeatureColorAssigner` instance will be used.
    imp_feat : dict, optional
        A dictionary where keys are feature names (strings) and values are lists of
        importance metrics `[mean, std, rank]` for displaying feature importance.
        If provided, importance information will be rendered on feature nodes.
        Defaults to `None`.

    Returns
    -------
    dot_script : str
        The Graphviz DOT script string representing the program's tree structure.

    Raises
    ------
    ValueError
        If `feature_names` is provided but its length is less than `program.n_features`,
        indicating insufficient names for all features.
    """
    if feature_names and len(feature_names) < program.n_features:
        raise ValueError(f'There are insufficient feature_names ({len(feature_names)}) '
                         f'supplied for this program (which expects {program._n_features}).')

    # Initialise the color assigners, if not provided
    operator_color_assigner = operator_col_fn if operator_col_fn is not None \
                                                else OperatorColorAssigner()
    feature_color_assigner = feature_col_fn if feature_col_fn is not None \
                                                else FeatureColorAssigner()

    # Stack to manage operator arity and children, for building edges.
    # Each element: [remaining_arity, parent_id, child1_id, child2_id, ...]
    terminals_stack: List[List[Union[int, Any]]] = []

    dot_output = ''

    for i, node in enumerate(program.program):
        current_node_id = i + start

        if isinstance(node, _Function):
            # This is an operator node containing a Function
            terminals_stack.append([node.arity, current_node_id]) # Add operator to stack
            node_fill_color = operator_color_assigner.get_color(node.name)
            dot_output += (f'{current_node_id} [label="{node.name}", style=filled, '
                           f'fillcolor="{node_fill_color}", color="#999999"] ;\n')
        else: # This is a terminal node containing a Feature (or a constant)
            # _output_node handles feature/constant specific styling and labels
            dot_output += _output_node(current_node_id, node, feature_names,
                                       feature_color_assigner, imp_feat)

            # Handle the degenerative case where the program is a single node (a terminal)
            if i == 0 and not terminals_stack:
                return dot_output # No edges to draw for a single-node program

            # Append the current node as a child to the last operator on the stack.
            if terminals_stack:
                terminals_stack[-1][0] -= 1 # Decrement arity count for the parent operator
                terminals_stack[-1].append(current_node_id) # Add current node as a child

            # Process completed operators: when an operator's arity count reaches 0,
            # all its children have been processed, and its edges can be drawn.
            while terminals_stack and terminals_stack[-1][0] == 0:
                parent_id = terminals_stack[-1][1]
                # Children are from index 2 onwards in the `terminals_stack` sublist.
                children_ids = terminals_stack[-1][2:]
                for child_id in children_ids:
                    dot_output += f'{parent_id} -> {child_id} ;\n'

                terminals_stack.pop() # Remove the completed operator from the stack.

                # If there's a parent operator for the just-completed sub-tree (i.e., not root)
                if terminals_stack:
                    # The just-completed operator (parent_id) now becomes a child of its parent.
                    terminals_stack[-1].append(parent_id)
                    terminals_stack[-1][0] -= 1 # Decrement arity of its parent.

    # This part should ideally be unreachable if the program structure is a valid tree.
    # It might indicate an incomplete tree or an issue in the traversal logic.
    return dot_output

@validate_params(
    {
        "target_feature_name": [str],
        "target_class_names": [None, list],
        "feature_names": [None, list],
        "tree_name": [None, str],
        "imp_feat": [None, dict],
        "output_filename": [str],
        "source_filename": [None, str],
        "bg_color": [str],
        "root_bg_color": [str],
        "root_text": [str],
        "feat_color_list": [None, list],
        "oper_color_list": [None, list],
        "show_fitness": [bool],
    },
    prefer_skip_nested_validation=True
)
def graphviz_tree(
        model: Any,
        target_feature_name: str = 'Target',
        target_class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        tree_name: Optional[str] = None,
        imp_feat: Optional[Dict[str, List[Union[float, int]]]] = None,
        output_filename: str = DEFAULT_OUTPUT_FILENAME,
        source_filename: Optional[str] = None,
        bg_color: str = 'white',
        root_bg_color: str = '#ddddff',
        root_text: str = DEFAULT_ROOT_TEXT,
        feat_color_list: Optional[List[str]] = None,
        oper_color_list: Optional[List[str]] = None,
        show_fitness: bool = False
) -> Tuple[str, graphviz.Source]:
    """
    Generates a Graphviz DOT script and renders an image for a Zuffy Fuzzy Pattern Tree model.

    This function visualises the entire Zuffy model, which typically consists of
    multiple sub-estimators (one tree for each target class) and a central
    root node that represents the Winner-Takes-All (WTA) mechanism linking them.

    Parameters
    ----------
    model : object
        The Zuffy Fuzzy Pattern Tree model object. Expected to have the
        `multi_.estimators_` attribute (a list of individual `gplearn._Program`
        or similar sub-estimators) and a `classes_` attribute (for target labels).
    target_feature_name : str, optional
        The name to display for the target feature in the root (WTA) node.
        Defaults to 'Target'.
    target_class_names : list of str, optional
        A list of string names for each target class. If `None`, `model.classes_`
        will be used. If `model.classes_` is also unavailable, generic names like
        'Class 0', 'Class 1' will be generated. Defaults to `None`.
    feature_names : list of str, optional
        A list of string names for the input features. If `None`, the function
        attempts to retrieve them from `model.multi_.estimator.feature_names_in_`
        (scikit-learn standard) or `model.multi_.estimator.feature_names`.
        Defaults to `None`.
    tree_name : str, optional
        An overall title for the entire Graphviz tree plot. If `None`, no title
        will be displayed. Defaults to `None`.
    imp_feat : dict, optional
        A dictionary of feature importances. Keys are feature names (strings)
        and values are lists of importance metrics `[mean, std, rank]`. If
        provided, this information will be rendered on the corresponding
        feature nodes. Defaults to `None`.
    output_filename : str, optional
        The base filename for the output image file (e.g., 'my_model' will
        generate 'my_model.png'). Defaults to `DEFAULT_OUTPUT_FILENAME` ('zuffy_output').
    source_filename : str, optional
        If provided, the generated Graphviz DOT script will also be written
        to this file. Defaults to `None`.
    bg_color : str, optional
        The background color for the entire graph. Can be a color name (e.g., 'white')
        or a hexadecimal color code. Defaults to 'white'.
    root_bg_color : str, optional
        The background color for the central root (WTA) node. Defaults to 'grey'.
    root_text : str, optional
        The main text label to display within the central root (WTA) node.
        Defaults to `DEFAULT_ROOT_TEXT` ('WTA').
    feat_color_list : list of str, optional
        A list of hexadecimal color codes to be used as a custom color pool for
        `FeatureColorAssigner`. These colors will be prioritised. If `None`,
        default feature colors are used. Defaults to `None`.
    oper_color_list : list of str, optional
        A list of hexadecimal color codes to be used as a custom color pool for
        `OperatorColorAssigner`. These colors will be prioritised. If `None`,
        default operator colors are used. Defaults to `None`.
    show_fitness : bool, optional
        If `True`, the raw fitness value of each sub-estimator will be displayed
        alongside its class name in the root node. Defaults to `False`.

    Returns
    -------
    dot_script : str
        The complete Graphviz DOT script string that defines the visualization.
    graph : graphviz.Source
        A `graphviz.Source` object, which can be directly rendered or displayed.

    Raises
    ------
    AttributeError
        If the `model` object does not have the expected `multi_` or
        `multi_.estimators_` attributes.
    ValueError
        If `target_class_names` is provided but its length is insufficient to
        represent all the sub-estimators (classes) in the model.
    """

    # Basic model structure validation for essential attributes.
    if not hasattr(model, 'multi_') or not hasattr(model.multi_, 'estimators_'):
        raise AttributeError(
            "The model must have a 'multi_' attribute, which in turn must have "
            "an 'estimators_' attribute (e.g., `model.multi_.estimators_`)."
        )

    # Initialise the color assigners
    operator_color_assigner = OperatorColorAssigner(oper_color_list)
    feature_color_assigner = FeatureColorAssigner(feat_color_list)

    # Determine feature names for the graph. Prioritise explicitly provided names.
    if feature_names is None:
        # Check for standard scikit-learn 'feature_names_in_' attribute on the inner estimator.
        if hasattr(model.multi_.estimators_[0], 'feature_names_in_') and \
           model.multi_.estimators_[0].feature_names_in_ is not None:
            feature_names = model.multi_.estimators_[0].feature_names_in_.tolist()
        # Fallback to older/non-standard 'feature_names' attribute.
        elif hasattr(model.multi_.estimators_[0], 'feature_names') and \
             model.multi_.estimators_[0].feature_names is not None:
            feature_names = model.multi_.estimators_[0].feature_names
        # If no feature names found, `feature_names` remains None, and
        # _output_node will use generic names.
    feature_names = sanitise_names(feature_names) # Sanitise for HTML display

    # Determine target class names. Prioritise explicitly provided names.
    if target_class_names is None:
        if hasattr(model, 'classes_') and model.classes_ is not None:
            target_class_names = list(model.classes_)
        else:
            # Fallback if classes_ attribute is not present on the model.
            target_class_names = [f'Class {i}' for i in range(len(model.multi_.estimators_))]

    # Ensure provided target_class_names are sufficient and sanitise them.
    if len(target_class_names) < len(model.multi_.estimators_):
        raise ValueError(f'Insufficient `target_class_names` ({len(target_class_names)}) supplied'
                         f' to represent each of the {len(model.multi_.estimators_)} classes.')
    target_class_names = sanitise_names(target_class_names)

    wta_id = (len(model.multi_.estimators_) + 1) * NODE_ID_OFFSET # Unique ID for the WTA root node.

    # Build the DOT script incrementally.
    dot_script_parts = [
        'digraph G {',
        f'bgcolor="{bg_color}"',
        f'fontname="{DEFAULT_GRAPH_FONT}"',
        f'fontsize="{DEFAULT_GRAPH_FONT_SIZE}"',
        f'node [fontname="{DEFAULT_GRAPH_FONT}"]',
    ]

    if tree_name:
        dot_script_parts.append(f'label="{html.escape(str(tree_name))}"')
        dot_script_parts.append('labelloc = t') # Place label at the top.

    wta_edges: List[str] = [] # Edges from WTA root to each sub-tree.
    wta_ports_html: List[str] = [] # HTML for ports in the WTA root node.

    for idx, estimator in enumerate(model.multi_.estimators_):
        # Each estimator in `multi_.estimators_` is expected to be a gplearn _Program
        # or an object wrapping it, providing a `_program` attribute.
        if not hasattr(estimator, "_program"):
            raise ValueError(
                "Each estimator in `model.multi_.estimators_` is expected to have "
                "a `_program` attribute (e.g., the gplearn _Program object) "
                "to build its tree visualization, but it is missing for estimator "
                f"at index {idx} (class: {target_class_names[idx]})."
            )

        # Generate DOT for each sub-estimator. Pass color assigners for consistency.
        dot_script_parts.append(graph_tree_class(
            estimator._program,
            start=idx * NODE_ID_OFFSET, # Offset node IDs for this sub-tree.
            feature_names=feature_names,
            operator_col_fn=operator_color_assigner,
            feature_col_fn=feature_color_assigner,
            imp_feat=imp_feat
        ))

        # Add edge from WTA root node to the root of the current sub-tree.
        wta_edges.append(f'{wta_id}:port_{idx} -> {idx * NODE_ID_OFFSET};')

        # Add fitness information to the WTA port label if `show_fitness` is True.
        fitness_info = ""
        if show_fitness and hasattr(estimator._program, 'raw_fitness_'):
            fitness_info = f" ({estimator._program.raw_fitness_:.3f})"

        # Sanitise class names for display in the HTML label of the WTA node.
        sanitised_target_class = html.escape(target_class_names[idx])
        wta_ports_html.append(f"<td port=\"port_{idx}\">{html.escape(target_feature_name)}="
                              f"{sanitised_target_class}{fitness_info}</td>")

    # Define the central root node (WTA - Winner Takes All).
    # It uses an HTML-like label for complex layout with ports.
    dot_script_parts.append(
        f'{wta_id} [label=<<table border="1" cellborder="1" bgcolor="{root_bg_color}">'
        f'<tr><td colspan="{len(model.multi_.estimators_)}">{root_text}</td></tr>'
        f'<tr>{"".join(wta_ports_html)}</tr></table>>, '
        f'color="black", shape=plaintext, width=4, fontname="{DEFAULT_GRAPH_FONT}"] ;\n'
    )
    dot_script_parts.extend(wta_edges) # Add all WTA-to-sub-tree edges.
    dot_script_parts.append('}') # Close the main digraph.

    full_dot_script = "\n".join(dot_script_parts)
    graph = graphviz.Source(full_dot_script)

    # Render the graph to an image file.
    # `view=False` to prevent opening a viewer, `cleanup=True` to remove intermediate files.
    graph.render(output_filename, format='png', view=False, cleanup=True)

    if source_filename is not None:
        # Write the raw DOT script to a specified file for debugging/review.
        with open(source_filename, 'w', encoding="utf-8") as file:
            file.write(full_dot_script)

    return full_dot_script, graph

@validate_params(
    {
        "target_class_names": [None, list],
        "output_filename": [None, str]
    },
    prefer_skip_nested_validation=True
)
def plot_evolution(model: Any, target_class_names: Optional[List[str]] = None,
                   output_filename: Optional[str] = None) -> None:
    """
    Plots the evolution metrics (tree length, fitness, generation duration) for
    each sub-estimator in a Fuzzy Pattern Tree model over generations.

    This function generates a multi-panel plot, with each row representing a
    target class's evolutionary progress. It visualises average and best
    tree lengths, fitness values, and generation times across generations.

    Parameters
    ----------
    model : object
        The genetic programming model object (e.g., from `gplearn`'s
        `SymbolicClassifier` or `SymbolicRegressor`). Expected to have
        `multi_.estimators_` (a list of sub-estimators) and `classes_` attributes.
        Each estimator within `multi_.estimators_` must also have a
        `run_details_` attribute, containing 'generation', 'average_length',
        'best_length', 'average_fitness', 'best_fitness', and 'generation_time' lists.
    target_class_names : list of str, optional
        A list of string names for each target class. If `None`, `model.classes_`
        will be used. Defaults to `None`.
    output_filename : str, optional
        The full path and filename (e.g., 'evolution_plot.png') to save the plot.
        If `None`, the plot will be displayed interactively. Defaults to `None`.

    Returns
    -------
    None
        The function displays or saves a Matplotlib plot.

    Raises
    ------
    AttributeError
        If `model.multi_` or `model.multi_.estimators_` attributes are missing.
        Also, if any individual estimator in `multi_.estimators_` lacks the
        `run_details_` attribute.
    ValueError
        If `target_class_names` is provided but its length is insufficient to
        represent each model class.
    """
    if not hasattr(model, 'multi_') or not hasattr(model.multi_, 'estimators_'):
        raise AttributeError(
            "The model must have a 'multi_' attribute, which in turn must have "
            "an 'estimators_' attribute (e.g., `model.multi_.estimators_`)."
        )

    # Determine target class names (similar logic to graphviz_tree for consistency).
    if target_class_names is None:
        if hasattr(model, 'classes_'):
            target_class_names = list(model.classes_)
        else:
            target_class_names = [f'Class {i}' for i in range(len(model.multi_.estimators_))]

    if len(target_class_names) < len(model.multi_.estimators_):
        raise ValueError(f'Insufficient `target_class_names` ({len(target_class_names)}) supplied'
                         f' to represent each of the {len(model.multi_.estimators_)} classes.')
    target_class_names = sanitise_names(target_class_names) # Sanitise for plot titles.

    num_estimators = len(model.multi_.estimators_)
    # Create a figure with dynamic height based on the number of estimators.
    fig = plt.figure(figsize=(11, 2.5 * num_estimators))

    num_cols = 3 # Three subplots per row (Length, Fitness, Duration).

    for idx, estimator in enumerate(model.multi_.estimators_):
        # Ensure `run_details_` is available on each sub-estimator for plotting.
        if not hasattr(estimator, 'run_details_'):
            raise AttributeError(
                f"Estimator for class '{target_class_names[idx]}' is missing "
                "'run_details_' attribute. Ensure the gplearn model was fitted with "
                "`verbose=1` or `low_memory=False` to store run details."
            )

        run_details = estimator.run_details_

        # Plot 1: Tree Length Evolution
        ax1 = fig.add_subplot(num_estimators, num_cols, idx * num_cols + 1)
        ax1.set_title(f'Class: {target_class_names[idx]}\nTree Length '
                      f'(Final Avg: {run_details["average_length"][-1]:.2f})')
        ax1.plot(run_details['generation'], run_details['average_length'],
                 color='tab:blue', label='Average')
        ax1.plot(run_details['generation'], run_details['best_length'],
                 color='tab:orange', label='Best')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Length')
        ax1.legend()

        # Plot 2: Fitness Evolution (smaller is better)
        ax2 = fig.add_subplot(num_estimators, num_cols, idx * num_cols + 2)
        ax2.set_title(f'Fitness (smaller is better)\n'
                      f'Final Best: {run_details["best_fitness"][-1]:.3f}')
        ax2.plot(run_details['generation'], run_details['average_fitness'],
                 color='tab:purple', label='Average')
        ax2.plot(run_details['generation'], run_details['best_fitness'],
                 color='tab:green', label='Best')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.legend()

        # Plot 3: Generation Duration
        ax3 = fig.add_subplot(num_estimators, num_cols, idx * num_cols + 3)
        # Calculate average generation time if data is available, otherwise handle gracefully.
        if run_details['generation_time']:
            avg_gen_time = sum(run_details['generation_time']) / len(run_details['generation_time'])
        else:
            avg_gen_time = 0.0 # Default if no timing data
        ax3.set_title(f'Generation Duration\nAverage: {avg_gen_time:.4f}s')
        ax3.plot(run_details['generation'], run_details['generation_time'], color='#ffcc33')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Duration (s)')

    plt.tight_layout() # Adjust subplot parameters for a tight layout.

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory.

@validate_params(
    {
        "reg": [HasMethods("fit")],
        "X_test": ["array-like"],
        "y_test": ["array-like"],
        "features": [None, list],
        "output_filename": [None, str],
        "n_jobs": [None, numbers.Integral],
        "n_repeats": [numbers.Integral, Interval(numbers.Integral, 1, None, closed='left')],
    },
    prefer_skip_nested_validation=True
)
def show_feature_importance(reg: Any, X_test: np.ndarray, y_test: np.ndarray,
                            features: Optional[List[str]] = None,
                            n_jobs: Optional[int] = None, n_repeats: int = 20,
                            output_filename: Optional[str] = None
                            ) -> Dict[str, List[Union[float, int]]]:
    """
    Calculates and displays permutation feature importances for a given model.

    This function computes feature importances using the permutation importance method
    from `sklearn.inspection` and then plots a bar chart of these importances.
    It returns a dictionary containing the detailed importance metrics.

    Parameters
    ----------
    reg : object
        The regressor model object. It must be a fitted scikit-learn compatible
        regressor (or classifier that has a `score` method for the relevant metric)
        with `fit`, `predict`, and `score` methods.
    X_test : numpy.ndarray
        The test input data, used to evaluate the model's performance during permutation.
    y_test : numpy.ndarray
        The true target values for the test data.
    features : list of str, optional
        A list of string names for the input features. If `None`, the function
        attempts to retrieve names from `reg.multi_.estimators_[0].feature_names_in_`
        or `reg.multi_.estimators_[0].feature_names`. If still `None`, generic
        'X0', 'X1', etc., names will be used. Defaults to `None`.
    n_jobs : int, optional
        Number of jobs to run in parallel for `permutation_importance`.
        `-1` means using all available processors. `None` (default) means `1` job.
    n_repeats : int, optional
        The number of times to permute a feature. Higher values increase reliability
        but also computation time. Defaults to 20.
    output_filename : str, optional
        The full path and filename (e.g., 'feature_importance.png') to save the plot.
        If `None`, the plot will be displayed interactively. Defaults to `None`.

    Returns
    -------
    imp_feat_dict : dict
        A dictionary containing feature importances. Keys are original feature names
        (strings), and values are lists: `[mean_importance, std_dev_importance, rank]`.
        Only features with a positive mean importance (or non-zero standard deviation
        if mean is zero) are included.
    """
    # Attempt to get feature names from the model if not explicitly provided
    if features is None:
        if hasattr(reg, 'multi_') and hasattr(reg.multi_, 'estimators_') and reg.multi_.estimators_:
            # Try to get from the first estimator if it exists
            first_estimator = reg.multi_.estimators_[0]
            if hasattr(first_estimator, 'feature_names'): # gplearn standard
                features = first_estimator.feature_names
            elif hasattr(first_estimator, 'feature_names_in_'): # scikit-learn standard
                features = first_estimator.feature_names_in_.tolist()

    if features is None:
        # Fallback if feature names are not provided and not found in the model
        features = [f'X{i}' for i in range(X_test.shape[1])]

    # Sanitise feature names for use as plot labels
    sanitised_features = sanitise_names(features)

    print(f"Calculating permutation importances with {n_repeats} repeats and "
          f"{n_jobs if n_jobs is not None else 1} jobs...")
    start_time = time.time()
    # `permutation_importance` requires a fitted estimator and test data
    result = permutation_importance(reg, X_test, y_test, n_repeats=n_repeats, n_jobs=n_jobs)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    print('\n*** FEATURE IMPORTANCES (Permutation) ***')
    imp_feat_dict: Dict[str, List[Union[float, int]]] = {}
    imp_graph_names: List[str] = []
    imp_graph_values: List[float] = []

    # Sort features by importance mean in descending order
    sorted_indices = result.importances_mean.argsort()[::-1]

    rank = 1
    for i in sorted_indices:
        mean_importance = result.importances_mean[i]
        std_importance = result.importances_std[i]

        # Only include features with positive mean importance or non-zero standard deviation
        # if mean is zero, to avoid clutter from truly unimportant features.
        if mean_importance > 0 or (mean_importance == 0 and std_importance > 0):
            original_feature_name = features[i] # Use original for internal dict keys
            display_feature_name = sanitised_features[i] # Use sanitised for graph labels

            imp_feat_dict[original_feature_name] = [mean_importance, std_importance, rank]
            imp_graph_names.append(display_feature_name)
            imp_graph_values.append(mean_importance)
            print(f"Rank {rank}: {display_feature_name:<40} {mean_importance:.3f} "
                  f"+/- {std_importance:.3f}")
            rank += 1

    if not imp_graph_names:
        print("No features with non-zero importance found.")
        return imp_feat_dict # Return empty if no important features

    # Plot permutation feature importances
    fig, ax = plt.subplots()
    ax.bar(imp_graph_names, imp_graph_values, color='#ffcc33')
    ax.set_title(f"Feature Importances Using {n_repeats} permutations")
    ax.set_ylabel("Mean Accuracy Decrease")
    ax.set_xlabel("Fuzzy Features")
    plt.setp(ax.get_xticklabels(), rotation=20, horizontalalignment='right') # , fontsize='small')
    fig.tight_layout()

    if output_filename:
        plt.savefig(output_filename)
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory

    return imp_feat_dict

@validate_params(
    {
        "iter_perf": ["array-like"],
        "best_iter": [None, int],
        "title": [None, str],
        "output_filename": [None, str],
        "col_iter_acc": [None, str],
        "col_best_iter": [None, str],
        "col_tree_size": [None, str],
    },
    prefer_skip_nested_validation=True
)
def plot_iteration_performance(iter_perf: np.ndarray, best_iter: Optional[int] = None,
                title: Optional[str] = "Iteration Performance",
                output_filename: Optional[str] = None, col_iter_acc: Optional[str] = "#1c9fea",
                col_best_iter: Optional[str] = "#386938",
                col_tree_size: Optional[str] = "#96a5f7") -> None:
    """
    tbd
    """

    # decode iteration_performance_list.append([score, tree_size, class_scores])
    y1 = []
    y2 = []
    for i in iter_perf:
        y1.append(i[0])
        y2.append(i[1])

    # Calculate standard deviation
    std_dev_y1 = np.std(y1)
    std_dev_y2 = np.std(y2)

    #x = np.arange(1, it_params['n_iter']+1)
    x = np.arange(1, len(iter_perf)+1)

    fig, ax1 = plt.subplots()

    # Plotting the first value as a bar chart
    bars = ax1.bar(x, y1, color=col_iter_acc)

    # Change the color of the score_best_iter bar
    if best_iter is not None:
        bars[best_iter].set_color(col_best_iter)

    plt.xticks(fontsize=8)  # Set x-axis label font size to 8
    plt.yticks(fontsize=8)  # Set y-axis label font size to 8
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(f'Accuracy (std dev={std_dev_y1:.3f})', color=col_iter_acc)

    # Adjust the y-axis limits to the min and max of y1
    ax1.set_ylim([min(y1)-0.03, max(y1)+0.03])

    # Set x-axis ticks to show only integer values
    ax1.set_xticks(x)

    # Add horizontal line for mean of y1 values
    mean_y1 = np.mean(y1)
    ax1.axhline(mean_y1, color=col_iter_acc, linestyle='--', linewidth=1,
                label=f'Mean of Accuracy: {mean_y1:.2f}')

    # Creating a secondary y-axis for the second value
    ax2 = ax1.twinx()

    # Adjust the y-axis limits to the min and max of y2
    ax2.set_ylim([min(y2)-1, max(y2)+1])
    ax2.set_yticks(y2)
    ax2.tick_params(axis='y', labelsize=7)

    ax2.scatter(x, y2, color=col_tree_size, marker='v', edgecolors='black',
                linewidth=0.7, s=50, alpha=0.8)
    ax2.set_ylabel(f'Tree Size (std dev={std_dev_y2:.3f})', color=col_tree_size)

    # Draw lines from top of bar to tree size ball
    for xb, h, yb in zip(x, y1, y2):
        ax2.plot([xb, xb], [h, yb], color=col_tree_size, linestyle=':', linewidth=1)

    plt.title(title)
    # Custom legend elements
    custom_legend = [
        Line2D([], [], color=col_iter_acc, linestyle='--', linewidth=1,
               label=f'Mean of Accuracy: {mean_y1:.3f}'),
        Line2D([], [], color=col_tree_size, linestyle='None', marker='v', markersize=5,
               markeredgecolor='black', markeredgewidth=0.7, alpha=0.7, label='Tree Size'),
        Patch(color=col_iter_acc, label="Iteration Accuracy"),
        Patch(color=col_best_iter, label=f"Best Iteration ({max(y1):.3f})"),
    ]

    # Add custom legend
    ax1.legend(loc='lower left', handles=custom_legend, fontsize=7, bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename + f"_{mean_y1*1000:.0f}")
    else:
        plt.show()
    plt.close(fig) # Close the figure to free memory
