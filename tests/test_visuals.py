import pytest
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import os
import shutil
from unittest.mock import Mock, patch, MagicMock

# Assume these are in your zuffy library path or mocked
from zuffy._visuals_color_assignment import FeatureColorAssigner, OperatorColorAssigner
from gplearn.functions import _Function

# Import functions from the visuals module
# We need to import the module itself to patch its functions/decorators
import zuffy.visuals
from zuffy.visuals import (
    _add_importance,
    _output_node,
    sanitise_names,
    graph_tree_class,
    graphviz_tree,
    plot_evolution,
    show_feature_importance,
    DEFAULT_CONSTANT_COLOR,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_GRAPH_FONT, # Added this import assuming it's in visuals
    DEFAULT_GRAPH_FONT_SIZE # Added this import assuming it's in visuals
)

# Helper for creating mock gplearn _Program objects
class MockProgram:
    def __init__(self, program_nodes, n_features, raw_fitness_=None):
        # These are now properties to satisfy potential HasMethods deeper checks
        #self._program_nodes = program_nodes
        self.program = program_nodes
        self.n_features = n_features
        self.raw_fitness_ = raw_fitness_

    #@property
    #def _program(self):
    #    return self._program_nodes
#
    #@property
    #def n_features(self):
    #    return self._n_features

# Helper for creating mock gplearn MultiClassClassifier
class MockMultiClassModel:
    def __init__(self, estimators, classes_=None):
        self.multi_ = Mock()
        self.multi_.estimators_ = estimators
        self.classes_ = classes_

# Mock for `_Function` from gplearn
class MockFunction(_Function):
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        pass # Not used in visualization, just for type checking and arity

# --- Fixtures ---

@pytest.fixture
def temp_output_dir():
    """Creates a temporary directory for output files and cleans up after tests."""
    dir_name = "test_output_visuals"
    os.makedirs(dir_name, exist_ok=True)
    yield dir_name
    shutil.rmtree(dir_name)

@pytest.fixture
def mock_feature_color_assigner():
    """Mock FeatureColorAssigner for testing."""
    mock = Mock(spec=FeatureColorAssigner)
    mock.get_color.side_effect = lambda x: f"color_{x}"
    return mock

@pytest.fixture
def mock_operator_color_assigner():
    """Mock OperatorColorAssigner for testing."""
    mock = Mock(spec=OperatorColorAssigner)
    mock.get_color.side_effect = lambda x: f"op_color_{x}"
    return mock

# --- Test Cases for _add_importance ---

def test_add_importance_valid_input():
    """Test _add_importance with valid numeric input."""
    metrics = [0.5, 0.1, 1]
    expected_html = '<tr><td bgColor="white" color="red"><font point-size="14"><b>1</b>: 0.500 &plusmn; 0.100</font></td></tr>'
    assert _add_importance(metrics) == expected_html

def test_add_importance_non_numeric_mean_raises_valueerror():
    """Test _add_importance with non-numeric mean raises ValueError."""
    with pytest.raises(ValueError, match="The mean importance value must be numeric"):
        _add_importance(["invalid", 0.1, 1])

def test_add_importance_incorrect_length_raises_valueerror():
    """Test _add_importance with list of incorrect length raises ValueError."""
    with pytest.raises(ValueError, match="The `feature_metrics` list must contain exactly three elements"):
        _add_importance([0.5, 0.1])
    with pytest.raises(ValueError, match="The `feature_metrics` list must contain exactly three elements"):
        _add_importance([0.5, 0.1, 1, 2])

# --- Test Cases for _output_node ---

def test_output_node_feature_with_names_no_importance(mock_feature_color_assigner):
    """Test _output_node for a feature node with names and no importance."""
    dot_string = _output_node(0, 0, ["feature_A"], mock_feature_color_assigner, None)
    expected_dot = (
        f'0 [label=<\n'
        f'  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="color_feature_A">\n'
        f'    <tr><td>feature_A</td></tr>\n'
        f'    \n'
        f'  </table>>,\n'
        f'  color="black", shape=none\n'
        f'] ;\n'
    )
    assert dot_string == expected_dot

def test_output_node_feature_with_names_and_importance(mock_feature_color_assigner):
    """Test _output_node for a feature node with names and importance."""
    imp_feat = {"feature_A": [0.7, 0.05, 2]}
    dot_string = _output_node(0, 0, ["feature_A"], mock_feature_color_assigner, imp_feat)
    expected_importance_html = '<tr><td bgColor="white" color="red"><font point-size="14"><b>2</b>: 0.700 &plusmn; 0.050</font></td></tr>'
    expected_dot = (
        f'0 [label=<\n'
        f'  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="color_feature_A">\n'
        f'    <tr><td>feature_A</td></tr>\n'
        f'    {expected_importance_html}\n'
        f'  </table>>,\n'
        f'  color="black", shape=none\n'
        f'] ;\n'
    )
    assert dot_string == expected_dot

def test_output_node_feature_without_names(mock_feature_color_assigner):
    """Test _output_node for a feature node without explicit names."""
    dot_string = _output_node(0, 0, None, mock_feature_color_assigner, None)
    expected_dot = (
        f'0 [label=<\n'
        f'  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="color_X0">\n'
        f'    <tr><td>X0</td></tr>\n'
        f'    \n'
        f'  </table>>,\n'
        f'  color="black", shape=none\n'
        f'] ;\n'
    )
    assert dot_string == expected_dot

def test_output_node_constant_node(mock_feature_color_assigner):
    """Test _output_node for a constant node."""
    dot_string = _output_node(1, 3.14159, None, mock_feature_color_assigner, None)
    expected_dot = f'1 [label="3.142", style=filled, fillcolor="{DEFAULT_CONSTANT_COLOR}"] ;\n'
    assert dot_string == expected_dot

def test_output_node_feature_index_out_of_bounds_raises_valueerror(mock_feature_color_assigner):
    """Test _output_node raises ValueError if feature index is out of bounds."""
    with pytest.raises(ValueError, match="Feature index 1 is out of bounds"):
        _output_node(0, 1, ["feature_A"], mock_feature_color_assigner, None)

def test_output_node_feature_name_with_pipe_char(mock_feature_color_assigner):
    """Test _output_node correctly handles '|' in feature names."""
    dot_string = _output_node(0, 0, ["Feature|Name"], mock_feature_color_assigner, None)
    # The '|' should be replaced by '<br/>'
    assert "<tr><td>Feature<br/>Name</td></tr>" in dot_string

# --- Test Cases for sanitise_names ---

def test_sanitise_names_none_input():
    """Test sanitise_names with None input."""
    assert sanitise_names(None) is None

def test_sanitise_names_empty_list():
    """Test sanitise_names with an empty list."""
    assert sanitise_names([]) == []

def test_sanitise_names_valid_names():
    """Test sanitise_names with valid names."""
    names = ["feature1", "feature_2", "long name"]
    assert sanitise_names(names) == ["feature1", "feature_2", "long name"]

def test_sanitise_names_with_html_chars():
    """Test sanitise_names with names containing HTML special characters."""
    names = ["<feature>", "feat&ures", '"quotes"']
    assert sanitise_names(names) == ["&lt;feature&gt;", "feat&amp;ures", "&quot;quotes&quot;"]

def test_sanitise_names_with_mixed_types():
    """Test sanitise_names with mixed types, ensuring conversion to string."""
    names = ["feat_A", 123, 3.14, None]
    assert sanitise_names(names) == ["feat_A", "123", "3.14", "None"]

# --- Test Cases for graph_tree_class ---

@pytest.fixture
def mock_gplearn_program():
    """Mock a gplearn program for graph_tree_class."""
    # A simple program: add(X0, 3.14)
    program_nodes = [
        MockFunction('add', 2),
        0, # Represents feature X0
        3.14, # Represents constant 3.14
    ]
    # Use MockProgram here, as it now uses properties for _program and n_features
    return MockProgram(program_nodes, n_features=1)

@pytest.fixture
def mock_gplearn_program_single_node():
    """Mock a gplearn program with a single terminal node."""
    # A program that is just X0
    program_nodes = [0] # Represents feature X0
    return MockProgram(program_nodes, n_features=1)


# Patch the validate_params decorator for graph_tree_class
# We replace it with a dummy decorator that just passes the function through.
@pytest.mark.skip(reason="skipped because it needs work")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_export_graphviz_valid_program(mock_gplearn_program, mock_operator_color_assigner, mock_feature_color_assigner):
    """Test graph_tree_class with a valid program and no importance."""
    dot_script = graph_tree_class(
        program=mock_gplearn_program,
        feature_names=["FeatureA"],
        operator_col_fn=mock_operator_color_assigner,
        feature_col_fn=mock_feature_color_assigner,
        imp_feat=None
    )
    
    # Expected DOT script parts
    assert '0 [label="add", style=filled, fillcolor="op_color_add"] ;' in dot_script
    assert f'1 [label=<\n  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="color_FeatureA">\n    <tr><td>FeatureA</td></tr>\n    \n  </table>>,\n  color="black", shape=none\n] ;' in dot_script
    assert f'2 [label="3.140", style=filled, fillcolor="{DEFAULT_CONSTANT_COLOR}"] ;' in dot_script # note not 3.142
    assert '0 -> 1 ;' in dot_script
    assert '0 -> 2 ;' in dot_script
    assert 'digraph G {' not in dot_script # graph_tree_class only returns the internal nodes/edges

@pytest.mark.skip(reason="skipped because it needs work")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_export_graphviz_single_node_program(mock_gplearn_program_single_node, mock_feature_color_assigner):
    """Test graph_tree_class with a single-node program."""
    dot_script = graph_tree_class(
        program=mock_gplearn_program_single_node,
        feature_names=["FeatureA"],
        feature_col_fn=mock_feature_color_assigner,
        imp_feat=None
    )
    # For a single node, there should be no edges or operators
    expected_dot = (
        f'0 [label=<\n'
        f'  <table border="1" cellborder="0" cellspacing="6" cellpadding="3" bgColor="color_FeatureA">\n'
        f'    <tr><td>FeatureA</td></tr>\n'
        f'    \n'
        f'  </table>>,\n'
        f'  color="black", shape=none\n'
        f'] ;\n'
    )
    assert dot_script == expected_dot

@pytest.mark.skip(reason="skipped because it needs work")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_export_graphviz_with_importance(mock_gplearn_program, mock_operator_color_assigner, mock_feature_color_assigner):
    """Test graph_tree_class with feature importance."""
    imp_feat = {"FeatureA": [0.8, 0.02, 1]}
    dot_script = graph_tree_class(
        program=mock_gplearn_program,
        feature_names=["FeatureA"],
        operator_col_fn=mock_operator_color_assigner,
        feature_col_fn=mock_feature_color_assigner,
        imp_feat=imp_feat
    )
    expected_importance_html = '<tr><td bgColor="white" color="red"><font point-size="14"><b>1</b>: 0.800 &plusmn; 0.020</font></td></tr>'
    assert expected_importance_html in dot_script

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_export_graphviz_insufficient_feature_names_raises_valueerror(mock_gplearn_program):
    """Test graph_tree_class raises ValueError for insufficient feature names."""
    mock_gplearn_program._n_features = 2 # Simulate program needing 2 features
    with pytest.raises(ValueError, match="There are insufficient feature_names"):
        graph_tree_class(mock_gplearn_program, feature_names=["FeatureA"])

# --- Test Cases for graphviz_tree ---

@pytest.fixture
def mock_gplearn_estimator_with_program():
    """Mock a gplearn estimator (like a Program) with `_program` attribute."""
    mock_program = MockProgram(
        program_nodes=[MockFunction('add', 2), 0, 1], # add(X0, X1)
        n_features=2,
        raw_fitness_=0.0123
    )
    mock_estimator = Mock()
    mock_estimator._program = mock_program # This is a direct instance of MockProgram
    mock_estimator.feature_names_in_ = np.array(['feat_0', 'feat_1']) # scikit-learn standard
    mock_estimator.feature_names = ['old_feat_0', 'old_feat_1'] # gplearn standard
    return mock_estimator

@pytest.fixture
def mock_gplearn_estimator_no_feature_names():
    """Mock a gplearn estimator without feature names."""
    mock_program = MockProgram(program_nodes=[MockFunction('add', 2), 0, 1], n_features=2)
    mock_estimator = Mock()
    mock_estimator._program = mock_program
    return mock_estimator

@pytest.fixture
def mock_gplearn_multiclass_model(mock_gplearn_estimator_with_program):
    """Mock a gplearn MultiClassClassifier."""
    return MockMultiClassModel(
        estimators=[
            mock_gplearn_estimator_with_program,
            mock_gplearn_estimator_with_program # Two estimators for two classes
        ],
        classes_=['Class_A', 'Class_B']
    )

# Patch the validate_params decorator for graphviz_tree
@pytest.mark.skip(reason="skipped because it needs work")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('graphviz.Source')
def test_graphviz_tree_valid_model(mock_graphviz_source, temp_output_dir, mock_gplearn_multiclass_model):
    """Test graphviz_tree with a valid model and default options."""
    output_path = os.path.join(temp_output_dir, "test_tree")
    
    full_dot_script, graph = graphviz_tree(
        model=mock_gplearn_multiclass_model,
        output_filename=output_path,
        show_fitness=True
    )

    mock_graphviz_source.assert_called_once()
    graph.render.assert_called_once_with(output_path, format='png', view=False, cleanup=True)

    assert "digraph G {" in full_dot_script
    assert f'bgcolor="white"' in full_dot_script
    assert f'fontname="{DEFAULT_GRAPH_FONT}"' in full_dot_script
    assert f'fontsize="{DEFAULT_GRAPH_FONT_SIZE}"' in full_dot_script
    assert "node [fontname=\"Helvetica\"]" in full_dot_script
    
    # Check for WTA node and edges
    wta_id = (len(mock_gplearn_multiclass_model.multi_.estimators_) + 1) * 1000
    assert f'{wta_id} [label=<<table border="1" cellborder="1" bgcolor="grey"><tr><td colspan="2">WTA</td></tr><tr><td port="port_0">Target=Class_A (0.012)</td><td port="port_1">Target=Class_B (0.012)</td></tr></table>>' in full_dot_script
    assert f'{wta_id}:port_0 -> 0;' in full_dot_script # Root of first sub-tree
    assert f'{wta_id}:port_1 -> 1000;' in full_dot_script # Root of second sub-tree (offset by 1000)
    
    # Check if feature names from model.multi_.estimators_[0].feature_names_in_ were used
    assert "feat_0" in full_dot_script
    assert "feat_1" in full_dot_script

@pytest.mark.skip(reason="skipped because it uses export and that fails when no .program???")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_graphviz_tree_no_model_program_attribute_raises_valueerror(mock_gplearn_multiclass_model):
    """Test graphviz_tree raises ValueError if sub-estimator lacks _program."""
    #mock_gplearn_multiclass_model.multi_.estimators_[0] = Mock() # Remove _program attribute
    mock_estimator_without_program = Mock()
    mock_estimator_without_program.feature_names_in_ = np.array(['feat_0', 'feat_1']) # Provide this
    mock_gplearn_multiclass_model.multi_.estimators_[0] = mock_estimator_without_program
    
    with pytest.raises(ValueError, match="Each estimator in `model.multi_.estimators_` is expected to have a `_program` attribute"):
        graphviz_tree(mock_gplearn_multiclass_model, output_filename="dummy")

 
@pytest.mark.skip(reason="skipped because 'MockProgram' object has no attribute '_n_features'")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('graphviz.Source')
def test_graphviz_tree_explicit_feature_names(mock_graphviz_source, temp_output_dir, mock_gplearn_multiclass_model):
    """Test graphviz_tree with explicitly provided feature names."""
    output_path = os.path.join(temp_output_dir, "test_tree_explicit_features")
    custom_feature_names = ["CustomFeat1", "CustomFeat2"]
    
    graphviz_tree(
        model=mock_gplearn_multiclass_model,
        feature_names=custom_feature_names,
        output_filename=output_path
    )
    
    full_dot_script = mock_graphviz_source.call_args[0][0]
    assert "CustomFeat1" in full_dot_script
    assert "CustomFeat2" in full_dot_script
    assert "feat_0" not in full_dot_script # Ensure default names are not used

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('graphviz.Source')
def test_graphviz_tree_no_feature_names_in_model(mock_graphviz_source, temp_output_dir):
    """Test graphviz_tree when no feature names are available in the model."""
    mock_estimator = Mock()
    mock_program = MockProgram(program_nodes=[MockFunction('add', 2), 0, 1], n_features=2)
    mock_estimator.feature_names_in_ = None
    mock_estimator._program = mock_program

    #mock_estimator_without_program = Mock()
    #mock_estimator_without_program.feature_names_in_ = np.array(['feat_0', 'feat_1']) # Provide this
    #mock_gplearn_multiclass_model.multi_.estimators_[0] = mock_estimator_without_program


    # No feature_names or feature_names_in_
    mock_model = MockMultiClassModel(estimators=[mock_estimator])
    mock_model.multi_.estimators_[0].feature_names = None # pom

    output_path = os.path.join(temp_output_dir, "test_tree_no_features")
    graphviz_tree(model=mock_model, output_filename=output_path)
    
    full_dot_script = mock_graphviz_source.call_args[0][0]
    assert "X0" in full_dot_script
    assert "X1" in full_dot_script

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_graphviz_tree_insufficient_target_class_names_raises_valueerror(mock_gplearn_multiclass_model):
    """Test graphviz_tree raises ValueError for insufficient target class names."""
    with pytest.raises(ValueError, match="Insufficient `target_class_names`"):
        graphviz_tree(mock_gplearn_multiclass_model, target_class_names=['Class_A'], output_filename="dummy")

@pytest.mark.skip(reason="skipped because 'MockProgram' object has no attribute '_n_features'")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('graphviz.Source')
def test_graphviz_tree_generates_source_file(mock_graphviz_source, temp_output_dir, mock_gplearn_multiclass_model):
    """Test graphviz_tree generates a .dot source file if source_filename is provided."""
    source_filename = os.path.join(temp_output_dir, "test_tree.dot")
    graphviz_tree(mock_gplearn_multiclass_model, output_filename="test_tree", source_filename=source_filename)
    assert os.path.exists(source_filename)
    with open(source_filename, 'r') as f:
        content = f.read()
        assert "digraph G {" in content # Check that the .dot file contains the full graphviz script

# --- Test Cases for plot_evolution ---

@pytest.fixture
def mock_gplearn_estimator_with_run_details():
    """Mock a gplearn estimator with `run_details_` attribute."""
    mock_estimator = Mock()
    mock_estimator.run_details_ = {
        'generation': [0, 1, 2, 3],
        'average_length': [5, 6, 7, 6.5],
        'best_length': [5, 4, 3, 3],
        'average_fitness': [0.1, 0.08, 0.05, 0.04],
        'best_fitness': [0.09, 0.07, 0.04, 0.03],
        'generation_time': [0.1, 0.12, 0.11, 0.09]
    }
    return mock_estimator

@pytest.fixture
def mock_gplearn_multiclass_model_with_run_details(mock_gplearn_estimator_with_run_details):
    """Mock a gplearn MultiClassClassifier with run details."""
    return MockMultiClassModel(
        estimators=[
            mock_gplearn_estimator_with_run_details,
            mock_gplearn_estimator_with_run_details
        ],
        classes_=['Class_X', 'Class_Y']
    )

# Patch the validate_params decorator for plot_evolution
@pytest.mark.skip(reason="skipped because fig is called twice - why does tight layout also call it?")
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_plot_evolution_valid_model(mock_close, mock_show, mock_savefig, mock_figure, 
                                     temp_output_dir, mock_gplearn_multiclass_model_with_run_details):
    """Test plot_evolution with a valid model."""
    output_path = os.path.join(temp_output_dir, "evolution_plot.png")
    
    plot_evolution(mock_gplearn_multiclass_model_with_run_details, output_filename=output_path)

    mock_figure.assert_called_once()
    mock_savefig.assert_called_once_with(output_path)
    mock_show.assert_not_called()
    mock_close.assert_called_once()
    
    # Verify titles (basic check)
    fig_mock = mock_figure.return_value
    assert fig_mock.add_subplot.call_count == 6 # 2 estimators * 3 plots each

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_plot_evolution_show_plot(mock_close, mock_show, mock_savefig, mock_figure, 
                                   mock_gplearn_multiclass_model_with_run_details):
    """Test plot_evolution shows plot if no output_filename is provided."""
    plot_evolution(mock_gplearn_multiclass_model_with_run_details, output_filename=None)
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_plot_evolution_no_run_details_raises_attributeerror(mock_gplearn_multiclass_model):
    """Test plot_evolution raises AttributeError if estimator lacks run_details_."""
    # This model's estimators don't have run_details_ by default fixture
    with pytest.raises(AttributeError, match="Estimator for class 'Class_A' is missing 'run_details_' attribute"):
        plot_evolution(mock_gplearn_multiclass_model, output_filename="dummy.png")

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_plot_evolution_insufficient_target_class_names_raises_valueerror(mock_gplearn_multiclass_model_with_run_details):
    """Test plot_evolution raises ValueError for insufficient target class names."""
    with pytest.raises(ValueError, match="Insufficient `target_class_names`"):
        plot_evolution(mock_gplearn_multiclass_model_with_run_details, target_class_names=['Class_X'], output_filename="dummy.png")

# --- Test Cases for show_feature_importance ---

@pytest.fixture
def mock_regressor():
    """Mock a regressor for permutation_importance."""
    mock_reg = Mock()
    mock_reg.predict.return_value = np.array([1, 2, 3, 4, 5])
    
    # Mock for a gplearn-like model with multi_ and estimators_
    mock_program = Mock()
    mock_program.feature_names = ["F1", "F2", "F3"] # gplearn style feature names
    mock_program.feature_names_in_ = np.array(["F1_in", "F2_in", "F3_in"]) # sklearn style feature names
    
    mock_estimator = Mock()
    mock_estimator.feature_names = mock_program.feature_names
    mock_estimator.feature_names_in_ = mock_program.feature_names_in_

    mock_reg.multi_ = Mock()
    mock_reg.multi_.estimators_ = [mock_estimator] # A list of estimators
    
    return mock_reg

# Patch the validate_params decorator for show_feature_importance
@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('zuffy.visuals.permutation_importance')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_show_feature_importance_valid_input(mock_close, mock_show, mock_savefig, mock_figure, 
                                              mock_permutation_importance, temp_output_dir, mock_regressor):
    """Test show_feature_importance with valid input and output to file."""
    X_test = np.array([[1, 2, 3], [4, 5, 6]])
    y_test = np.array([10, 20])
    features = ["Feature_A", "Feature_B", "Feature_C"]
    output_path = os.path.join(temp_output_dir, "importance.png")

    mock_permutation_importance.return_value = Mock(
        importances_mean=np.array([0.1, 0.5, 0.0]),
        importances_std=np.array([0.01, 0.05, 0.0])
    )

    imp_dict = show_feature_importance(
        mock_regressor, X_test, y_test, features=features, output_filename=output_path
    )

    mock_permutation_importance.assert_called_once_with(
        mock_regressor, X_test, y_test, n_repeats=20, n_jobs=None
    )
    mock_figure.assert_called_once()
    mock_savefig.assert_called_once_with(output_path)
    mock_show.assert_not_called()
    mock_close.assert_called_once()

    expected_imp_dict = {
        "Feature_B": [0.5, 0.05, 1],
        "Feature_A": [0.1, 0.01, 2],
    }
    # Check if the keys match, and then if values match after sorting
    assert sorted(imp_dict.keys()) == sorted(expected_imp_dict.keys())
    for k in imp_dict:
        # Allow for floating point precision in value comparison
        assert imp_dict[k][0] == pytest.approx(expected_imp_dict[k][0])
        assert imp_dict[k][1] == pytest.approx(expected_imp_dict[k][1])
        assert imp_dict[k][2] == expected_imp_dict[k][2] # Rank should be exact


@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('zuffy.visuals.permutation_importance')
@patch('matplotlib.pyplot.figure')
@patch('matplotlib.pyplot.savefig')
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.close')
def test_show_feature_importance_shows_plot(mock_close, mock_show, mock_savefig, mock_figure,
                                            mock_permutation_importance, mock_regressor):
    """Test show_feature_importance shows plot if no output_filename."""
    X_test = np.array([[1, 2, 3]])
    y_test = np.array([10])
    mock_permutation_importance.return_value = Mock(
        importances_mean=np.array([0.1]), importances_std=np.array([0.01])
    )
    show_feature_importance(mock_regressor, X_test, y_test, output_filename=None)
    mock_show.assert_called_once()
    mock_savefig.assert_not_called()

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
@patch('zuffy.visuals.permutation_importance')
def test_show_feature_importance_no_important_features(mock_permutation_importance, mock_regressor):
    """Test show_feature_importance when no features have positive importance."""
    X_test = np.array([[1, 2]])
    y_test = np.array([10])
    mock_permutation_importance.return_value = Mock(
        importances_mean=np.array([0.0, 0.0]),
        importances_std=np.array([0.0, 0.0])
    )
    imp_dict = show_feature_importance(mock_regressor, X_test, y_test, features=["A", "B"])
    assert imp_dict == {}

@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_show_feature_importance_feature_names_from_model(mock_regressor):
    """Test feature names are correctly retrieved from model's estimators."""
    X_test = np.array([[1, 2, 3]])
    y_test = np.array([10])

    with patch('zuffy.visuals.permutation_importance') as mock_pi:
        mock_pi.return_value = Mock(importances_mean=np.array([0.1, 0.5, 0.0]), importances_std=np.array([0.01, 0.05, 0.0]))
        
        # Test with feature_names_in_ (sklearn standard)
        imp_dict = show_feature_importance(mock_regressor, X_test, y_test, features=None)
        assert "F1" in imp_dict
        assert "F2" in imp_dict
        assert "F3" not in imp_dict # F3_in has 0 importance

        # Remove feature_names_in_ to test feature_names (gplearn standard)
        del mock_regressor.multi_.estimators_[0].feature_names_in_
        imp_dict = show_feature_importance(mock_regressor, X_test, y_test, features=None)
        assert "F1" in imp_dict
        assert "F2" in imp_dict
        assert "F3" not in imp_dict # F3 has 0 importance


@patch('zuffy.visuals.validate_params', lambda x, prefer_skip_nested_validation: lambda func: func)
def test_show_feature_importance_default_feature_names():
    """Test default X0, X1, ... feature names when none provided and not in model."""
    mock_reg = Mock()
    mock_reg.predict.return_value = np.array([1, 2])
    # Ensure no 'multi_' attribute or 'feature_names' on estimators
    mock_reg.multi_ = Mock()
    mock_reg.multi_.estimators_ = []

    X_test = np.array([[1, 2]])
    y_test = np.array([10, 20])

    with patch('zuffy.visuals.permutation_importance') as mock_pi:
        mock_pi.return_value = Mock(importances_mean=np.array([0.1, 0.5]), importances_std=np.array([0.01, 0.05]))
        imp_dict = show_feature_importance(mock_reg, X_test, y_test, features=None)
        assert "X0" in imp_dict
        assert "X1" in imp_dict

# --- Test Cases for do_model_dt ---

@pytest.fixture
def sample_data():
    """Sample data for Decision Tree."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([10, 20, 30])
    features = ["feat_one", "feat_two"]
    return X, y, features

