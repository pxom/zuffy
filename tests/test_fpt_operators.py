import pytest
import numpy as np

# Import the functions from your module
# Assuming your code is in 'fuzzy_operators.py'
import zuffy._fpt_operators as fops

# Reusable test data using pytest fixtures
@pytest.fixture
def sample_data():
    """Provides a set of numpy arrays for testing."""
    return {
        "a": np.array([0.1, 0.5, 0.8]),
        "b": np.array([0.3, 0.5, 0.7]),
        "c": np.array([0.9, 0.2, 0.6]),
        "d": np.array([0.0, 1.0, 0.4]),
        "neg": np.array([-0.2, 0.5, -0.9])
    }

# --- Tests for Private Helper Functions ---

class TestPrivateFunctions:
    """Tests for the internal '_' prefixed functions."""

    def test_weighted_average(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        x = 0.25
        expected = x * a + (1 - x) * b
        np.testing.assert_allclose(fops._weighted_average(a, b, x), expected)
        # Test with scalars
        assert fops._weighted_average(0.2, 0.8, 0.5) == 0.5

    def test_ordered_weighted_average(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        x = 0.75
        expected = x * np.maximum(a, b) + (1 - x) * np.minimum(a, b)
        np.testing.assert_allclose(fops._ordered_weighted_average(a, b, x), expected)
        # Test with scalars
        assert fops._ordered_weighted_average(0.2, 0.8, 0.5) == 0.5

    def test_minimum(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = np.array([0.1, 0.5, 0.7])
        np.testing.assert_allclose(fops._minimum(a, b), expected)

    def test_maximum(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = np.array([0.3, 0.5, 0.8])
        np.testing.assert_allclose(fops._maximum(a, b), expected)

    def test_diluter(self, sample_data):
        a, neg_data = sample_data["a"], sample_data["neg"]
        expected = np.array([np.sqrt(0.1), np.sqrt(0.5), np.sqrt(0.8)])
        np.testing.assert_allclose(fops._diluter(a), expected)
        # Test that negative inputs are handled correctly
        expected_neg = np.array([0, np.sqrt(0.5), 0])
        np.testing.assert_allclose(fops._diluter(neg_data), expected_neg)

    def test_diluter_power(self, sample_data):
        a = sample_data["a"]
        power = 1/3
        expected = a**power
        np.testing.assert_allclose(fops._diluter_power(a, power), expected)

    def test_concentrator(self, sample_data):
        a = sample_data["a"]
        expected = np.array([0.01, 0.25, 0.64])
        np.testing.assert_allclose(fops._concentrator(a), expected)

    def test_concentrator_power(self, sample_data):
        a = sample_data["a"]
        power = 3
        expected = a**power
        np.testing.assert_allclose(fops._concentrator_power(a, power), expected)
        
    def test_fuzzy_and(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = a * b
        np.testing.assert_allclose(fops._fuzzy_and(a, b), expected)

    def test_fuzzy_or(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = a + b - a * b
        np.testing.assert_allclose(fops._fuzzy_or(a, b), expected)

    def test_complement(self, sample_data):
        a = sample_data["a"]
        expected = 1 - a
        np.testing.assert_allclose(fops._complement(a), expected)

    def test_intensifier(self):
        x = np.array([0.2, 0.5, 0.8, -0.1])
        n = 2
        expected = np.array([
            0.5**(1 - n) * (0.2**n),
            1 - 0.5**(1 - n) * (1 - 0.5)**n, # Should be exactly 0.5
            1 - 0.5**(1 - n) * (1 - 0.8)**n,
            0
        ])
        np.testing.assert_allclose(fops._intensifier(x), expected)
        assert fops._intensifier(0.5) == 0.5 # Test scalar midpoint

    def test_diffuser(self):
        x = np.array([0.2, 0.5, 0.8, -0.1])
        n = 2
        expected = np.array([
            0.5**(1 - 1/n) * x[0]**(1/n),
            1 - 0.5**(1-1/n) * (1 - 0.5)**(1/n), # Should be exactly 0.5
            1 - 0.5**(1 - 1/n) * (1 - x[2])**(1/n),
            0
        ])
        np.testing.assert_allclose(fops._diffuser(x), expected)
        assert round(fops._diffuser(0.5)+0,10) == 0.5 # Test scalar midpoint

    def test_if_gte(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = np.where(a >= b, a, b) # This is equivalent to np.maximum
        np.testing.assert_allclose(fops._if_gte(a, b), expected)

    def test_if_gte_else(self, sample_data):
        a, b, c, d = sample_data["a"], sample_data["b"], sample_data["c"], sample_data["d"]
        expected = np.where(a >= b, c, d)
        np.testing.assert_allclose(fops._if_gte_else(a, b, c, d), expected)

    def test_if_lt(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = np.where(a < b, a, b) # This is equivalent to np.minimum
        np.testing.assert_allclose(fops._if_lt(a, b), expected)

    def test_if_lt_else(self, sample_data):
        a, b, c, d = sample_data["a"], sample_data["b"], sample_data["c"], sample_data["d"]
        expected = np.where(a < b, c, d)
        np.testing.assert_allclose(fops._if_lt_else(a, b, c, d), expected)

    def test_lukasiewicz_t_norm(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        expected = np.maximum(0, a + b - 1)
        np.testing.assert_allclose(fops._lukasiewicz_t_norm(a, b), expected)
        
    def test_hamacher_t_norm(self, sample_data):
        a, b = sample_data["a"], sample_data["b"]
        lambda_param = 0.5
        numerator = a * b
        denominator = lambda_param + (1 - lambda_param) * (a + b - a * b)
        expected = numerator / denominator
        np.testing.assert_allclose(fops._hamacher_t_norm(a, b, lambda_param), expected)
        
        # Test division by zero case
        if 0:
            result_zero = fops._hamacher_t_norm(0.0, 0.0, 0.0)
            print("Result of div0 is",result_zero)
            assert result_zero == 0.0

        # Test invalid lambda
        with pytest.raises(ValueError, match="Parameter lambda_param must be non-negative."):
            fops._hamacher_t_norm(a, b, -1.0)

# --- Tests for gplearn Wrapped Functions ---

class TestGplearnFunctions:
    """Tests for the gplearn-compatible Function objects."""

    def test_static_gplearn_functions(self, sample_data):
        """Test gplearn functions with fixed arity."""
        a, b, c, d = sample_data["a"], sample_data["b"], sample_data["c"], sample_data["d"]

        # Arity 1
        np.testing.assert_allclose(fops.COMPLEMENT(a), fops._complement(a))
        np.testing.assert_allclose(fops.DILUTER(a), fops._diluter(a))
        np.testing.assert_allclose(fops.DILUTER3(a), fops._diluter_power(a, 1/3))
        np.testing.assert_allclose(fops.CONCENTRATOR(a), fops._concentrator(a))
        np.testing.assert_allclose(fops.INTENSIFIER(a), fops._intensifier(a))

        # Arity 2
        np.testing.assert_allclose(fops.MAXIMUM(a, b), fops._maximum(a, b))
        np.testing.assert_allclose(fops.MINIMUM(a, b), fops._minimum(a, b))
        np.testing.assert_allclose(fops.FUZZY_AND(a, b), fops._fuzzy_and(a, b))
        np.testing.assert_allclose(fops.FUZZY_OR(a, b), fops._fuzzy_or(a, b))
        np.testing.assert_allclose(fops.LUKASIEWICZ_AND(a, b), fops._lukasiewicz_t_norm(a, b))
        np.testing.assert_allclose(fops.LUKASIEWICZ_OR(a, b), fops._lukasiewicz_t_conorm(a, b))
        np.testing.assert_allclose(fops.HAMACHER050(a, b), fops._hamacher_t_norm(a, b, 0.5))

        # Arity 4
        np.testing.assert_allclose(fops.IFGTE2(a, b, c, d), fops._if_gte_else(a, b, c, d))
        np.testing.assert_allclose(fops.IFLT2(a, b, c, d), fops._if_lt_else(a, b, c, d))

    def test_dynamically_generated_wa(self, sample_data):
        """Test one of the dynamically generated WA operators."""
        a, b = sample_data["a"], sample_data["b"]
        # Test WA_P3, which corresponds to a weight of 0.3
        param = 0.3
        expected = fops._weighted_average(a, b, param)
        np.testing.assert_allclose(fops.WA_P3(a, b), expected)
        assert fops.WA_P3.name == 'WA_P3'
        #assert fops.WA