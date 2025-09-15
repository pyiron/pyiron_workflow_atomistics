"""
Unit tests for pyiron_workflow_atomistics.calculator module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.calculators.calculator import Calculator

import pyiron_workflow_atomistics.calculator as calc_module


class TestCalculatorFunctions(unittest.TestCase):
    """Test calculator module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        self.test_calc = Mock(spec=Calculator)
        self.test_engine = Mock()
        self.test_engine.get_calculate_fn.return_value = (Mock(), {})

    def test_validate_calculation_inputs_with_engine(self):
        """Test validation with calculation engine."""
        result = calc_module.validate_calculation_inputs(
            calculation_engine=self.test_engine
        ).run()
        self.assertTrue(result)

    def test_validate_calculation_inputs_with_fn_and_kwargs(self):
        """Test validation with function and kwargs."""
        result = calc_module.validate_calculation_inputs(
            calc_structure_fn=Mock(), calc_structure_fn_kwargs={}
        ).run()
        self.assertTrue(result)

    def test_validate_calculation_inputs_invalid_engine_and_fn(self):
        """Test validation fails when both engine and fn are provided."""
        with self.assertRaises(ValueError):
            calc_module.validate_calculation_inputs(
                calculation_engine=self.test_engine,
                calc_structure_fn=Mock(),
                calc_structure_fn_kwargs={},
            ).run()

    def test_validate_calculation_inputs_invalid_missing_both(self):
        """Test validation fails when neither engine nor fn/kwargs provided."""
        with self.assertRaises(ValueError):
            calc_module.validate_calculation_inputs().run()

    def test_validate_calculation_inputs_invalid_partial_fn(self):
        """Test validation fails when only fn or kwargs provided."""
        with self.assertRaises(ValueError):
            calc_module.validate_calculation_inputs(calc_structure_fn=Mock()).run()
        with self.assertRaises(ValueError):
            calc_module.validate_calculation_inputs(calc_structure_fn_kwargs={}).run()

    def test_calculate_structure_node_with_engine(self):
        """Test calculate_structure_node with calculation engine."""
        mock_output = Mock()
        mock_fn = Mock(return_value=mock_output)
        self.test_engine.get_calculate_fn.return_value = (mock_fn, {})

        result = calc_module.calculate_structure_node(
            structure=self.test_atoms, calculation_engine=self.test_engine
        ).run()

        self.assertEqual(result, mock_output)
        mock_fn.assert_called_once_with(structure=self.test_atoms)

    def test_calculate_structure_node_with_fn_and_kwargs(self):
        """Test calculate_structure_node with function and kwargs."""
        mock_output = Mock()
        mock_fn = Mock(return_value=mock_output)
        test_kwargs = {"param1": "value1"}

        result = calc_module.calculate_structure_node(
            structure=self.test_atoms,
            _calc_structure_fn=mock_fn,
            _calc_structure_fn_kwargs=test_kwargs,
        ).run()

        self.assertEqual(result, mock_output)
        mock_fn.assert_called_once_with(structure=self.test_atoms, **test_kwargs)

    def test_convert_EngineOutput_to_output_dict(self):
        """Test conversion of EngineOutput to dictionary."""
        mock_output = Mock()
        mock_output.to_dict.return_value = {"energy": 1.0, "forces": [[0, 0, 0]]}

        result = calc_module.convert_EngineOutput_to_output_dict(mock_output).run()

        self.assertEqual(result, {"energy": 1.0, "forces": [[0, 0, 0]]})
        mock_output.to_dict.assert_called_once()

    def test_extract_output_values_from_EngineOutput_single(self):
        """Test extracting values from single EngineOutput."""
        mock_output = Mock()
        mock_output.to_dict.return_value = {"energy": 1.0, "volume": 10.0}

        result = calc_module.extract_output_values_from_EngineOutput(
            mock_output, "energy"
        ).run()

        self.assertEqual(result, 1.0)

    def test_extract_output_values_from_EngineOutput_list(self):
        """Test extracting values from list of EngineOutputs."""
        mock_output1 = Mock()
        mock_output1.to_dict.return_value = {"energy": 1.0}
        mock_output2 = Mock()
        mock_output2.to_dict.return_value = {"energy": 2.0}

        result = calc_module.extract_output_values_from_EngineOutput(
            [mock_output1, mock_output2], "energy"
        ).run()

        self.assertEqual(result, [1.0, 2.0])

    def test_extract_values_from_dict(self):
        """Test extracting values from dictionary."""
        test_dict = [{"energy": 1.0, "volume": 10.0}, {"energy": 2.0, "volume": 20.0}]

        result = calc_module.extract_values_from_dict(test_dict, "energy").run()

        self.assertEqual(result, [1.0, 2.0])

    def test_extract_values_from_dict_missing_key(self):
        """Test extracting values with missing key."""
        test_dict = [{"energy": 1.0}]  # Missing 'volume' key

        result = calc_module.extract_values_from_dict(test_dict, "volume").run()

        self.assertTrue(np.isnan(result))

    def test_fillin_default_calckwargs(self):
        """Test filling in default calculation kwargs."""
        calc_kwargs = {"param1": "value1"}
        default_values = {"param2": "default2", "param3": "default3"}

        result = calc_module.fillin_default_calckwargs(
            calc_kwargs, default_values
        ).run()

        expected = {"param1": "value1", "param2": "default2", "param3": "default3"}
        self.assertEqual(result, expected)

    def test_fillin_default_calckwargs_with_override(self):
        """Test filling in defaults with user override."""
        calc_kwargs = {"param1": "value1", "param2": "override2"}
        default_values = {"param2": "default2", "param3": "default3"}

        result = calc_module.fillin_default_calckwargs(
            calc_kwargs, default_values
        ).run()

        expected = {
            "param1": "value1",
            "param2": "override2",  # User value should override default
            "param3": "default3",
        }
        self.assertEqual(result, expected)

    def test_fillin_default_calckwargs_properties_tuple(self):
        """Test that properties are converted to tuple."""
        calc_kwargs = {"properties": ["energy", "forces"]}

        result = calc_module.fillin_default_calckwargs(calc_kwargs).run()

        self.assertEqual(result["properties"], ("energy", "forces"))

    def test_fillin_default_calckwargs_remove_keys(self):
        """Test removing specified keys."""
        calc_kwargs = {"param1": "value1", "param2": "value2", "param3": "value3"}
        remove_keys = ["param2"]

        result = calc_module.fillin_default_calckwargs(
            calc_kwargs, remove_keys=remove_keys
        ).run()

        expected = {"param1": "value1", "param3": "value3"}
        self.assertEqual(result, expected)

    def test_generate_kwargs_variant(self):
        """Test generating single kwargs variant."""
        base_kwargs = {"param1": "value1", "param2": "value2"}

        result = calc_module.generate_kwargs_variant(
            base_kwargs, "param1", "new_value1"
        ).run()

        expected = {"param1": "new_value1", "param2": "value2"}
        self.assertEqual(result, expected)

        # Ensure original is not modified
        self.assertEqual(base_kwargs, {"param1": "value1", "param2": "value2"})

    def test_generate_kwargs_variants(self):
        """Test generating multiple kwargs variants."""
        base_kwargs = {"param1": "value1", "param2": "value2"}
        values = ["new1", "new2", "new3"]

        result = calc_module.generate_kwargs_variants(
            base_kwargs, "param1", values
        ).run()

        expected = [
            {"param1": "new1", "param2": "value2"},
            {"param1": "new2", "param2": "value2"},
            {"param1": "new3", "param2": "value2"},
        ]
        self.assertEqual(result, expected)

    def test_add_arg_to_kwargs_list_single_value(self):
        """Test adding single value to kwargs list."""
        kwargs_list = [{"param1": "value1"}, {"param1": "value2"}]

        result = calc_module.add_arg_to_kwargs_list(
            kwargs_list, "param2", "new_value"
        ).run()

        expected = [
            {"param1": "value1", "param2": "new_value"},
            {"param1": "value2", "param2": "new_value"},
        ]
        self.assertEqual(result, expected)

    def test_add_arg_to_kwargs_list_list_value(self):
        """Test adding list of values to kwargs list."""
        kwargs_list = [{"param1": "value1"}, {"param1": "value2"}]
        values = ["new1", "new2"]

        result = calc_module.add_arg_to_kwargs_list(kwargs_list, "param2", values).run()

        expected = [
            {"param1": "value1", "param2": "new1"},
            {"param1": "value2", "param2": "new2"},
        ]
        self.assertEqual(result, expected)

    def test_add_arg_to_kwargs_list_remove_if_exists(self):
        """Test removing existing key when adding new one."""
        kwargs_list = [{"param1": "value1", "param2": "old2"}]

        result = calc_module.add_arg_to_kwargs_list(
            kwargs_list, "param2", "new_value", remove_if_exists=True
        ).run()

        expected = [{"param1": "value1", "param2": "new_value"}]
        self.assertEqual(result, expected)

    def test_add_arg_to_kwargs_list_key_exists_error(self):
        """Test error when key already exists and remove_if_exists=False."""
        kwargs_list = [{"param1": "value1", "param2": "existing"}]

        with self.assertRaises(ValueError):
            calc_module.add_arg_to_kwargs_list(
                kwargs_list, "param2", "new_value", remove_if_exists=False
            ).run()


if __name__ == "__main__":
    unittest.main()
