"""
Unit tests for pyiron_workflow_atomistics.utils module.
"""

import unittest
from unittest.mock import Mock

from ase import Atoms
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

import pyiron_workflow_atomistics.utils as utils_module


class TestUtilsFunctions(unittest.TestCase):
    """Test utils module functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_atoms = bulk("Al", crystalstructure="fcc", a=4.0, cubic=True)
        self.test_pmg_structure = AseAtomsAdaptor.get_structure(self.test_atoms)

    def test_convert_structure_ase_to_pmg(self):
        """Test converting ASE structure to pymatgen."""
        result = utils_module.convert_structure(
            self.test_atoms, target="pymatgen"
        ).run()

        # Should return a pymatgen Structure
        from pymatgen.core import Structure

        self.assertIsInstance(result, Structure)
        self.assertEqual(len(result), len(self.test_atoms))

    def test_convert_structure_pmg_to_ase(self):
        """Test converting pymatgen structure to ASE."""
        result = utils_module.convert_structure(
            self.test_pmg_structure, target="ase"
        ).run()

        # Should return an ASE Atoms object
        self.assertIsInstance(result, Atoms)
        self.assertEqual(len(result), len(self.test_pmg_structure))

    def test_convert_structure_unknown_target(self):
        """Test converting structure with unknown target."""
        with self.assertRaises(ValueError):
            utils_module.convert_structure(self.test_atoms, target="unknown").run()

    def test_extract_outputs_from_EngineOutputs(self):
        """Test extracting outputs from EngineOutput objects."""
        # Create mock EngineOutput objects
        mock_output1 = Mock()
        mock_output1.convergence = True
        mock_output1.final_energy = 1.0
        mock_output1.final_volume = 10.0

        mock_output2 = Mock()
        mock_output2.convergence = True
        mock_output2.final_energy = 2.0
        mock_output2.final_volume = 20.0

        mock_output3 = Mock()
        mock_output3.convergence = False  # Not converged
        mock_output3.final_energy = 3.0
        mock_output3.final_volume = 30.0

        engine_outputs = [mock_output1, mock_output2, mock_output3]
        keys = ["final_energy", "final_volume"]

        # Test with only_converged=True
        result = utils_module.extract_outputs_from_EngineOutputs(
            engine_outputs, keys, only_converged=True
        )

        expected = {"final_energy": [1.0, 2.0], "final_volume": [10.0, 20.0]}
        self.assertEqual(result, expected)

        # Test with only_converged=False
        result = utils_module.extract_outputs_from_EngineOutputs(
            engine_outputs, keys, only_converged=False
        )

        expected = {"final_energy": [1.0, 2.0, 3.0], "final_volume": [10.0, 20.0, 30.0]}
        self.assertEqual(result, expected)

    def test_get_working_subdir_kwargs(self):
        """Test getting working subdirectory kwargs."""
        calc_kwargs = {"param1": "value1", "working_directory": "/base"}
        base_dir = "/base"
        new_dir = "subdir"

        result = utils_module.get_working_subdir_kwargs(
            calc_kwargs, base_dir, new_dir
        ).run()

        expected = {"param1": "value1", "working_directory": "/base/subdir"}
        self.assertEqual(result, expected)

    def test_get_calc_fn_calc_fn_kwargs_from_calculation_engine_with_engine(self):
        """Test getting calc function and kwargs from calculation engine."""
        from pyiron_workflow_atomistics.calculator import Engine

        class MockEngine(Engine):
            def __init__(self):
                self.get_calculate_fn = Mock()
                self.get_calculate_fn.return_value = (Mock(), {"param": "value"})

        mock_engine = MockEngine()

        result = utils_module.get_calc_fn_calc_fn_kwargs_from_calculation_engine(
            calculation_engine=mock_engine,
            structure=self.test_atoms,
            calc_structure_fn=None,
            calc_structure_fn_kwargs=None,
        ).run()

        mock_engine.get_calculate_fn.assert_called_once_with(self.test_atoms)
        self.assertEqual(len(result), 2)  # Should return (fn, kwargs)

    def test_get_calc_fn_calc_fn_kwargs_from_calculation_engine_without_engine(self):
        """Test getting calc function and kwargs without calculation engine."""
        mock_fn = Mock()
        mock_kwargs = {"param": "value"}

        result = utils_module.get_calc_fn_calc_fn_kwargs_from_calculation_engine(
            calculation_engine=None,
            structure=self.test_atoms,
            calc_structure_fn=mock_fn,
            calc_structure_fn_kwargs=mock_kwargs,
        ).run()

        self.assertEqual(result, (mock_fn, mock_kwargs))

    def test_add_string(self):
        """Test adding strings."""
        result = utils_module.add_string("hello", " world").run()
        self.assertEqual(result, "hello world")

    def test_modify_dataclass(self):
        """Test modifying dataclass instance."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str
            field2: int

        original = TestClass("value1", 42)

        result = utils_module.modify_dataclass(original, "field1", "new_value").run()

        # Original should be unchanged
        self.assertEqual(original.field1, "value1")
        self.assertEqual(original.field2, 42)

        # Result should have new value
        self.assertEqual(result.field1, "new_value")
        self.assertEqual(result.field2, 42)

    def test_modify_dataclass_unknown_field(self):
        """Test modifying dataclass with unknown field."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str

        original = TestClass("value1")

        with self.assertRaises(KeyError):
            utils_module.modify_dataclass(original, "unknown_field", "value").run()

    def test_modify_dataclass_multi(self):
        """Test modifying dataclass with multiple fields."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str
            field2: int
            field3: float

        original = TestClass("value1", 42, 3.14)

        result = utils_module.modify_dataclass_multi(
            original, ["field1", "field3"], ["new_value1", 2.71]
        ).run()

        # Original should be unchanged
        self.assertEqual(original.field1, "value1")
        self.assertEqual(original.field2, 42)
        self.assertEqual(original.field3, 3.14)

        # Result should have new values
        self.assertEqual(result.field1, "new_value1")
        self.assertEqual(result.field2, 42)
        self.assertEqual(result.field3, 2.71)

    def test_modify_dataclass_multi_length_mismatch(self):
        """Test modifying dataclass with mismatched lengths."""
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str

        original = TestClass("value1")

        with self.assertRaises(ValueError):
            utils_module.modify_dataclass_multi(
                original, ["field1", "field2"], ["value1"]
            ).run()

    def test_modify_dict(self):
        """Test modifying dictionary."""
        original = {"param1": "value1", "param2": "value2", "param3": "value3"}

        result = utils_module.modify_dict(original, {"param2": "new_value2"}).run()

        # Original should be unchanged
        self.assertEqual(
            original, {"param1": "value1", "param2": "value2", "param3": "value3"}
        )

        # Result should have new value
        expected = {"param1": "value1", "param2": "new_value2", "param3": "value3"}
        self.assertEqual(result, expected)

    def test_modify_dict_unknown_key(self):
        """Test modifying dictionary with unknown key."""
        original = {"param1": "value1"}

        with self.assertRaises(KeyError):
            utils_module.modify_dict(original, {"unknown_param": "value"}).run()

    def test_get_subdirpaths(self):
        """Test getting subdirectory paths."""
        parent_dir = "/base"
        subdirs = ["calc1", "calc2", "calc3"]

        result = utils_module.get_subdirpaths(parent_dir, subdirs).run()

        expected = ["/base/calc1", "/base/calc2", "/base/calc3"]
        self.assertEqual(result, expected)

    def test_get_per_atom_quantity(self):
        """Test getting per-atom quantity."""
        quantity = 100.0
        structure = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = utils_module.get_per_atom_quantity(quantity, structure).run()

        expected = 25.0  # 100.0 / 4 atoms
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
