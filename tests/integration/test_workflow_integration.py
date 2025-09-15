"""
Integration tests for pyiron_workflow_atomistics workflow components.
"""

import unittest
from unittest.mock import Mock

import numpy as np
from ase import Atoms
from ase.build import bulk

import pyiron_workflow_atomistics.bulk as bulk_module
import pyiron_workflow_atomistics.calculator as calc_module
import pyiron_workflow_atomistics.gb.analysis as gb_analysis_module
import pyiron_workflow_atomistics.gb.cleavage as gb_cleavage_module
import pyiron_workflow_atomistics.utils as utils_module


class TestWorkflowIntegration(unittest.TestCase):
    """Test integration between workflow components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test structures
        self.simple_atoms = bulk("Al", crystalstructure="fcc", a=4.0)
        self.layered_atoms = self._create_layered_structure()

    def _create_layered_structure(self):
        """Create a layered structure for GB testing."""
        positions = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],  # z=0 layer
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2],
            [1, 1, 2],  # z=2 layer
            [0, 0, 4],
            [1, 0, 4],
            [0, 1, 4],
            [1, 1, 4],  # z=4 layer
            [0, 0, 6],
            [1, 0, 6],
            [0, 1, 6],
            [1, 1, 6],  # z=6 layer
        ]
        return Atoms("H16", positions=positions, cell=[2, 2, 8])

    def test_calculator_validation_integration(self):
        """Test calculator validation integration."""
        # Test valid engine configuration
        mock_engine = Mock()
        result = calc_module.validate_calculation_inputs(calculation_engine=mock_engine)
        self.assertTrue(result)

        # Test valid function configuration
        mock_fn = Mock()
        mock_kwargs = {"param": "value"}
        result = calc_module.validate_calculation_inputs(
            calc_structure_fn=mock_fn, calc_structure_fn_kwargs=mock_kwargs
        )
        self.assertTrue(result)

        # Test invalid configuration
        with self.assertRaises(ValueError):
            calc_module.validate_calculation_inputs()

    def test_bulk_structure_generation_integration(self):
        """Test bulk structure generation integration."""
        # Generate structures with different strain types
        iso_structures = bulk_module.generate_structures(
            base_structure=self.simple_atoms,
            axes=["iso"],
            strain_range=(-0.1, 0.1),
            num_points=5,
        )

        axis_structures = bulk_module.generate_structures(
            base_structure=self.simple_atoms,
            axes=["a", "b", "c"],
            strain_range=(-0.05, 0.05),
            num_points=3,
        )

        # Check that structures are generated correctly
        self.assertEqual(len(iso_structures), 5)
        self.assertEqual(len(axis_structures), 3)

        # Check that all structures have the same number of atoms
        for struct in iso_structures + axis_structures:
            self.assertEqual(len(struct), len(self.simple_atoms))

    def test_equation_of_state_integration(self):
        """Test equation of state integration."""
        # Create test data
        volumes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        energies = 0.1 * (volumes - 14.0) ** 2 + 5.0

        # Test different EOS types
        eos_types = ["sj", "birchmurnaghan"]
        for eos_type in eos_types:
            e0, v0, B = bulk_module.equation_of_state(
                energies, volumes, eos_type=eos_type
            )

            # Check that results are reasonable
            self.assertIsInstance(e0, float)
            self.assertIsInstance(v0, float)
            self.assertIsInstance(B, float)
            self.assertGreater(B, 0)

    def test_structure_conversion_integration(self):
        """Test structure conversion integration."""
        # Test ASE to pymatgen conversion
        pmg_struct = utils_module.convert_structure(
            self.simple_atoms, target="pymatgen"
        )

        # Test pymatgen to ASE conversion
        ase_struct = utils_module.convert_structure(pmg_struct, target="ase")

        # Check that conversion preserves structure
        self.assertEqual(len(ase_struct), len(self.simple_atoms))
        np.testing.assert_array_almost_equal(
            ase_struct.get_positions(), self.simple_atoms.get_positions()
        )

    def test_gb_analysis_integration(self):
        """Test GB analysis integration."""
        # Create a mock featuriser
        mock_featuriser = Mock()
        mock_featuriser.return_value = {"feature1": 1.0, "feature2": 2.0}

        # Test GB plane finding
        result = gb_analysis_module.find_GB_plane(
            atoms=self.layered_atoms, featuriser=mock_featuriser, axis="c"
        )

        # Check that result contains expected keys
        expected_keys = [
            "gb_frac",
            "gb_cart",
            "mid_index",
            "sel_indices",
            "bulk_indices",
            "sel_fracs",
            "scores",
            "region_start_frac",
            "region_end_frac",
            "extended_sel_indices",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Test getting sites on plane
        sites = gb_analysis_module.get_sites_on_plane(
            atoms=self.layered_atoms,
            axis="c",
            target_coord=3.0,
            tol=0.5,
            use_fractional=False,
        )

        self.assertIsInstance(sites, list)

    def test_gb_cleavage_integration(self):
        """Test GB cleavage integration."""
        # Test finding viable cleavage planes
        planes = gb_cleavage_module.find_viable_cleavage_planes_around_plane(
            structure=self.layered_atoms,
            axis="c",
            plane_coord=3.0,
            coord_tol=1.0,
            layer_tolerance=0.1,
            fractional=False,
        )

        self.assertIsInstance(planes, list)

        # Test cleaving structure
        if planes:  # Only test if planes were found
            cleaved = gb_cleavage_module.cleave_axis_aligned(
                structure=self.layered_atoms,
                axis="c",
                plane_coord=planes[0],
                separation=2.0,
                use_fractional=False,
            )

            self.assertIsInstance(cleaved, Atoms)
            self.assertEqual(len(cleaved), len(self.layered_atoms))

    def test_full_cleavage_workflow_integration(self):
        """Test full cleavage workflow integration."""
        # Test complete cleavage workflow
        cleaved_structures, cleavage_plane_coords = (
            gb_cleavage_module.cleave_gb_structure(
                base_structure=self.layered_atoms,
                axis_to_cleave="c",
                target_coord=3.0,
                tol=0.5,
                cleave_region_halflength=2.0,
                layer_tolerance=0.1,
                separation=2.0,
                use_fractional=False,
            )
        )

        self.assertIsInstance(cleaved_structures, list)
        self.assertIsInstance(cleavage_plane_coords, list)
        self.assertEqual(len(cleaved_structures), len(cleavage_plane_coords))

        # Each cleaved structure should have the same number of atoms
        for struct in cleaved_structures:
            self.assertIsInstance(struct, Atoms)
            self.assertEqual(len(struct), len(self.layered_atoms))

    def test_calculator_kwargs_integration(self):
        """Test calculator kwargs integration."""
        # Test filling in default kwargs
        calc_kwargs = {"param1": "value1"}
        default_values = {"param2": "default2", "properties": ["energy", "forces"]}

        result = calc_module.fillin_default_calckwargs(calc_kwargs, default_values)

        # Check that properties are converted to tuple
        self.assertIsInstance(result["properties"], tuple)
        self.assertEqual(result["properties"], ("energy", "forces"))

        # Test generating kwargs variants
        variants = calc_module.generate_kwargs_variants(
            result, "param1", ["new1", "new2", "new3"]
        )

        self.assertEqual(len(variants), 3)
        for i, variant in enumerate(variants):
            self.assertEqual(variant["param1"], f"new{i+1}")

    def test_utils_integration(self):
        """Test utils module integration."""
        # Test dataclass modification
        from dataclasses import dataclass

        @dataclass
        class TestClass:
            field1: str
            field2: int

        original = TestClass("value1", 42)

        # Test single modification
        modified = utils_module.modify_dataclass(original, "field1", "new_value")
        self.assertEqual(modified.field1, "new_value")
        self.assertEqual(modified.field2, 42)

        # Test multiple modifications
        modified_multi = utils_module.modify_dataclass_multi(
            original, ["field1", "field2"], ["new_value1", 100]
        )
        self.assertEqual(modified_multi.field1, "new_value1")
        self.assertEqual(modified_multi.field2, 100)

        # Test dictionary modification
        original_dict = {"param1": "value1", "param2": "value2"}
        modified_dict = utils_module.modify_dict(
            original_dict, {"param2": "new_value2"}
        )
        self.assertEqual(modified_dict["param2"], "new_value2")
        self.assertEqual(original_dict["param2"], "value2")  # Original unchanged

    def test_per_atom_quantity_integration(self):
        """Test per-atom quantity calculation integration."""
        quantity = 100.0
        structure = Atoms("H4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

        per_atom = utils_module.get_per_atom_quantity(quantity, structure)

        self.assertEqual(per_atom, 25.0)  # 100.0 / 4 atoms

    def test_working_directory_integration(self):
        """Test working directory handling integration."""
        calc_kwargs = {"param1": "value1", "working_directory": "/base"}
        base_dir = "/base"
        new_dir = "subdir"

        result = utils_module.get_working_subdir_kwargs(calc_kwargs, base_dir, new_dir)

        expected = {"param1": "value1", "working_directory": "/base/subdir"}
        self.assertEqual(result, expected)

        # Test getting subdirectory paths
        parent_dir = "/base"
        subdirs = ["calc1", "calc2", "calc3"]

        paths = utils_module.get_subdirpaths(parent_dir, subdirs)

        expected_paths = ["/base/calc1", "/base/calc2", "/base/calc3"]
        self.assertEqual(paths, expected_paths)


if __name__ == "__main__":
    unittest.main()
