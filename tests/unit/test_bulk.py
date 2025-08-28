"""
Unit tests for pyiron_workflow_atomistics.bulk module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.build import bulk

import pyiron_workflow_atomistics.bulk as bulk_module
import warnings

class TestBulkFunctions(unittest.TestCase):
    """Test bulk module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple cubic structure for testing
        self.test_atoms = bulk('Al', crystalstructure='fcc', a=4.0, cubic=True)
        self.test_atoms = self.test_atoms.repeat((2, 2, 2))  # 32 atoms
        
    def test_generate_structures_iso(self):
        """Test generating structures with isotropic *linear* strain."""
        strain_range = (-0.1, 0.1)
        num_points = 5
        expected_strains = np.linspace(strain_range[0], strain_range[1], num_points)

        structures = bulk_module.generate_structures(
            base_structure=self.test_atoms,
            axes=['iso'],
            strain_range=strain_range,
            num_points=num_points
        ).run()

        # Count check
        self.assertEqual(len(structures), num_points)

        # Original cell metrics
        orig_cell = self.test_atoms.get_cell()
        a0, b0, c0 = orig_cell.lengths()
        alpha0, beta0, gamma0 = orig_cell.angles()
        orig_cell_array = np.array(orig_cell)  # 3x3

        for struct, eps in zip(structures, expected_strains):
            # Atom count unchanged
            self.assertEqual(len(struct), len(self.test_atoms))

            # Angle preservation (isotropic scaling should not change angles)
            alpha, beta, gamma = struct.get_cell().angles()
            self.assertAlmostEqual(alpha, alpha0, places=6)
            self.assertAlmostEqual(beta, beta0, places=6)
            self.assertAlmostEqual(gamma, gamma0, places=6)

            # Per-axis linear strain should match eps
            a, b, c = struct.get_cell().lengths()
            self.assertAlmostEqual((a - a0) / a0, eps, places=6)
            self.assertAlmostEqual((b - b0) / b0, eps, places=6)
            self.assertAlmostEqual((c - c0) / c0, eps, places=6)

            # (Equivalent, stronger) check: full cell scaled by (1+eps)
            scaled = (1.0 + eps) * orig_cell_array
            np.testing.assert_allclose(np.array(struct.get_cell()),
                                    scaled, rtol=1e-7, atol=1e-12)
            
    def test_generate_structures_axis_a(self):
        """Test generating structures with strain along axis a."""
        structures = bulk_module.generate_structures(
            base_structure=self.test_atoms,
            axes=['a'],
            strain_range=(-0.1, 0.1),
            num_points=4 # Really important this never lands on 0.0
        ).run()
        
        self.assertEqual(len(structures), 4)
        
        # Check that only the a-axis changes
        original_cell = self.test_atoms.get_cell()
        for i, struct in enumerate(structures):
            new_cell = struct.get_cell()
            # a-axis should change
            self.assertNotEqual(np.linalg.norm(new_cell[0]), np.linalg.norm(original_cell[0]))
            # b and c axes should remain the same
            np.testing.assert_array_almost_equal(new_cell[1], original_cell[1])
            np.testing.assert_array_almost_equal(new_cell[2], original_cell[2])
            
    def test_generate_structures_axis_b(self):
        """Test generating structures with strain along axis b."""
        structures = bulk_module.generate_structures(
            base_structure=self.test_atoms,
            axes=['b'],
            strain_range=(-0.05, 0.05),
            num_points=4
        ).run()
        
        self.assertEqual(len(structures), 4)
        
        original_cell = self.test_atoms.get_cell()
        for struct in structures:
            new_cell = struct.get_cell()
            # b-axis should change
            self.assertNotEqual(np.linalg.norm(new_cell[1]), np.linalg.norm(original_cell[1]))
            # a and c axes should remain the same
            np.testing.assert_array_almost_equal(new_cell[0], original_cell[0])
            np.testing.assert_array_almost_equal(new_cell[2], original_cell[2])
            
    def test_generate_structures_axis_c(self):
        """Test generating structures with strain along axis c."""
        structures = bulk_module.generate_structures(
            base_structure=self.test_atoms,
            axes=['c'],
            strain_range=(-0.05, 0.05),
            num_points=4
        ).run()
        
        self.assertEqual(len(structures), 4)
        
        original_cell = self.test_atoms.get_cell()
        for struct in structures:
            new_cell = struct.get_cell()
            # c-axis should change
            self.assertNotEqual(np.linalg.norm(new_cell[2]), np.linalg.norm(original_cell[2]))
            # a and b axes should remain the same
            np.testing.assert_array_almost_equal(new_cell[0], original_cell[0])
            np.testing.assert_array_almost_equal(new_cell[1], original_cell[1])
            
    def test_generate_structures_multiple_axes(self):
        """Test generating structures with strain along multiple axes."""
        structures = bulk_module.generate_structures(
            base_structure=self.test_atoms,
            axes=['a', 'b'],
            strain_range=(-0.1, 0.1),
            num_points=4
        ).run()
        
        self.assertEqual(len(structures), 4)
        
        original_cell = self.test_atoms.get_cell()
        for struct in structures:
            new_cell = struct.get_cell()
            # Both a and b axes should change
            self.assertNotEqual(np.linalg.norm(new_cell[0]), np.linalg.norm(original_cell[0]))
            self.assertNotEqual(np.linalg.norm(new_cell[1]), np.linalg.norm(original_cell[1]))
            # c-axis should remain the same
            np.testing.assert_array_almost_equal(new_cell[2], original_cell[2])
            
    def test_generate_structures_unknown_axis(self):
        strain_range = (-0.1, 0.1)
        num_points = 4

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            structures = bulk_module.generate_structures(
                base_structure=self.test_atoms,
                axes=['a', 'unknown', 'b'],
                strain_range=strain_range,
                num_points=num_points
            ).run()

        # One warning per generated structure for the unknown axis
        unknown_msgs = [
            str(rec.message) for rec in w
            if issubclass(rec.category, UserWarning)
            and "Unknown axis label: unknown" in str(rec.message)
        ]
        self.assertEqual(len(unknown_msgs), num_points)

        self.assertEqual(len(structures), num_points)

        # Cell checks: a and b scaled by (1+eps); c unchanged
        original_cell = self.test_atoms.get_cell()
        a0, b0, c0 = original_cell.lengths()
        epsilons = np.linspace(*strain_range, num_points)

        for struct, eps in zip(structures, epsilons):
            a, b, c = struct.get_cell().lengths()
            self.assertAlmostEqual(a, a0 * (1 + eps), places=6)
            self.assertAlmostEqual(b, b0 * (1 + eps), places=6)
            self.assertAlmostEqual(c, c0, places=6)

    def test_equation_of_state_sj(self):
        """Test equation of state fitting with SJ method."""
        # Create test data: simple quadratic E(V) relationship
        volumes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        energies = 0.1 * (volumes - 14.0)**2 + 5.0  # Parabola centered at V=14
        
        e0, v0, B = bulk_module.equation_of_state(energies, volumes, eos_type="sj").run()
        
        # Check that results are reasonable
        self.assertIsInstance(e0, float)
        self.assertIsInstance(v0, float)
        self.assertIsInstance(B, float)
        
        # For this simple case, v0 should be close to 14.0
        self.assertAlmostEqual(v0, 14.0, places=1)
        
        # B should be positive (bulk modulus)
        self.assertGreater(B, 0)
        
    def test_equation_of_state_birchmurnaghan(self):
        """Test equation of state fitting with Birch-Murnaghan method."""
        volumes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
        energies = 0.1 * (volumes - 14.0)**2 + 5.0
        
        e0, v0, B = bulk_module.equation_of_state(energies, volumes, eos_type="birchmurnaghan").run()
        
        # Check that results are reasonable
        self.assertIsInstance(e0, float)
        self.assertIsInstance(v0, float)
        self.assertIsInstance(B, float)
        
        # B should be positive
        self.assertGreater(B, 0)

    def test_get_bulk_structure_basic(self):
        """Test getting basic bulk structure."""
        struct = bulk_module.get_bulk_structure(name='Al').run()
        
        self.assertIsInstance(struct, Atoms)
        self.assertGreater(len(struct), 0)
        
    def test_get_bulk_structure_with_parameters(self):
        """Test getting bulk structure with specific parameters."""
        struct = bulk_module.get_bulk_structure(
            name='Al',
            crystalstructure='fcc',
            a=4.0,
            cubic=True
        ).run()
        
        self.assertIsInstance(struct, Atoms)
        self.assertAlmostEqual(np.linalg.norm(struct.get_cell()[0]), 4.0, places=2)
        
    def test_get_bulk_structure_cubic(self):
        """Test getting cubic bulk structure."""
        struct = bulk_module.get_bulk_structure(
            name='Al',
            cubic=True,
            a=4.0
        ).run()
        
        self.assertIsInstance(struct, Atoms)
        cell = struct.get_cell()
        # All cell vectors should have the same length for cubic
        lengths = [np.linalg.norm(cell[i]) for i in range(3)]
        self.assertAlmostEqual(lengths[0], lengths[1], places=2)
        self.assertAlmostEqual(lengths[1], lengths[2], places=2)

    def test_rattle_structure_with_rattle(self):
        """Test rattling structure with specified displacement."""
        original_positions = self.test_atoms.get_positions().copy()
        
        rattled = bulk_module.rattle_structure(self.test_atoms, rattle=0.1).run()
        
        # Check that positions have changed
        new_positions = rattled.get_positions()
        displacement = np.linalg.norm(new_positions - original_positions, axis=1)
        
        # Some atoms should have moved
        self.assertGreater(np.max(displacement), 0.01)
        
        # Original structure should be unchanged
        np.testing.assert_array_almost_equal(
            self.test_atoms.get_positions(), original_positions
        )
        
    def test_rattle_structure_no_rattle(self):
        """Test rattling structure with no displacement."""
        original_positions = self.test_atoms.get_positions().copy()
        
        rattled = bulk_module.rattle_structure(self.test_atoms, rattle=None).run()
        
        # Positions should be identical (just a copy)
        new_positions = rattled.get_positions()
        np.testing.assert_array_almost_equal(new_positions, original_positions)

    def test_get_cubic_equil_lat_param(self):
        """Test getting cubic equilibrium lattice parameter."""
        v0 = 64.0  # Volume for 4x4x4 cube
        
        a0 = bulk_module.get_cubic_equil_lat_param(v0).run()
        
        expected = 4.0  # 4^3 = 64
        self.assertAlmostEqual(a0, expected, places=2)

    # @patch('pyiron_workflow_atomistics.bulk.calculate_structure_node')
    # def test_evaluate_structures_with_engine(self, mock_calc):
    #     """Test evaluating structures with calculation engine."""
    #     # Mock the calculation engine
    #     mock_engine = Mock()
    #     mock_engine.calculate_fn.return_value = (Mock(), {'working_directory': '/tmp'})
        
    #     # Mock the calculation function
    #     mock_calc.node_function.return_value = Mock()
        
    #     structures = [self.test_atoms, self.test_atoms.copy()]
        
    #     with patch('os.makedirs'):
    #         result = bulk_module.evaluate_structures(
    #             structures=structures,
    #             calculation_engine=mock_engine
    #         ).run()
        
    #     self.assertEqual(len(result), 2)
    #     mock_calc.node_function.assert_called()

    # def test_evaluate_structures_validation_error(self):
    #     """Test that evaluate_structures raises validation error."""
    #     with self.assertRaises(ValueError):
    #         bulk_module.evaluate_structures(
    #             structures=[self.test_atoms],
    #             # Missing both engine and fn/kwargs
    #         ).run()


if __name__ == '__main__':
    unittest.main()
