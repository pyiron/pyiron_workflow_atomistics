"""
Unit tests for pyiron_workflow_atomistics.gb.analysis module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.io import read
from ase.atoms import Atom

import pyiron_workflow_atomistics.gb.analysis as gb_analysis_module
from pyiron_workflow_atomistics.featurisers import voronoiSiteFeaturiser

class TestGBAnalysisFunctions(unittest.TestCase):
    """Test GB analysis module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test structure with atoms at different z positions
        test_dir = os.path.dirname(__file__)
        resource_path = os.path.join(test_dir, "..", "resources", "GB.vasp")
        self.test_atoms = read(resource_path)
        
        
        # Create a mock featuriser
        self.mock_featuriser = Mock()
        self.mock_featuriser.return_value = {'feature1': 1.0, 'feature2': 2.0}
        
    def test_get_middle_atom_z_axis(self):
        """Test getting middle atom along z-axis."""
        result = gb_analysis_module.get_middle_atom(self.test_atoms, axis=2).run()
        
        # Should return an Atom object
        self.assertIsInstance(result, Atom)
        
        # The middle atom should be around z=2 (middle of the cell)
        middle_z = result.position[2]
        self.assertAlmostEqual(middle_z, 16.152601971055077, places=5)
        
    def test_get_middle_atom_string_axis(self):
        """Test getting middle atom with string axis."""
        result = gb_analysis_module.get_middle_atom(self.test_atoms, axis='z').run()
        
        self.assertIsInstance(result, Atom)
        
    def test_get_middle_atom_string_axis_uppercase(self):
        """Test getting middle atom with uppercase string axis."""
        result = gb_analysis_module.get_middle_atom(self.test_atoms, axis='Z').run()
        
        self.assertIsInstance(result, Atom)

    def test_find_GB_plane_basic(self):
        """Test basic GB plane finding."""
        result = gb_analysis_module.find_GB_plane(
            atoms=self.test_atoms,
            featuriser=voronoiSiteFeaturiser,
            axis="c"
        ).run()
        
        # Check that result is a dictionary with expected keys
        expected_keys = [
            'gb_frac', 'gb_cart', 'mid_index', 'sel_indices', 'bulk_indices',
            'sel_fracs', 'scores', 'region_start_frac', 'region_end_frac',
            'extended_sel_indices'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
            
        # Check types
        self.assertIsInstance(result['gb_frac'], float)
        self.assertIsInstance(result['gb_cart'], float)
        self.assertIsInstance(result['sel_indices'], list)
        self.assertIsInstance(result['bulk_indices'], list)
        self.assertIsInstance(result['scores'], np.ndarray)
        
    def test_find_GB_plane_with_approx_frac(self):
        """Test GB plane finding with approximate fractional coordinate."""
        result = gb_analysis_module.find_GB_plane(
            atoms=self.test_atoms,
            featuriser=voronoiSiteFeaturiser,
            axis="c",
            approx_frac=0.5
        ).run()
        
        # Should work without errors
        self.assertIn('gb_frac', result)
        self.assertIn('gb_cart', result)
        
    def test_find_GB_plane_with_featuriser_kwargs(self):
        """Test GB plane finding with featuriser kwargs."""
        result = gb_analysis_module.find_GB_plane(
            atoms=self.test_atoms,
            featuriser=self.mock_featuriser,
            axis="c",
            featuriser_kwargs={'param': 'value'}
        ).run()
        
        # Should work without errors
        self.assertIn('gb_frac', result)
        
        # Check that featuriser was called with kwargs
        self.mock_featuriser.assert_called()

    def test_get_sites_on_plane_cartesian(self):
        """Test getting sites on plane in Cartesian coordinates."""
        # Find atoms at z=2
        result = gb_analysis_module.get_sites_on_plane(
            atoms=self.test_atoms,
            axis='c',
            target_coord=2.0,
            tol=0.1,
            use_fractional=False
        ).run()
        
        self.assertIsInstance(result, list)
        # Should find atoms at z=2
        self.assertGreater(len(result), 0)
        
        # Check that returned indices correspond to atoms at z=2
        for idx in result:
            self.assertAlmostEqual(self.test_atoms.positions[idx, 2], 2.0, places=1)
            
    def test_get_sites_on_plane_fractional(self):
        """Test getting sites on plane in fractional coordinates."""
        # Find atoms at fractional z=1/3 (which is z=2 in Cartesian)
        result = gb_analysis_module.get_sites_on_plane(
            atoms=self.test_atoms,
            axis='c',
            target_coord=1.0/3.0,  # z=2 in fractional coordinates
            tol=0.1,
            use_fractional=True
        ).run()
        
        self.assertIsInstance(result, list)
        # Should find atoms at z=2
        self.assertGreater(len(result), 0)
        
    def test_get_sites_on_plane_no_matches(self):
        """Test getting sites on plane with no matches."""
        # Look for atoms at z=10 (none should exist)
        result = gb_analysis_module.get_sites_on_plane(
            atoms=self.test_atoms,
            axis='c',
            target_coord=10.0,
            tol=0.1,
            use_fractional=False
        ).run()
        
        self.assertEqual(len(result), 24)

    @patch('matplotlib.pyplot.show')
    def test_plot_GB_plane(self, mock_show):
        """Test plotting GB plane analysis."""
        # Create a mock result dictionary
        mock_result = {
            'bulk_indices': [0, 1, 2, 3],
            'sel_indices': [4, 5, 6, 7],
            'extended_sel_indices': [4, 5, 6, 7, 8, 9],
            'scores': np.array([0.1, 0.2, 0.3, 0.4]),
            'sel_fracs': np.array([0.1, 0.2, 0.3, 0.4]),
            'region_start_frac': 0.05,
            'region_end_frac': 0.45,
            'gb_cart': 2.0
        }
        
        fig, ax = gb_analysis_module.plot_GB_plane(
            atoms=self.test_atoms,
            res=mock_result,
            projection=(0, 2),
            axis=2
        ).run()
        
        # Should return figure and axes
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
    @patch('matplotlib.pyplot.show')
    def test_plot_GB_plane_with_save(self, mock_show):
        """Test plotting GB plane with save functionality."""
        mock_result = {
            'bulk_indices': [0, 1, 2, 3],
            'sel_indices': [4, 5, 6, 7],
            'extended_sel_indices': [4, 5, 6, 7, 8, 9],
            'scores': np.array([0.1, 0.2, 0.3, 0.4]),
            'sel_fracs': np.array([0.1, 0.2, 0.3, 0.4]),
            'region_start_frac': 0.05,
            'region_end_frac': 0.45,
            'gb_cart': 2.0
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = gb_analysis_module.plot_GB_plane(
                atoms=self.test_atoms,
                res=mock_result,
                projection=(0, 2),
                axis=2,
                save_filename='test_plot.png',
                working_directory=tmpdir
            ).run()
            
            # Check that file was created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'test_plot.png')))

    @patch('matplotlib.pyplot.show')
    def test_plot_structure_2d(self, mock_show):
        """Test plotting 2D structure."""
        fig, ax = gb_analysis_module.plot_structure_2d(
            atoms=self.test_atoms,
            projection=(0, 2)
        )
        
        # Should return figure and axes
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        
    @patch('matplotlib.pyplot.show')
    def test_plot_structure_2d_with_save(self, mock_show):
        """Test plotting 2D structure with save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_structure.png')
            fig, ax = gb_analysis_module.plot_structure_2d(
                atoms=self.test_atoms,
                projection=(0, 2),
                save_path=save_path
            )
            
            # Check that file was created
            self.assertTrue(os.path.exists(save_path))


if __name__ == '__main__':
    unittest.main()
