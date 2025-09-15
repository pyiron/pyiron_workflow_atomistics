"""
Unit tests for pyiron_workflow_atomistics.featurisers module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
import pandas as pd

import pyiron_workflow_atomistics.featurisers as featurisers_module


class TestFeaturisersFunctions(unittest.TestCase):
    """Test featurisers module functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test structure
        positions = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ]
        self.test_atoms = Atoms('H8', positions=positions, cell=[2, 2, 2])

    @patch('pyiron_workflow_atomistics.featurisers.VoronoiNN')
    def test_voronoiSiteFeaturiser(self, mock_voronoi_nn):
        """Test Voronoi site featuriser."""
        # Mock the VoronoiNN
        mock_nn = Mock()
        mock_voronoi_nn.return_value = mock_nn
        
        # Mock the coordination number
        mock_nn.get_cn.return_value = 12
        
        # Mock the Voronoi polyhedra
        mock_polyhedra = {
            'neighbor1': {
                'volume': 1.0,
                'n_verts': 6,
                'face_dist': 2.0,
                'area': 0.5
            },
            'neighbor2': {
                'volume': 2.0,
                'n_verts': 8,
                'face_dist': 2.5,
                'area': 0.8
            }
        }
        mock_nn.get_voronoi_polyhedra.return_value = mock_polyhedra
        
        result = featurisers_module.voronoiSiteFeaturiser(self.test_atoms, 0)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that expected keys are present
        expected_keys = [
            'VorNN_CoordNo', 'VorNN_tot_vol', 'VorNN_tot_area',
            'VorNN_volumes_std', 'VorNN_volumes_mean', 'VorNN_volumes_min', 'VorNN_volumes_max',
            'VorNN_vertices_std', 'VorNN_vertices_mean', 'VorNN_vertices_min', 'VorNN_vertices_max',
            'VorNN_areas_std', 'VorNN_areas_mean', 'VorNN_areas_min', 'VorNN_areas_max',
            'VorNN_distances_std', 'VorNN_distances_mean', 'VorNN_distances_min', 'VorNN_distances_max'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
            
        # Check specific values
        self.assertEqual(result['VorNN_CoordNo'], 12)
        self.assertEqual(result['VorNN_tot_vol'], 3.0)  # 1.0 + 2.0
        self.assertEqual(result['VorNN_tot_area'], 1.3)  # 0.5 + 0.8

    def test_distanceMatrixSiteFeaturiser(self):
        """Test distance matrix site featuriser."""
        result = featurisers_module.distanceMatrixSiteFeaturiser(self.test_atoms, 0, k=4)
        
        # Check that result is a dictionary
        self.assertIsInstance(result, dict)
        
        # Check that expected keys are present
        expected_keys = [
            'Dist_knn_1', 'Dist_knn_2', 'Dist_knn_3', 'Dist_knn_4',
            'Dist_min', 'Dist_mean', 'Dist_std', 'Dist_max'
        ]
        for key in expected_keys:
            self.assertIn(key, result)
            
        # Check that distances are reasonable
        self.assertGreater(result['Dist_min'], 0)
        self.assertGreater(result['Dist_mean'], result['Dist_min'])
        self.assertGreater(result['Dist_max'], result['Dist_mean'])
        
    def test_distanceMatrixSiteFeaturiser_insufficient_neighbors(self):
        """Test distance matrix featuriser with insufficient neighbors."""
        # Create a structure with only 2 atoms
        simple_atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        
        result = featurisers_module.distanceMatrixSiteFeaturiser(simple_atoms, 0, k=4)
        
        # Should still work, but with NaN values for missing neighbors
        self.assertIn('Dist_knn_1', result)
        self.assertIn('Dist_knn_4', result)
        
        # First neighbor should be valid
        self.assertIsInstance(result['Dist_knn_1'], float)
        self.assertGreater(result['Dist_knn_1'], 0)

    def test_soapSiteFeaturiser(self):
        """Test SOAP site featuriser."""
        # Skip this test if dscribe is not installed
        try:
            from dscribe.descriptors import SOAP
        except ImportError:
            self.skipTest("dscribe is not installed, skipping SOAP featuriser test.")

        with patch('dscribe.descriptors.SOAP') as mock_soap:
            # Mock the SOAP descriptor
            mock_descriptor = Mock()
            mock_soap.return_value = mock_descriptor

            # Mock the create method
            mock_features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            mock_descriptor.create.return_value = mock_features

            result = featurisers_module.soapSiteFeaturiser(
                self.test_atoms, 
                site_indices=[0, 1], 
                r_cut=6.0, 
                n_max=10, 
                l_max=10
            )

            # Check that SOAP was called with correct parameters
            mock_soap.assert_called_once_with(
                species=self.test_atoms.get_chemical_symbols(),
                r_cut=6.0,
                n_max=10,
                l_max=10,
                periodic=False
            )

            # Check that create was called
            mock_descriptor.create.assert_called_once_with(
                self.test_atoms, 
                centers=[0, 1], 
                n_jobs=-1
            )

            # Check result
            np.testing.assert_array_equal(result, mock_features)
        
    def test_soapSiteFeaturiser_import_error(self):
        """Test SOAP featuriser when dscribe is not available."""
        with patch.dict('sys.modules', {'dscribe': None, 'dscribe.descriptors': None}):
            with self.assertRaises(ImportError) as cm:
                featurisers_module.soapSiteFeaturiser(
                    self.test_atoms,
                    site_indices=[0, 1]
                )
            
            # Optionally check the error message
            self.assertIn("dscribe is not installed", str(cm.exception))

    def test_summarize_cosine_groups(self):
        """Test cosine similarity grouping."""
        # Create test matrix with similar rows
        A = np.array([
            [1, 0, 0],  # Row 0
            [0.9, 0.1, 0],  # Row 1 - similar to row 0
            [0, 1, 0],  # Row 2
            [0, 0.9, 0.1],  # Row 3 - similar to row 2
            [0, 0, 1],  # Row 4 - unique
        ])
        
        result = featurisers_module.summarize_cosine_groups(A, threshold=0.9)
        
        # Check that result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check that it has expected columns
        self.assertIn('rep', result.columns)
        self.assertIn('same', result.columns)
        
        # Should group similar rows
        self.assertGreater(len(result), 0)
        
    def test_summarize_cosine_groups_with_ids(self):
        """Test cosine similarity grouping with external IDs."""
        A = np.array([
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0, 1, 0],
        ])
        ids = ['atom_0', 'atom_1', 'atom_2']
        
        result = featurisers_module.summarize_cosine_groups(A, threshold=0.9, ids=ids)
        
        # Check that IDs are used correctly
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('rep', result.columns)
        self.assertIn('same', result.columns)
        
    def test_summarize_cosine_groups_exclude_singletons(self):
        """Test cosine similarity grouping excluding singletons."""
        A = np.array([
            [1, 0, 0],
            [0.9, 0.1, 0],
            [0, 0, 1],  # This should be excluded as singleton
        ])
        
        result = featurisers_module.summarize_cosine_groups(
            A, threshold=0.9, include_singletons=False
        )
        
        # Should only include groups with multiple members
        for _, row in result.iterrows():
            self.assertGreaterEqual(len(row['same']) + 1, 2)  # rep + same >= 2

    def test_pca_whiten_fit(self):
        """Test PCA whitening with fitting."""
        # Create test data
        X = np.random.randn(100, 10)
        
        Z, model = featurisers_module.pca_whiten(X, n_components=0.95, method='pca')
        
        # Check that result is reasonable
        self.assertIsInstance(Z, np.ndarray)
        self.assertIsInstance(model, dict)
        
        # Check model keys
        expected_keys = ['mu', 'V', 'eigvals', 'k', 'method', 'eps']
        for key in expected_keys:
            self.assertIn(key, model)
            
        # Check dimensions
        self.assertEqual(Z.shape[0], X.shape[0])
        self.assertLessEqual(Z.shape[1], X.shape[1])
        
    def test_pca_whiten_transform(self):
        """Test PCA whitening with existing model."""
        # Create test data
        X = np.random.randn(100, 10)
        
        # First fit a model
        _, model = featurisers_module.pca_whiten(X, n_components=0.95, method='pca')
        
        # Then transform new data
        X_new = np.random.randn(50, 10)
        Z, model_out = featurisers_module.pca_whiten(X_new, model=model)
        
        # Check that result is reasonable
        self.assertIsInstance(Z, np.ndarray)
        self.assertEqual(Z.shape[0], X_new.shape[0])
        
    def test_pca_whiten_zca(self):
        """Test ZCA whitening."""
        X = np.random.randn(100, 10)
        
        Z, model = featurisers_module.pca_whiten(X, n_components=0.95, method='zca')
        
        # Check that result is reasonable
        self.assertIsInstance(Z, np.ndarray)
        self.assertIsInstance(model, dict)
        self.assertEqual(model['method'], 'zca')
        
    def test_pca_whiten_invalid_method(self):
        """Test PCA whitening with invalid method."""
        X = np.random.randn(100, 10)
        
        with self.assertRaises(ValueError):
            featurisers_module.pca_whiten(X, method='invalid')
            
    def test_pca_whiten_invalid_n_components(self):
        """Test PCA whitening with invalid n_components."""
        X = np.random.randn(100, 10)
        
        with self.assertRaises(ValueError):
            featurisers_module.pca_whiten(X, n_components=1.5)  # > 1.0


if __name__ == '__main__':
    unittest.main()
