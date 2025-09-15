"""
Unit tests for pyiron_workflow_atomistics.gb.cleavage module.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
import pandas as pd
from ase.lattice.cubic import BodyCenteredCubic as bcc
from ase.build import stack
import pyiron_workflow_atomistics.gb.cleavage as gb_cleavage_module
from pyiron_workflow_atomistics.structure_manipulator.tools import add_vacuum


class TestGBCleavageFunctions(unittest.TestCase):
    """Test GB cleavage module functions."""

    def setUp(self):
        """Set up test fixtures."""

        surface1 = [1, 1, 1]
        surface2 = [1, 1, -1]
        rotation_axis = [1, -1, 0]
        element = "Fe"
        lc = 2.8318488966083
        # GB_name = "S3-RA110-S1-11"

        # The minimum required length of the cell.
        req_length = 30

        v1 = list(-np.cross(rotation_axis, surface1))
        v2 = list(-np.cross(rotation_axis, surface2))

        length = 0
        n = 0

        vacuum_slab = 15
        while length < req_length:
            n += 1

            slab1 = bcc(
                symbol=element,
                latticeconstant=lc,
                directions=[rotation_axis, v1, surface1],
                size=[1, 1, n],
            )
            slab2 = bcc(
                symbol=element,
                latticeconstant=lc,
                directions=[rotation_axis, v2, surface2],
                size=[1, 1, n],
            )

            gb = stack(slab1, slab2)
            length = gb.cell[-1, -1]

            # Rattle the gb structure using rattle in ASE to perturb symmetry
            # gb.rattle()
            gb_pmg = add_vacuum(gb, vacuum_slab).run()
        self.test_atoms = gb_pmg

    def test_find_viable_cleavage_planes_around_plane_cartesian(self):
        """Test finding viable cleavage planes around a plane in Cartesian coordinates."""
        result = gb_cleavage_module.find_viable_cleavage_planes_around_plane(
            structure=self.test_atoms,
            axis="c",
            plane_coord=20,  # Between z=2 and z=4 layers
            coord_tol=1.0,
            layer_tolerance=0.1,
            fractional=False,
        ).run()

        self.assertIsInstance(result, list)
        # Should find at least one viable plane (between z=2 and z=4)
        self.assertGreater(len(result), 0)

    def test_find_viable_cleavage_planes_around_plane_fractional(self):
        """Test finding viable cleavage planes around a plane in fractional coordinates."""
        result = gb_cleavage_module.find_viable_cleavage_planes_around_plane(
            structure=self.test_atoms,
            axis="c",
            plane_coord=0.5,  # 3.0/8.0 in fractional coordinates
            coord_tol=0.125,  # 1.0/8.0 in fractional coordinates
            layer_tolerance=0.1,
            fractional=True,
        ).run()

        self.assertIsInstance(result, list)

    def test_find_viable_cleavage_planes_around_site_cartesian(self):
        """Test finding viable cleavage planes around a site in Cartesian coordinates."""
        result = gb_cleavage_module.find_viable_cleavage_planes_around_site(
            structure=self.test_atoms,
            axis="c",
            site_index=4,  # Atom at z=2
            site_dist_threshold=2.0,
            layer_tolerance=0.1,
            fractional=False,
        ).run()

        self.assertIsInstance(result, list)

    def test_find_viable_cleavage_planes_around_site_fractional(self):
        """Test finding viable cleavage planes around a site in fractional coordinates."""
        result = gb_cleavage_module.find_viable_cleavage_planes_around_site(
            structure=self.test_atoms,
            axis="c",
            site_index=4,  # Atom at z=2
            site_dist_threshold=0.25,  # 2.0/8.0 in fractional coordinates
            layer_tolerance=0.1,
            fractional=True,
        ).run()

        self.assertIsInstance(result, list)

    def test_cleave_axis_aligned_cartesian(self):
        """Test cleaving structure along axis in Cartesian coordinates."""
        result = gb_cleavage_module.cleave_axis_aligned(
            structure=self.test_atoms,
            axis="c",
            plane_coord=20,
            separation=2.0,
            use_fractional=False,
        ).run()

        self.assertIsInstance(result, Atoms)
        self.assertEqual(len(result), len(self.test_atoms))

        # Check that atoms have been separated
        positions = result.get_positions()
        z_coords = positions[:, 2]

        # Should have atoms on both sides of the cleavage plane
        self.assertTrue(np.any(z_coords < 20))  # Below plane
        self.assertTrue(np.any(z_coords > 20))  # Above plane

    def test_cleave_axis_aligned_fractional(self):
        """Test cleaving structure along axis in fractional coordinates."""
        result = gb_cleavage_module.cleave_axis_aligned(
            structure=self.test_atoms,
            axis="c",
            plane_coord=0.5,  # 3.0/8.0 in fractional coordinates
            separation=2.0,
            use_fractional=True,
        ).run()

        self.assertIsInstance(result, Atoms)
        self.assertEqual(len(result), len(self.test_atoms))

    def test_cleave_axis_aligned_axis_a(self):
        """Test cleaving structure along axis a."""
        result = gb_cleavage_module.cleave_axis_aligned(
            structure=self.test_atoms,
            axis="a",
            plane_coord=1.0,
            separation=1.0,
            use_fractional=False,
        ).run()

        self.assertIsInstance(result, Atoms)

        # Check that atoms have been separated along x-axis
        positions = result.get_positions()
        x_coords = positions[:, 0]
        self.assertTrue(np.any(x_coords < 1.0))  # Left of plane
        self.assertTrue(np.any(x_coords > 1.0))  # Right of plane

    @patch("matplotlib.pyplot.show")
    def test_plot_structure_with_cleavage(self, mock_show):
        """Test plotting structure with cleavage planes."""
        cleavage_planes = [2.0, 4.0]

        fig, ax = gb_cleavage_module.plot_structure_with_cleavage(
            structure=self.test_atoms,
            cleavage_planes=cleavage_planes,
            projection=(0, 2),
        ).run()

        # Should return figure and axes
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

    @patch("matplotlib.pyplot.show")
    def test_plot_structure_with_cleavage_with_save(self, mock_show):
        """Test plotting structure with cleavage planes and saving."""
        cleavage_planes = [2.0, 4.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_cleavage.png")
            fig, ax = gb_cleavage_module.plot_structure_with_cleavage(
                structure=self.test_atoms,
                cleavage_planes=cleavage_planes,
                projection=(0, 2),
                save_path=save_path,
            ).run()

            # Check that file was created
            self.assertTrue(os.path.exists(save_path))

    def test_cleave_gb_structure(self):
        """Test cleaving GB structure."""
        cleaved_structures, cleavage_plane_coords = (
            gb_cleavage_module.cleave_gb_structure(
                base_structure=self.test_atoms,
                axis_to_cleave="c",
                target_coord=20.0,
                tol=0.5,
                cleave_region_halflength=2.0,
                layer_tolerance=0.1,
                separation=2.0,
                use_fractional=False,
            ).run()
        )

        self.assertIsInstance(cleaved_structures, list)
        self.assertIsInstance(cleavage_plane_coords, list)
        self.assertEqual(len(cleaved_structures), len(cleavage_plane_coords))

        # Each cleaved structure should have the same number of atoms
        for struct in cleaved_structures:
            self.assertIsInstance(struct, Atoms)
            self.assertEqual(len(struct), len(self.test_atoms))

    def test_get_cleavage_calc_names(self):
        """Test getting cleavage calculation names."""
        parent_dir = "/base/dir"
        cleavage_planes = [2.0, 3.5, 4.0]

        result = gb_cleavage_module.get_cleavage_calc_names(
            parent_dir, cleavage_planes
        ).run()

        self.assertEqual(len(result), len(cleavage_planes))

        # Check that each name contains the parent directory and plane coordinate
        for i, name in enumerate(result):
            self.assertIn(parent_dir, name)
            self.assertIn(str(np.round(cleavage_planes[i], 3)), name)

    def test_get_results_df(self):
        """Test getting results DataFrame."""
        # Create mock data
        cleavage_coords = [2.0, 3.0]
        cleaved_structures = [self.test_atoms, self.test_atoms.copy()]
        uncleaved_energy = 10.0

        # Create mock DataFrame with calc_output column
        class MockOutput:
            def __init__(
                self,
                final_energy,
                final_structure,
                final_volume,
                final_forces,
                final_stress,
            ):
                self.final_energy = final_energy
                self.final_structure = final_structure
                self.final_volume = final_volume
                self.final_forces = final_forces
                self.final_stress = final_stress
                self.convergence = True  # To pass only_converged=True in extract_outputs_from_EngineOutputs

        mock_output1 = MockOutput(
            final_energy=12.0,
            final_structure=self.test_atoms,
            final_volume=16.0,
            final_forces=np.zeros((len(self.test_atoms), 3)),
            final_stress=np.zeros((3, 3)),
        )

        mock_output2 = MockOutput(
            final_energy=13.0,
            final_structure=self.test_atoms,
            final_volume=16.0,
            final_forces=np.zeros((len(self.test_atoms), 3)),
            final_stress=np.zeros((3, 3)),
        )

        mock_df = pd.DataFrame(
            {"calc_output": [mock_output1, mock_output2], "convergence": [True, True]}
        )
        result = gb_cleavage_module.get_results_df(
            df=mock_df,
            cleavage_coords=cleavage_coords,
            cleaved_structures=cleaved_structures,
            uncleaved_energy=uncleaved_energy,
            cleavage_axis="c",
        ).run()

        self.assertIsInstance(result, pd.DataFrame)

        # Check that DataFrame has expected columns
        expected_columns = [
            "cleavage_coord",
            "initial_structure",
            "final_structure",
            "energy",
            "cleavage_energy",
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)

        # Check that cleavage energies are calculated
        self.assertEqual(len(result), len(cleavage_coords))
        self.assertTrue(all(result["cleavage_energy"] > 0))


if __name__ == "__main__":
    unittest.main()
