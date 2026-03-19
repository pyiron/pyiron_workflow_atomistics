import unittest

from gb_code.gb_generator import GB_character
from gb_code.inplane_shift import generate_shifts, get_all_shifted_structures


class TestGBCode(unittest.TestCase):
    def _get_sigma5_100_fcc(self):
        gb = GB_character()
        gb.ParseGB([1, 0, 0], "fcc", 4.05, 2, 1, [0, 3, 1])
        gb.CSL_Bicrystal_Atom_generator()
        return gb

    def test_to_pymatgen_sets_grain_ids(self):
        structure = self._get_sigma5_100_fcc().build(
            dim=[1, 1, 1], gb_normal="z"
        ).to_pymatgen(element="Al")

        self.assertGreater(len(structure), 0)
        self.assertEqual(set(structure.site_properties["grain_id"]), {1, 2})
        self.assertEqual({str(site.specie) for site in structure}, {"Al"})
        self.assertGreater(structure.lattice.c, 0)

    def test_to_ase_sets_arrays_and_info(self):
        atoms = self._get_sigma5_100_fcc().build(
            dim=[1, 1, 1], gb_normal="y"
        ).to_ase(element="Al")

        self.assertGreater(len(atoms), 0)
        self.assertEqual(set(atoms.arrays["grain_id"]), {1, 2})
        self.assertEqual(set(atoms.get_tags()), {1, 2})
        self.assertIn("gb_plane_y", atoms.info)

    def test_inplane_shift_returns_requested_number_of_structures(self):
        gb = self._get_sigma5_100_fcc()
        gb.build(dim=[1, 1, 1])

        shifts = generate_shifts(gb, a=2, b=2)
        structures = get_all_shifted_structures(gb, a=2, b=2, output="ase", element="Al")

        self.assertEqual(len(shifts), 4)
        self.assertEqual(len(structures), 4)
        self.assertTrue(all(len(atoms) == len(structures[0]) for atoms in structures))

    def test_get_gbstruct_from_gbcode_respects_axis_and_element(self):
        try:
            from pyiron_workflow_atomistics.gb.gb_code.constructor import (
                get_gbstruct_from_gbcode,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"constructor dependencies unavailable: {exc}")

        req_length_grain = 15
        structure = get_gbstruct_from_gbcode(
            axis=[1, 0, 0],
            basis="fcc",
            lattice_param=4.05,
            m=2,
            n=1,
            GB1=[0, 3, 1],
            element="Al",
            req_length_grain=req_length_grain,
            grain_length_axis=2,
        ).run()

        self.assertGreater(len(structure), 0)
        self.assertEqual({str(site.specie) for site in structure}, {"Al"})
        self.assertEqual(set(structure.site_properties["grain_id"]), {1, 2})
        self.assertGreaterEqual(structure.lattice.abc[2], 2 * req_length_grain)

    def test_get_gbstruct_from_gbcode_invalid_axis_raises(self):
        try:
            from pyiron_workflow_atomistics.gb.gb_code.constructor import (
                get_gbstruct_from_gbcode,
            )
        except ModuleNotFoundError as exc:
            self.skipTest(f"constructor dependencies unavailable: {exc}")

        with self.assertRaisesRegex(
            ValueError, "grain_length_axis must be one of 0, 1, or 2"
        ):
            get_gbstruct_from_gbcode(
                axis=[1, 0, 0],
                basis="fcc",
                lattice_param=4.05,
                m=2,
                n=1,
                GB1=[0, 3, 1],
                element="Al",
                grain_length_axis=4,
            ).run()


if __name__ == "__main__":
    unittest.main()
