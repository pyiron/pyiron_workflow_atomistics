#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd
from ase import Atom
from ase.build import rotate
from ase.io.lammpsdata import write_lammps_data
from ase.io.vasp import read_vasp, write_vasp
from ase.lattice.hexagonal import HexagonalClosedPacked
from pymatgen.core import Lattice, Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation

Z_THRESH = 0.00005
P_SHIFT = 0.005


def parse_inputs():
    """Parse user inputs for GB structure."""
    parser = argparse.ArgumentParser(description="Generate GB bicrystals.")
    parser.add_argument("--tilt", type=str, choices=["0001", "01-10", "12-10"])
    parser.add_argument("--potential", type=str, choices=["Zope", "Hennig", "Sa"])
    parser.add_argument("--formula", type=str, default="Ti")
    parser.add_argument("--xmax", type=float, default=3.0)
    parser.add_argument("--ymax", type=float, default=2.0)
    parser.add_argument("--zmax", type=float, default=35.0)
    args = parser.parse_args()
    potential = args.potential

    if potential == "Zope":
        a = 2.9527
        c = 4.6808
        Ecoh = -4.85
    elif potential == "Hennig":
        a = 2.9305
        c = 4.6783
        Ecoh = -4.8312
    elif potential == "Sa":
        a = 2.9436
        c = 4.6927
        Ecoh = -4.8722
    return (
        args.tilt,
        args.potential,
        args.formula,
        args.xmax,
        args.ymax,
        args.zmax,
        a,
        c,
        Ecoh,
    )


def loop_replicate(formula, a, c, dirs, xmax, ymax, zmax):
    """Replicate CSL [0001] slabs."""
    s0 = HexagonalClosedPacked(
        symbol=formula, latticeconstant=(a, c), directions=dirs, size=[1, 1, 10]
    )
    s0.positions += [0, 0, Z_THRESH]
    s0.wrap()
    sx = int(xmax // s0.cell[0, 0]) + 1
    sy = int(ymax // s0.cell[1, 1]) + 1
    s = s0 * (sx, sy, 1)
    s.positions += [0, P_SHIFT, 0]

    v = dirs[2]
    dspace = 1 / np.sqrt(
        4 / 3 * (v[0] ** 2 + v[0] * v[1] + v[1] ** 2) / a**2 + v[3] ** 2 / c**2
    )
    new_zmax = (zmax // dspace + 1) * dspace - Z_THRESH
    del s[[a.index for a in s if a.position[2] > new_zmax]]
    s.cell[2, 2] = new_zmax

    return s, s0


def gen_prim_cell(a, c, elem):
    """Create primitive HCP cell using Pymatgen."""
    basis = [[0, 0, 0], [1 / 3, 2 / 3, 1 / 2]]
    lat = Lattice.hexagonal(a=a, c=c)
    s = Structure(lattice=lat, coords=basis, species=[elem for _ in basis])
    return s


def transform_cell(struc, x, y, filename):
    """Transform HCP cell based on 3-index orthogonal axes."""
    x = [x[0], x[1], x[3]]
    y = [y[0], y[1], y[3]]

    mat = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]])
    xp = np.dot(mat, np.array(x))
    xp = (xp / np.gcd.reduce(xp)).astype(int)
    yp = np.dot(mat, np.array(y))
    yp = (yp / np.gcd.reduce(yp)).astype(int)
    zp = np.cross(xp, yp)
    zp = (zp / np.gcd.reduce(zp.astype(int))).astype(int)

    matrix = [list(xp), list(yp), list(zp)]
    transform = SupercellTransformation(scaling_matrix=matrix)
    s2 = transform.apply_transformation(struc)
    s2.to(fmt="POSCAR", filename=filename)
    return s2


def slice_cell(infile, xmax=10, ymax=5, zmax=30, disp=[0.0, 0.0, 0.0]):
    """Create non-periodic cell of arbitrary height."""
    s = read_vasp(infile)
    s *= (1, 1, 10)

    # Rotate into place
    for i in range(20):
        ax, bx, cx = s.cell
        rotate(s, ax, (1, 0, 0), bx, (0, 1, 0), rotate_cell=True)

    # Square up the cell
    aa, bb, cc = s.cell
    assert abs(aa[1]) + abs(aa[2]) + abs(bb[0]) + abs(bb[2]) < 1e-5, (
        "Cell not very orthogonal!"
    )
    s.set_cell([aa[0], bb[1], cc[2]])
    s.positions += disp
    s.wrap(pbc=[True, True, False])
    assert min(s.positions[:, 2] > -2 * Z_THRESH), "Atoms with negative z value!"

    # Add back missing atom
    mask = s.positions[:, 1] < P_SHIFT + Z_THRESH
    zmid = np.sort(s.positions[mask, 2])[2]
    npl = sum(
        (s.positions[:, 2] < zmid + Z_THRESH) & (s.positions[:, 2] > zmid - Z_THRESH)
    )

    zmin = min(s.positions[:, 2])
    nmin = sum(s.positions[:, 2] < zmin + Z_THRESH)
    if nmin != npl:
        print(f"Missing atoms! Found {nmin} instead of {npl}")
        mask1 = s.positions[:, 1] < P_SHIFT + Z_THRESH
        zmid = np.sort(s.positions[mask1, 2])[1]
        mask2 = (s.positions[:, 2] < zmid + Z_THRESH) & (
            s.positions[:, 2] > zmid - Z_THRESH
        )
        ymid = max(s.positions[mask2, 1])
        s.append(Atom(formula, position=(s.cell[0, 0] / 2, ymid, zmin)))

    # Save large slab as reference (e.g., for DFT) and remove extra atoms
    s_ref = s.copy()
    del s_ref[[a.index for a in s_ref if a.position[2] > s_ref.cell[2, 2] - Z_THRESH]]

    # Remove atoms that lie outside the box of interest
    del s[[a.index for a in s if a.position[2] > zmax]]
    s.set_cell([aa[0], bb[1], zmax])

    # scale in x and y directions as needed
    sx = int(xmax // aa[0] + 1)
    sy = int(ymax // bb[1] + 1)

    return s * (sx, sy, 1), s_ref * (sx, sy, 1)


def write_cell(struct, folder, outfile, label):
    """Write the structure to a POSCAR file."""
    filepath = os.path.join(folder, outfile)
    struct.wrap(pbc=[True, True, False])
    write_vasp(filepath, struct, label=label, direct=True, sort=True)


def write_gb(filename, top_crystal, bottom_crystal, gap, vac, gb_normal=2):
    """Create a GB bicrystal structure in POSCAR and LAMMPSDATA formats."""
    offset = bottom_crystal.cell[gb_normal, gb_normal] + gap
    tc = top_crystal.copy()
    tc.positions[:, gb_normal] += offset
    tc.extend(bottom_crystal)
    tc.cell[gb_normal, gb_normal] += offset + vac
    tc.wrap(pbc=[True, True, False])
    write_vasp(
        f"{filename}.POSCAR",
        tc,
        label=f"bicrystal_gap{gap}_normal{gb_normal}",
        direct=True,
        sort=True,
    )
    write_lammps_data(f"{filename}.lammpsdata", tc)


if __name__ == "__main__":
    # Parse input arguments
    tilt, potential, formula, xmax, ymax, zmax0, a, c, Ecoh = parse_inputs()
    print(
        f"Generating {formula} {tilt} STGB structures for the {potential} potential..."
    )

    # Gather HCP plane data
    df = pd.read_csv(
        os.path.join("data", f"GB_{tilt.strip('-')}_sym_tilt.txt"), sep=" "
    )
    df = df[~df["angle"].isin([0, 60, 120, 180])].reset_index(drop=True)

    if tilt == "0001":
        for i, row in df.iterrows():
            folder = f"{row['angle']:.2f}_{''.join([str(x) for x in row['uyu':'uyw'].tolist()])}"
            root_folder = os.path.join("..", "..", tilt, potential, folder)
            os.makedirs(root_folder, exist_ok=True)
            print(row["angle"], row["uyu":"uyw"].tolist())

            upper_dirs = [
                row["uxu":"uxw"].tolist(),
                (-row["uzu":"uzw"]).tolist(),
                row["uyu":"uyw"].tolist(),
            ]
            upper, _ = loop_replicate(formula, a, c, upper_dirs, xmax, ymax, zmax0)
            upper.positions[:, 2] += P_SHIFT
            write_vasp(
                os.path.join(root_folder, "POSCAR_SUBSTRATE_2"),
                upper,
                label=f"{formula} upper {upper_dirs}",
                direct=True,
                sort=True,
            )

            lower_dirs = [
                row["lxu":"lxw"].tolist(),
                (-row["lzu":"lzw"]).tolist(),
                row["lyu":"lyw"].tolist(),
            ]
            lower, ref = loop_replicate(formula, a, c, lower_dirs, xmax, ymax, zmax0)
            write_vasp(
                os.path.join(root_folder, "POSCAR_SUBSTRATE_1"),
                lower,
                label=f"{formula} lower {lower_dirs}",
                direct=True,
                sort=True,
            )

            write_vasp(
                os.path.join(root_folder, "POSCAR_SUBSTRATE_REF"),
                ref,
                label=f"{formula} lower {lower_dirs}",
                direct=True,
                sort=True,
            )
            write_gb(
                os.path.join(root_folder, "bicrystal"), upper, lower, gap=0.0, vac=10
            )

            # break
    else:
        prim_cell = gen_prim_cell(a, c, formula)

        tilt_folder = tilt.strip("-")
        root_folder = os.path.join("..", "..", tilt_folder, potential)
        os.makedirs(root_folder, exist_ok=True)

        for i in range(len(df)):
            # Gather information
            angle = df.loc[i, "angle"]
            tilt = df.loc[i, "tu":"tw"].astype(int).tolist()
            plane = df.loc[i, "ph":"pl"].astype(int).tolist()
            vec = df.loc[i, "vu":"vw"].astype(int).tolist()
            print(angle, tilt, plane, vec)

            pp = f"{angle:.2f}_" + "".join(str(x) for x in plane)
            plane_folder = os.path.join(root_folder, pp)
            os.makedirs(plane_folder, exist_ok=True)

            h, k, i, l = plane
            dhkl = np.sqrt(1 / (4 / 3 * (h**2 + h * k + k**2) / a**2 + l**2 / c**2))
            zmax = (zmax0 // dhkl + 1) * dhkl - Z_THRESH

            temp_file = "data/Ti_Enze_transformed.POSCAR"

            ##########    Rotate to get proper orientation - UPPER
            transformed = transform_cell(prim_cell, vec.copy(), tilt.copy(), temp_file)

            # Slice the appropriate chunk
            udisp = [0.0, P_SHIFT, P_SHIFT]
            upper_cell, cell_ref = slice_cell(temp_file, xmax, ymax, zmax, disp=udisp)

            # Write slab to output
            final_file = "POSCAR_SUBSTRATE_2"
            label = f"UPPER - {tilt} tilt, {plane} plane"
            write_cell(upper_cell, plane_folder, final_file, label)
            write_cell(cell_ref, plane_folder, "POSCAR_SUBSTRATE_REF", label)

            ##########    Repeat with LOWER     #############
            vec[-1] *= -1
            transformed = transform_cell(prim_cell, vec.copy(), tilt.copy(), temp_file)

            # Slice the appropriate chunk
            ldisp = [0.0, P_SHIFT, Z_THRESH]
            lower_cell, cell_ref = slice_cell(temp_file, xmax, ymax, zmax, disp=ldisp)

            # Write slab to output
            final_file = "POSCAR_SUBSTRATE_1"
            label = f"LOWER - {tilt} tilt, {plane} plane"
            write_cell(lower_cell, plane_folder, final_file, label)

            # Write grain boundary structure
            outfile = os.path.join(plane_folder, f"{pp}_bicrystal")
            write_gb(outfile, upper_cell, lower_cell, gap=0.1, vac=0)

            # break
