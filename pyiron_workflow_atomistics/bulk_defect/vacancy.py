import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import Engine, run
from pyiron_workflow_atomistics.structure_manipulator.tools import create_supercell_with_min_dimensions


@pwf.as_function_node("vacancy_structure")
def create_vacancy_structure(structure, remove_atom_index=0):
    vac_structure = structure.copy()
    vac_structure.pop(remove_atom_index)
    return vac_structure


@pwf.as_function_node("vacancy_formation_energy")
def calculate_vacancy_formation_energy(vacancy_calc, supercell_calc):
    return vacancy_calc - supercell_calc


@pwf.as_macro_node("calc_supercell", "calc_vacancy", "vacancy_formation_energy")
def get_vacancy_formation_energy(wf,
                                 structure,
                                 engine: Engine,
                                 remove_atom_index=0,
                                 min_dimensions=[12, 12, 12],
                                 vacancy_subdir="vacancy_supercell",
                                 supercell_subdir="supercell"):
    wf.structure_supercell    = create_supercell_with_min_dimensions(structure, min_dimensions=min_dimensions)
    wf.structure_with_vacancy = create_vacancy_structure(wf.structure_supercell, remove_atom_index=remove_atom_index)
    wf.vacancy_calc   = run(wf.structure_with_vacancy, engine=engine.with_working_directory(vacancy_subdir))
    wf.supercell_calc = run(wf.structure_supercell,    engine=engine.with_working_directory(supercell_subdir))
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(
        wf.vacancy_calc.outputs.engine_output.final_energy,
        wf.supercell_calc.outputs.engine_output.final_energy,
    )
    return wf.supercell_calc, wf.vacancy_calc, wf.vacancy_formation_energy
