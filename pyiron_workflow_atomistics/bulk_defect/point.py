import pyiron_workflow as pwf
from pyiron_workflow_atomistics.structure_manipulator.tools import create_supercell_with_min_dimensions
from pyiron_workflow_atomistics.calculator import calculate_structure_node
import os

@pwf.as_function_node("vacancy_structure")
def create_vacancy_structure(structure, remove_atom_index=0):
    vac_structure = structure.copy()
    vac_structure.pop(remove_atom_index)
    return vac_structure

@pwf.as_function_node("vacancy_formation_energy")
def calculate_vacancy_formation_energy(vacancy_calc, supercell_calc):
    energy = vacancy_calc - supercell_calc
    return energy

@pwf.as_function_node("duplicate_engine")
def duplicate_engine(Engine,
                     working_directory):
    duplicate_engine = Engine.copy()
    duplicate_engine.working_directory = os.path.join(Engine.working_directory, working_directory)
    return duplicate_engine

@pwf.as_macro_node("calc_supercell", "calc_vacancy", "vacancy_formation_energy")#, "vacancy_structure", "structure_supercell")
def get_vacancy_formation_energy(wf,
                                 structure,
                                 Engine,
                                 remove_atom_index=0,
                                 min_dimensions=[12, 12, 12],
                                 vacancy_engine_working_directory="vacancy_supercell",
                                 supercell_engine_working_directory="supercell"):
    wf.structure_supercell = create_supercell_with_min_dimensions(structure, min_dimensions=min_dimensions)
    wf.structure_with_vacancy = create_vacancy_structure(wf.structure_supercell, remove_atom_index=remove_atom_index)
    wf.vacancy_engine = duplicate_engine(Engine, vacancy_engine_working_directory)
    wf.vacancy_calc = calculate_structure_node(wf.structure_with_vacancy, calculation_engine=wf.vacancy_engine)
    wf.supercell_engine = duplicate_engine(Engine, supercell_engine_working_directory)
    wf.supercell_calc = calculate_structure_node(wf.structure_supercell, calculation_engine=wf.supercell_engine)
    wf.vacancy_formation_energy = calculate_vacancy_formation_energy(wf.vacancy_calc.outputs.calc_output.final_energy,
                                                                    wf.supercell_calc.outputs.calc_output.final_energy)
    return wf.supercell_calc, wf.vacancy_calc, wf.vacancy_formation_energy