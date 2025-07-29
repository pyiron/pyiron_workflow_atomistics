from dataclasses import dataclass
import ase
from ase import Atoms
from typing import Literal, Union, Optional, Any
class PrintableClass:
    def __str__(self):
        items = [f"{k}={repr(v)}" for k, v in self.to_dict().items()]
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(items) + ")"

    def to_dict(self):
        from dataclasses import fields, is_dataclass

        if is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in fields(self)}
        else:
            # fallback: collect instance attributes
            return {
                k: getattr(self, k)
                for k in dir(self)
                if not k.startswith("_") and not callable(getattr(self, k))
            }


@dataclass
class Engine(PrintableClass):
    def __init__(self, engine_id: int, parameters: dict[str, Any]):
        self.engine_id: str = engine_id
        self.parameters = parameters
        
    def __call__(self, structure: Atoms):
        raise NotImplementedError("Subclasses must implement this method")
    
@dataclass
class CalcInputStatic(PrintableClass):
    """No parameters: just force maxiter/maxeval → 0 for a pure “static” run."""
    pass


@dataclass
class CalcInputMinimize(PrintableClass):
    """Tune your minimize call tolerances and limits."""
    energy_convergence_tolerance: float = 0.0000001
    force_convergence_tolerance: float = 0.0000001
    max_iterations: int = 1_000_000
    max_evaluations: int = 1_000_000
    relax_cell: bool = False
@dataclass
class CalcInputMD(PrintableClass):
    r"""
    MD parameters with selectable ensemble modes and thermostats.

    Attributes:
        units (str): LAMMPS units style (e.g., 'metal', 'real', 'lj', 'si', 'cgs').
        mode (str): Ensemble mode: 'NVE' (constant N, V, E), 'NVT' (constant N, V, T), or 'NPT' (constant N, P, T).
        thermostat (str): Thermostat algorithm:
            - 'nose-hoover': deterministic NH thermostat (NVT/NPT).
            - 'langevin': stochastic thermostat with friction and random kicks.
            - 'berendsen': weak-coupling thermostat.
            - 'andersen': random collision thermostat.
            - 'temp/rescale': simple velocity rescaling every n_print steps.
            - 'temp/csvr': canonical sampling via velocity rescaling.
        temperature (float): Target temperature in Kelvin.
        n_ionic_steps (int): Number of MD timesteps to run.
        n_print (int): Frequency of thermo output and temp/rescale interval (steps).
        pressure (float): Target pressure (Pa).
        time_step (float): Integration timestep (ps).
        temperature_damping_timescale (float): Thermostat damping timescale (ps).
        pressure_damping_timescale (float): Barostat damping timescale (ps).
        seed (int): Random seed for stochastic thermostats.
        initial_temperature (float): Initial temperature for velocity creation (Kelvin).
        delta_temp (float): Temperature change for ramping (Kelvin).
        delta_press (float): Pressure change for ramping (Pa).
    """
    mode: Literal['NVE', 'NVT', 'NPT'] = 'NVT'
    thermostat: Literal[
        'nose-hoover',
        'langevin',
        'berendsen',
        'andersen',
        'temp/rescale',
        'temp/csvr'
    ] = 'langevin'
    temperature: Optional[Union[int, float]] = 0
    n_ionic_steps: int = 10_000
    n_print: int = 100
    pressure: Optional[Union[int, float]] = None
    time_step: Optional[Union[int, float]] = 1.0
    temperature_damping_timescale: Optional[Union[int, float]] = 100.0
    pressure_damping_timescale: Optional[Union[int, float]] = 1000.0
    seed: Optional[int] = None
    initial_temperature: Optional[float] = None
    delta_temp: Optional[float] = None
    delta_press: Optional[float] = None
        
class EngineOutput(PrintableClass):
    final_structure = None
    final_results = None
    convergence = None
    final_energy = None
    final_forces = None
    final_stress = None
    final_volume = None
    final_stress_tensor = None
    final_stress_tensor_voigt = None
    energies = None
    forces = None
    stresses = None
    stresses_voigt = None
    structures = None
    magmoms = None
    n_ionic_steps = None


@dataclass
class BuildBulkStructure_Input(PrintableClass):
    element_name: str
    crystalstructure: str = None
    a: float = None 
    b: float = None
    c: float = None
    alpha: float = None
    covera: float = None
    u: float = None
    orthorhombic: bool = False
    cubic: bool = False
    basis: list[list[float]] = None
    structure: ase.Atoms = None

    """
    Dataclass to build a bulk structure.
    Parameters
    ----------
    element_name : str
        The name of the element to build the bulk structure for.
    crystalstructure : str
        The crystal structure of the element.
    a : float
        The lattice parameter of the bulk structure.
    b : float
        The lattice parameter of the bulk structure.
    c : float
        The lattice parameter of the bulk structure.
    alpha : float
        The lattice parameter of the bulk structure.
    orthorhombic : bool
        Whether to build an orthorhombic structure.
    cubic : bool
        Whether to build a cubic structure.
    basis : list[list[float]]
        The basis of the bulk structure.
    Returns
    -------
    structure : ase.Atoms
        The bulk structure.
    """