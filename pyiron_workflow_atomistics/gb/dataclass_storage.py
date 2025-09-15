from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CleaveGBStructureInput:
    axis_to_cleave: str = "c"
    tol: float = 0.3
    cleave_region_halflength: float = 5.0
    layer_tolerance: float = 0.3
    separation: float = 8.0
    use_fractional: bool = False
    cleavage_target_coord: float = 0.5


@dataclass
class FindGBPlaneInput:
    featuriser: Any
    gb_axis: str = "c"
    approx_frac: float = 0.5
    tolerance: float = 5.0
    bulk_offset: float = 10.0
    slab_thickness: float = 2.0
    featuriser_kwargs: Dict = field(default_factory=dict)
    n_bulk: int = 10
    threshold_frac: float = 0.1


@dataclass
class PlotGBPlaneInput:
    projection: Tuple[int, int] = (0, 2)
    reps: Tuple[int, int] = (5, 1)
    figsize: Tuple[float, float] = (10, 6)
    bulk_color: str = "C0"
    window_cmap: str = "viridis"
    plane_linestyles: Tuple[str, str] = ("--", "-")
    axis: int = 2
    dpi: int = 300
    save_path: Optional[str] = None


@dataclass
class PlotCleaveInput:
    projection: Tuple[int, int] = (0, 2)
    reps: Tuple[int, int] = (5, 1)
    figsize: Tuple[float, float] = (8, 6)
    atom_color: str = "C0"
    plane_color: str = "r"
    plane_linestyle: str = "--"
    atom_size: float = 30
    save_path: Optional[str] = None
    dpi: int = 300
    show_fractional_axes: bool = True
    ylims: List[float] = field(default_factory=lambda: [0, 61])


from dataclasses import asdict, dataclass


@dataclass
class CalcStructureInput:
    output_dir: str = "gb_cleavage/calculations"
    fmax: float = 0.01
    max_steps: int = 1000
    properties: Tuple[str, ...] = ("energy", "forces", "stresses", "volume")
    write_to_disk: bool = False
    initial_struct_path: str = "initial_structure.xyz"
    initial_results_path: str = "initial_results.json"
    traj_struct_path: str = "trajectory.xyz"
    traj_results_path: str = "trajectory_results.json"
    final_struct_path: str = "final_structure.xyz"
    final_results_path: str = "final_results.json"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this dataclass into a plain dict suitable
        for passing as **kwargs to calc_structure or similar.
        """
        return asdict(self)
