#!/usr/bin/env python3
"""
Simple working example of surface builder functions.
"""

from ase.build import bulk, surface
from pyiron_workflow import Workflow
from pyiron_workflow_atomistics.surface.utils import (
    create_bulk_from_symbol,
    create_surface_slab,
    create_surface_from_symbol,
    get_surface_info
)

def working_example():
    """Working example showing surface builders."""
    
    print("Surface Builder Functions - Working Example")
    print("=" * 50)
    
    # Method 1: Direct ASE approach
    print("\n1. Direct ASE approach:")
    fe_bulk = bulk("Fe", crystalstructure="bcc", a=2.87)
    fe_slab = surface(fe_bulk, (1, 1, 0), layers=4, vacuum=15.0)
    print(f"   Fe(110) slab: {len(fe_slab)} atoms")
    
    # Method 2: Pyiron workflow approach
    print("\n2. Pyiron workflow approach:")
    wf = Workflow("surface_example", delete_existing_savefiles=True)
    
    # Create bulk structure
    wf.fe_bulk = create_bulk_from_symbol("Fe", crystalstructure="bcc", a=2.87)
    wf.run()
    fe_bulk_wf = wf.fe_bulk.outputs.bulk_from_symbol.value
    print(f"   Fe bulk: {len(fe_bulk_wf)} atoms")
    
    # Create surface slab
    wf.fe_slab = create_surface_slab(fe_bulk_wf, miller_indices=(1, 1, 0), layers=4, vacuum=15.0)
    wf.run()
    fe_slab_wf = wf.fe_slab.outputs.surface_slab.value
    print(f"   Fe(110) slab: {len(fe_slab_wf)} atoms")
    
    # Method 3: Direct surface creation from symbol
    print("\n3. Direct surface creation from symbol:")
    wf.cu_slab = create_surface_from_symbol(
        "Cu", 
        miller_indices=(1, 1, 1), 
        layers=3, 
        vacuum=12.0,
        crystalstructure="fcc",
        a=3.61
    )
    wf.run()
    cu_slab = wf.cu_slab.outputs.surface_from_symbol.value
    print(f"   Cu(111) slab: {len(cu_slab)} atoms")
    
    # Method 4: Surface information
    print("\n4. Surface information:")
    wf.surface_info = get_surface_info(fe_slab_wf)
    wf.run()
    info = wf.surface_info.outputs.surface_info.value
    print(f"   Surface area: {info['surface_area']:.2f} Å²")
    print(f"   Slab thickness: {info['slab_thickness']:.2f} Å")
    print(f"   Vacuum thickness: {info['vacuum_thickness']:.2f} Å")
    print(f"   Chemical formula: {info['chemical_formula']}")
    
    # Method 5: Different crystal structures
    print("\n5. Different crystal structures:")
    
    # FCC metals
    wf.al_slab = create_surface_from_symbol("Al", (1, 1, 1), layers=3, crystalstructure="fcc", a=4.05)
    wf.cu_slab2 = create_surface_from_symbol("Cu", (1, 0, 0), layers=3, crystalstructure="fcc", a=3.61)
    wf.run()
    print(f"   Al(111): {len(wf.al_slab.outputs.surface_from_symbol.value)} atoms")
    print(f"   Cu(100): {len(wf.cu_slab2.outputs.surface_from_symbol.value)} atoms")
    
    # BCC metals
    wf.fe_slab2 = create_surface_from_symbol("Fe", (1, 1, 0), layers=3, crystalstructure="bcc", a=2.87)
    wf.w_slab = create_surface_from_symbol("W", (1, 0, 0), layers=3, crystalstructure="bcc", a=3.16)
    wf.run()
    print(f"   Fe(110): {len(wf.fe_slab2.outputs.surface_from_symbol.value)} atoms")
    print(f"   W(100): {len(wf.w_slab.outputs.surface_from_symbol.value)} atoms")
    
    # Semiconductors
    wf.si_slab = create_surface_from_symbol("Si", (1, 1, 1), layers=3, crystalstructure="diamond", a=5.43)
    wf.run()
    print(f"   Si(111): {len(wf.si_slab.outputs.surface_from_symbol.value)} atoms")
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nSurface builder functions are ready to use!")

if __name__ == "__main__":
    working_example()
