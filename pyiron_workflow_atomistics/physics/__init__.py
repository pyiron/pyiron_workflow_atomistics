"""Physics workflows organised by topic.

Import per-topic, not from this package directly::

    from pyiron_workflow_atomistics.physics.bulk           import eos_volume_scan
    from pyiron_workflow_atomistics.physics.surface        import calculate_surface_energy
    from pyiron_workflow_atomistics.physics.point_defect   import get_vacancy_formation_energy
    from pyiron_workflow_atomistics.physics.grain_boundary import pure_gb_study

This package intentionally re-exports nothing so the import path tells you
which topic each macro belongs to.
"""
