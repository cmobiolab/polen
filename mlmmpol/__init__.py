"""POLEN: POLarizable Embedding for Neural-network potentials (ML/MM in OpenMM with mutually polarizable AMOEBA embedding).

High-level interface (see docs/user_guide.md):
  load_plugin, amber_to_amoeba, ml_atoms_from_residues, create_mechanical_system, create_mupe_system,
  available_ligands, run_md, minimize.
Lower-level modules: amoeba_mixed, mechanical, mace_torchforce, embedding_head, export_head, ml_vdw, layered, zbl,
amber2amoeba, presets.
"""
from pathlib import Path as _Path

__version__ = (_Path(__file__).resolve().parents[1] / 'VERSION').read_text().strip() \
    if (_Path(__file__).resolve().parents[1] / 'VERSION').exists() else '0.1.0'


def __getattr__(name):
    # lazy imports: the heavy dependencies (OpenMM, PyTorch, MACE) are loaded only when used
    if name in ('load_plugin', 'plugin_dir', 'amber_to_amoeba', 'ml_atoms_from_residues', 'create_mupe_system',
                'create_mechanical_system', 'available_ligands', 'set_mts_groups', 'AmoebaSystem'):
        from . import api
        return getattr(api, name)
    if name == 'run_md':
        from .md import run_md
        return run_md
    if name == 'minimize':
        from .minimize import minimize
        return minimize
    raise AttributeError(name)
