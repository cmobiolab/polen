"""MACE as a TorchForce (openmm-torch): the whole model runs on the GPU, inside the OpenMM Context.

Reason: in openmm-ml 1.7 MACE enters as a PythonForce, and on the CUDA platform, with an AmoebaMultipoleForce in
the same system, the positions passed to the PythonForce during integrator steps are corrupted (errors of hundreds
of nm; OpenMM 8.5.2) and the dynamics blows up. In addition, the PythonForce copies positions and forces between
CPU and GPU at every step.

The module receives all positions (nm), selects the ML atoms, builds the neighbor list (all pairs within the MACE
cutoff) and returns the energy in kJ/mol; forces come from autograd in openmm-torch.
For periodic systems, ML positions are unwrapped relative to the first ML atom (the ML region must be a molecule
smaller than half the box).
"""

import os
from typing import Dict, List, Optional

import openmm
import torch

EV_TO_KJ = 96.4853
NM_TO_ANG = 10.0

MACE_MODELS = {
    'mace-off23-small': 'small',
    'mace-off23-medium': 'medium',
    'mace-off23-large': 'large',
    'mace-omol-0-extra-large': 'extra_large',   # charge and multiplicity as input (OMol25)
}


class MACETorchForceModule(torch.nn.Module):
    def __init__(self, model, atomic_numbers: List[int], indices: List[int], dtype: torch.dtype,
                 periodic: bool = False, energy_key: str = 'interaction_energy', charge: float = 0.0,
                 multiplicity: float = 1.0):
        super().__init__()
        from mace.tools import atomic_numbers_to_indices, to_one_hot, utils
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        node_attrs = to_one_hot(torch.tensor(atomic_numbers_to_indices(atomic_numbers, z_table=z_table),
                                             dtype=torch.long).unsqueeze(-1), num_classes=len(z_table))
        n = len(indices)
        self.model = model
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.register_buffer('node_attrs', node_attrs.to(dtype))
        self.register_buffer('ptr', torch.tensor([0, n], dtype=torch.long))
        self.register_buffer('batch', torch.zeros(n, dtype=torch.long))
        self.register_buffer('pbc', torch.tensor([False, False, False]))
        self.register_buffer('cell', torch.eye(3, dtype=dtype))
        self.register_buffer('eye', torch.eye(n, dtype=torch.bool))
        # total charge and multiplicity (used by MACE-OMOL; ignored by MACE-OFF)
        self.register_buffer('total_charge', torch.tensor([charge], dtype=dtype))
        self.register_buffer('total_spin', torch.tensor([multiplicity], dtype=dtype))
        # fixed list with all pairs (i != j): constant shapes, required for CUDA graphs; the MACE cutoff envelope
        # zeroes exactly the edges beyond r_max, so the result is the same as with a radius-based list
        ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        keep = ii != jj
        self.register_buffer('all_pairs', torch.stack([ii[keep], jj[keep]]))
        # fixed list only up to 48 atoms: with 53–56 ML atoms (Mg·ATP + waters) CUDA dynamics hangs or segfaults
        # inside the TorchForce (isolated evaluations work; 50 atoms ok, 53 not; the radius-based list works with 56).
        # Exact cause not identified; adjustable through an environment variable
        self.fixed_edges = n <= int(os.environ.get('MLMMPOL_MACE_FIXED_EDGES_MAX', '48'))
        self.cutoff = float(model.r_max.detach())
        self.periodic = periodic
        self.dtype = dtype
        self.energy_key = energy_key
        self.nm_to_ang = NM_TO_ANG
        self.ev_to_kj = EV_TO_KJ

    def forward(self, positions: torch.Tensor, boxvectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = positions[self.indices].to(self.dtype)
        if self.periodic and boxvectors is not None:
            box = boxvectors.to(self.dtype)
            # minimum image in a reduced triclinic box (OpenMM convention), first relative to atom 0 and then to the
            # centroid: in a large ML region (Mg²⁺·ATP + waters, up to ~1.7 nm) atom 0 can sit at one end, close to
            # the half-box-width limit; from the centroid, distances are about half
            for k in range(2):
                ref = x[0] if k == 0 else x.mean(dim=0)
                d = x - ref
                d = d - torch.outer(torch.round(d[:, 2] / box[2, 2]), box[2])
                d = d - torch.outer(torch.round(d[:, 1] / box[1, 1]), box[1])
                d = d - torch.outer(torch.round(d[:, 0] / box[0, 0]), box[0])
                x = ref + d
        x = x * self.nm_to_ang
        if self.fixed_edges:
            edge_index = self.all_pairs
        else:
            dist = torch.cdist(x, x)
            edge_index = torch.nonzero((dist < self.cutoff) & ~self.eye).t().contiguous()
        data: Dict[str, torch.Tensor] = {
            'ptr': self.ptr, 'node_attrs': self.node_attrs, 'batch': self.batch, 'pbc': self.pbc,
            'positions': x, 'edge_index': edge_index,
            'shifts': torch.zeros(edge_index.shape[1], 3, dtype=self.dtype, device=x.device),
            'cell': self.cell, 'total_charge': self.total_charge, 'total_spin': self.total_spin,
        }
        out = self.model(data, compute_force=False)
        energy = out[self.energy_key]
        assert energy is not None
        return energy.sum() * self.ev_to_kj


class ConstantJointEmbedding(torch.nn.Module):
    """MACE-OMOL charge/spin embedding precomputed for fixed charge and multiplicity (exact in that case).

    The mace-torch 0.3.16 GenericJointEmbedding stores its specifications in a mixed-type dict, which TorchScript
    does not accept; since the charge and multiplicity of the ML region do not change during dynamics, the vector is
    the same for all atoms and all steps.
    """

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer('value', value)

    def forward(self, batch: torch.Tensor, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value.expand(batch.shape[0], -1)


def to_cueq(model):
    """Convert the model to the cuEquivariance (NVIDIA) kernels; the resulting model lives on the GPU."""
    from mace.cli.convert_e3nn_cueq import run
    return run(model, device='cpu', return_model=True)            # no conv_fusion (does not compile in TorchScript)


def load_mace(name: str, dtype: torch.dtype, charge: float = 0.0, multiplicity: float = 1.0, cueq: bool = False):
    from mace.calculators.foundations_models import mace_off, mace_omol
    if name.startswith('mace-omol'):
        model = mace_omol(model=MACE_MODELS[name], device='cpu', return_raw_model=True).to(dtype)
        if cueq:
            model = to_cueq(model).to(dtype)
        if hasattr(model, 'joint_embedding'):
            feats = {'total_charge': torch.tensor([charge], dtype=dtype), 'total_spin': torch.tensor([multiplicity], dtype=dtype)}
            with torch.no_grad():
                dev = next(model.parameters()).device
            value = model.joint_embedding(torch.zeros(1, dtype=torch.long, device=dev),
                                              {k: feats[k].to(dev) for k in model.embedding_specs})[0].clone()
            model.joint_embedding = ConstantJointEmbedding(value)
            model.embedding_specs = {k: '' for k in model.embedding_specs}      # names only (TorchScript)
        return model
    model = mace_off(model=MACE_MODELS[name], device='cpu', return_raw_model=True).to(dtype)
    return to_cueq(model).to(dtype) if cueq else model


def make_mace_torchforce(name: str, topology, ml_atoms, precision: str = 'double', periodic: bool = False,
                         force_group: int = 0, charge: float = 0.0, multiplicity: float = 1.0, cueq: bool = False):
    """Create an openmmtorch.TorchForce with MACE on the ML region (TorchScript, runs on the Context GPU)."""
    import openmmtorch
    from e3nn.util import jit
    dtype = torch.float64 if precision == 'double' else torch.float32
    atoms = list(topology.atoms())
    module = MACETorchForceModule(jit.compile(load_mace(name, dtype, charge, multiplicity, cueq)), [atoms[i].element.atomic_number for i in ml_atoms],
                                  [int(i) for i in ml_atoms], dtype, periodic=periodic,
                                  energy_key='energy' if name.startswith('mace-omol') else 'interaction_energy',
                                  charge=charge, multiplicity=multiplicity)
    force = openmmtorch.TorchForce(torch.jit.script(module))
    force.setOutputsForces(False)
    if module.fixed_edges:
        force.setProperty('useCUDAGraphs', 'true')      # small ML regions: launch latency dominates
    force.setUsesPeriodicBoundaryConditions(periodic)
    force.setForceGroup(force_group)
    return force


def replace_python_force(system, torch_force):
    """Replace the PythonForce(s) of the system with the given TorchForce."""
    removed = 0
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), openmm.PythonForce):
            system.removeForce(i)
            removed += 1
    if removed != 1:
        raise RuntimeError(f'expected 1 PythonForce, found {removed}')
    system.addForce(torch_force)
    return system
