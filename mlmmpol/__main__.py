"""Command-line interface of POLEN.

  python -m mlmmpol amber2amoeba PRMTOP COORDS OUT_PREFIX --solute RES final.xyz final.key [--solute ...]
      Convert an Amber system into Tinker/AMOEBA files (OUT_PREFIX.xyz, OUT_PREFIX.key).

  python -m mlmmpol run --embedding {mechanical,mupe-ff,mupe-ml} --prmtop P --coords C --ml-residues RES [...]
      Build the ML/MM system and run restartable MD (see `python -m mlmmpol run -h`).
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _solutes(items):
    return [tuple(s) for s in (items or [])]


def cmd_amber2amoeba(a):
    from .amber2amoeba import write_system
    xyz, key, box, mapping = write_system(a.prmtop, a.coords, _solutes(a.solute), a.out_prefix)
    print(f'wrote {xyz} and {key}')


def cmd_run(a):
    import openmm.app as app
    from openmm import unit
    from . import api
    from .md import run_md
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.embedding == 'mechanical':
        top = app.AmberPrmtopFile(a.prmtop).topology
        ml = api.ml_atoms_from_residues(top, a.ml_residues)
        topology, system = api.create_mechanical_system(a.prmtop, ml, model=a.model, charge=a.charge)
        api.set_mts_groups(system)
        crd = app.AmberInpcrdFile(a.coords)
        positions, box = crd.getPositions(), crd.getBoxVectors()
    else:
        amo = api.amber_to_amoeba(a.prmtop, a.coords, _solutes(a.solute), out / 'amoeba')
        topology, positions, box = amo.topology, amo.positions, amo.box
        ml = amo.atoms_of_residues(a.ml_residues)
        system, _ = api.create_mupe_system(topology, amo.system, ml, multipoles='amoeba' if a.embedding == 'mupe-ff' else 'ml',
                                           model=a.model, charge=a.charge, preset=a.preset, ligand=a.ligand,
                                           ml_vdw=not a.no_ml_vdw, layered=a.layered, workdir=out)
    np.savetxt(out / 'ml_atoms.txt', ml, fmt='%d')
    print(f'{a.embedding}: {len(ml)} ML atoms, {system.getNumParticles()} particles', flush=True)
    props = {'Precision': 'mixed', 'DeviceIndex': str(a.device)} if a.platform == 'CUDA' else {}
    run_md(system, topology, positions, out, ns=a.ns, box=box, equil_ps=a.equil_ps, temperature=a.temperature,
           pressure=None if a.nvt else a.pressure, platform=a.platform, properties=props, seed=a.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m mlmmpol', description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('amber2amoeba', help='convert an Amber system into Tinker/AMOEBA files')
    p.add_argument('prmtop'); p.add_argument('coords'); p.add_argument('out_prefix')
    p.add_argument('--solute', nargs=3, action='append', metavar=('RESNAME', 'XYZ', 'KEY'),
                   help='Poltype parameters of a non-standard residue (repeat for each)')
    p.set_defaults(func=cmd_amber2amoeba)
    r = sub.add_parser('run', help='build the ML/MM system and run restartable MD')
    r.add_argument('--embedding', choices=['mechanical', 'mupe-ff', 'mupe-ml'], required=True)
    r.add_argument('--prmtop', required=True); r.add_argument('--coords', required=True, help='inpcrd/rst7')
    r.add_argument('--ml-residues', nargs='+', required=True, help='residue names of the ML region')
    r.add_argument('--solute', nargs=3, action='append', metavar=('RESNAME', 'XYZ', 'KEY'),
                   help='Poltype parameters (MuPE only; repeat for each non-standard residue)')
    r.add_argument('--model', default='mace-off23-small'); r.add_argument('--charge', type=int, default=0)
    r.add_argument('--preset', default=None, help='MuPE(ML) preset (default: the validated default)')
    r.add_argument('--ligand', default=None, help='ligand key of the ML–MM vdW refit (MuPE(ML))')
    r.add_argument('--no-ml-vdw', action='store_true', help='MuPE(ML) without the ML–MM vdW refit')
    r.add_argument('--layered', nargs=2, type=float, metavar=('R_IN', 'R_OUT'), default=None)
    r.add_argument('--ns', type=float, default=1.0); r.add_argument('--equil-ps', type=float, default=200.0)
    r.add_argument('--temperature', type=float, default=300.0); r.add_argument('--pressure', type=float, default=1.0)
    r.add_argument('--nvt', action='store_true'); r.add_argument('--seed', type=int, default=None)
    r.add_argument('--platform', default='CUDA'); r.add_argument('--device', type=int, default=0)
    r.add_argument('--out', required=True)
    r.set_defaults(func=cmd_run)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    sys.exit(main())
