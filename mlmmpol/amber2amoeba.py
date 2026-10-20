"""Convert an Amber dynamics frame (protein + solute + OPC water + ions) into an OpenMM AMOEBA system.

Route: write the whole system as a Tinker .xyz and load it with openmm.app.TinkerFiles (amoebabio18.prm + the
solutes' final.key). The OPC virtual sites (EPW) are dropped; the box (Amber truncated octahedron) becomes the
OpenMM triclinic box vectors.
- Protein: typed by Tinker's pdbxyz (amoebabio18) from a PDB with the original Amber residue names
  (HID/HIE/HIP/CYX...) as ATOM records (Tinker ignores HETATM and treats HIS as protonated His). The atom order
  is the Tinker one, and the protein comes first in the system.
- Solutes (ligands): types from the Poltype2 final.xyz/final.key (same atom order as the mol2/sdf).
- Water, Na+, Cl-, Mg2+ and Ca2+: amoebabio18 types.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

# workstation paths; on remote machines, taken from environment variables
BIO18 = os.environ.get('MLMMPOL_BIO18', '/home/rdcunha/Documentos/SOFT/poltype2/ParameterFiles/amoebabio18.prm')
PDBXYZ = os.environ.get('MLMMPOL_PDBXYZ', '/home/rdcunha/Documentos/SOFT/tinker/bin/pdbxyz')
WATER_O, WATER_H, NA, CL, MG, CA = 349, 350, 352, 361, 357, 358
NONPROTEIN = ('HOH', 'WAT', 'Na+', 'Cl-', 'NA', 'CL', 'MG', 'CA')
PROTONATION = ('HID', 'HIE', 'HIP', 'CYX', 'ASH', 'GLH', 'LYN', 'CYM')


def read_tinker_xyz(path):
    """Types and connectivity (0-based indices) of a Tinker .xyz (e.g. the Poltype final.xyz)."""
    lines = open(path).read().split('\n')
    n = int(lines[0].split()[0])
    types, bonds = [], []
    for k, l in enumerate(lines[1:n + 1]):
        f = l.split()
        types.append(int(f[5]))
        bonds += [(k, int(b) - 1) for b in f[6:] if int(b) - 1 > k]
    return types, bonds


def protein_to_tinker(prm, positions, exclude, workdir):
    """Type the protein with pdbxyz; return (atoms [(el, type, pos Å)], bonds), in Tinker order."""
    workdir = Path(workdir)
    labels = prm._prmtop._raw_data['RESIDUE_LABEL']
    mod = app.Modeller(prm.topology, positions)
    mod.delete([a for a in mod.topology.atoms() if a.residue.name in exclude])
    if mod.topology.getNumAtoms() == 0:
        return [], []
    for r in mod.topology.residues():
        amber = labels[r.index].strip()
        if amber in PROTONATION:
            r.name = amber
    pdb = workdir / 'prot_tinker.pdb'
    app.PDBFile.writeFile(mod.topology, mod.positions, open(pdb, 'w'))
    txt = open(pdb).read().split('\n')
    txt = ['ATOM  ' + l[6:] if l.startswith('HETATM') else l for l in txt if not l.startswith('CONECT')]
    open(pdb, 'w').write('\n'.join(txt))
    for f in ('prot_tinker.xyz', 'prot_tinker.seq'):
        (workdir / f).unlink(missing_ok=True)
    # parameters via the .key and PDB/chains on the command line: with pdbxyz built by the conda gfortran (Carbono),
    # stdin input disconnects after the first read (unit 5 becomes 'fort.5'); output identical to the stdin route
    (workdir / 'prot_tinker.key').write_text(f'parameters {BIO18}\n')
    run = subprocess.run([PDBXYZ, 'prot_tinker.pdb', 'ALL'], cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True,
                         text=True)
    if not (workdir / 'prot_tinker.xyz').exists():
        raise RuntimeError(f'pdbxyz did not produce prot_tinker.xyz (exit code {run.returncode}):\n{(run.stdout + run.stderr)[-2000:]}')
    lines = open(workdir / 'prot_tinker.xyz').read().split('\n')
    n = int(lines[0].split()[0])
    if '.' in lines[1].split()[0]:                  # box line (the PDB had CRYST1)
        lines = lines[:1] + lines[2:]
    if n != mod.topology.getNumAtoms():
        raise RuntimeError(f'pdbxyz: {n} atoms, expected {mod.topology.getNumAtoms()} (see {workdir})')
    atoms, bonds = [], []
    for k, l in enumerate(lines[1:n + 1]):
        f = l.split()
        el = f[1][:2] if f[1][:2] in ('Na', 'Cl', 'Ca', 'Mg', 'Zn') else f[1][0]
        atoms.append((el, int(f[5]), np.array([float(f[2]), float(f[3]), float(f[4])])))
        bonds += [(k, int(b) - 1) for b in f[6:] if int(b) - 1 > k]
    return atoms, bonds


def write_system(prmtop, rst, solutes, out_prefix, ion_types=None):
    """solutes: list of (amber_residue_name, Poltype final.xyz, final.key); returns (xyz, key, box, mapping).

    mapping[i_amber] = index in the AMOEBA system of the non-protein atoms (None for the protein, whose order is the
    Tinker one, and for the dropped virtual sites).
    """
    prm = app.AmberPrmtopFile(str(prmtop))
    crd = app.AmberInpcrdFile(str(rst))
    pos = crd.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    top = prm.topology
    ion_types = ion_types or {'Na+': NA, 'Cl-': CL, 'MG': MG, 'CA': CA}
    sol = {res: read_tinker_xyz(xyz) for res, xyz, _ in solutes}
    workdir = Path(out_prefix).resolve().parent
    atoms, bonds = protein_to_tinker(prm, crd.getPositions(), NONPROTEIN + tuple(sol), workdir)
    mapping = [None] * top.getNumAtoms()
    for res in top.residues():
        ra = list(res.atoms())
        start = len(atoms)
        if res.name in sol:
            types, rbonds = sol[res.name]
            assert len(types) == len(ra), (res.name, len(types), len(ra))
            for a, t in zip(ra, types):
                mapping[a.index] = len(atoms)
                atoms.append((a.element.symbol, t, pos[a.index]))
            bonds += [(start + i, start + j) for i, j in rbonds]
        elif res.name in ('WAT', 'HOH'):
            real = [a for a in ra if a.element is not None]
            assert [a.element.symbol for a in real] == ['O', 'H', 'H'], [a.name for a in ra]
            for a in real:
                mapping[a.index] = len(atoms)
                atoms.append((a.element.symbol, WATER_O if a.element.symbol == 'O' else WATER_H, pos[a.index]))
            bonds += [(start, start + 1), (start, start + 2)]
        elif res.name in ion_types:
            a = ra[0]
            mapping[a.index] = len(atoms)
            atoms.append((a.element.symbol, ion_types[res.name], pos[a.index]))
        # the remaining residues are protein (already included by pdbxyz)
    neigh = [[] for _ in atoms]
    for i, j in bonds:
        neigh[i].append(j + 1)
        neigh[j].append(i + 1)
    xyz_path = Path(f'{out_prefix}.xyz')
    with open(xyz_path, 'w') as f:
        f.write(f'{len(atoms):6d}  converted from {Path(str(rst)).name}\n')
        for k, (el, t, p) in enumerate(atoms):
            f.write(f'{k + 1:6d}  {el:3s}{p[0]:12.6f}{p[1]:12.6f}{p[2]:12.6f}{t:6d}' + ''.join(f'{n:6d}' for n in sorted(neigh[k])) + '\n')
    key_path = Path(f'{out_prefix}.key')
    with open(key_path, 'w') as f:
        for _, _, key in solutes:
            f.write('\n'.join(l for l in open(key).read().split('\n') if not l.startswith('parameters')) + '\n')
    box = crd.getBoxVectors()
    return xyz_path, key_path, box, mapping


def _whole_molecules_only(fn):
    """OpenMM 8.5.2 TinkerFiles recognizes ATP (and other ligand nucleotides) as a nucleic acid and returns only part
    of the atoms (34 of 43: the extra phosphates are left out), which breaks the topology. If the recognition function
    does not cover the whole molecule, return None and TinkerFiles falls back to the generic treatment."""
    def wrapped(molecule, atoms):
        out = fn(molecule, atoms)
        if out and sum(len(a) for a, _ in out) != len(molecule):
            return None
        return out
    wrapped._mlmmpol = True
    return wrapped


def _generic_on_error(fn):
    """Truncated protein fragments (QM/AMOEBA validation clusters) break the TinkerFiles amino-acid recognition at the
    ends; recognition only names the residues (parameters come from the atom types), so if it fails the fragment goes
    to the generic treatment."""
    def wrapped(molecule, atoms):
        try:
            return fn(molecule, atoms)
        except (AttributeError, KeyError, IndexError, TypeError):
            return None
    wrapped._mlmmpol = True
    return wrapped


if not getattr(app.TinkerFiles._processNucleicAcidChain, '_mlmmpol', False):
    app.TinkerFiles._processNucleicAcidChain = staticmethod(
        _whole_molecules_only(app.TinkerFiles._processNucleicAcidChain))
if not getattr(app.TinkerFiles._processPeptideChain, '_mlmmpol', False):
    app.TinkerFiles._processPeptideChain = staticmethod(
        _whole_molecules_only(_generic_on_error(app.TinkerFiles._processPeptideChain)))


def load_amoeba(xyz_path, key_path, box, cutoff=0.9, eps=1e-5, polarization='mutual'):
    """OpenMM AMOEBA system (PME) from the .xyz + amoebabio18.prm + the solute keys."""
    tk = app.TinkerFiles(str(xyz_path), [BIO18, str(key_path)], periodicBoxVectors=box)
    system = tk.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=cutoff * unit.nanometer,
                             polarization=polarization, mutualInducedTargetEpsilon=eps)
    return tk, system
