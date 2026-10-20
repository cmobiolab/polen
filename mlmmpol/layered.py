"""Setup of the plugin layered polarization (MLPolMultipoleForce.setLayeredPolarization).

Switching groups = residues (one per residue that contains no central atom). Reference atom: water O, amino-acid
CA, otherwise the first heavy atom. Atoms of the central residues are always mutual (-1).
"""

REF_ATOM_NAMES = ('O', 'CA')      # water (HOH: O), amino acids (CA)


def layered_groups(topology, center_atoms):
    center = set(int(i) for i in center_atoms)
    group_of = [-1] * topology.getNumAtoms()
    group_ref = []
    for res in topology.residues():
        atoms = list(res.atoms())
        if any(a.index in center for a in atoms):
            continue
        by_name = {a.name: a for a in atoms}
        ref = next((by_name[n] for n in REF_ATOM_NAMES if n in by_name), None)
        if ref is None:
            ref = next(a for a in atoms if a.element is None or a.element.symbol != 'H')
        for a in atoms:
            group_of[a.index] = len(group_ref)
        group_ref.append(ref.index)
    return group_of, group_ref


def configure_layered_polarization(force, topology, center_atoms, r_in, r_out, tau=0.02):
    """Enable the layers on an MLPolMultipoleForce: mutual up to r_in, smooth switch up to r_out, direct beyond (nm)."""
    group_of, group_ref = layered_groups(topology, center_atoms)
    force.setLayeredPolarization([int(i) for i in center_atoms], group_of, group_ref, float(r_in), float(r_out), float(tau))
    return group_of, group_ref
