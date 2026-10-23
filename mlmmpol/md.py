"""Restartable ML/MM molecular dynamics: minimization, equilibration and production with periodic checkpoints.

run_md writes, in out_dir:
  topo.pdb        topology and positions after equilibration;
  prod.dcd        production trajectory (one frame per report interval, wrapped into the box);
  energies.csv    t_ps, potential and kinetic energy, temperature, volume, density;
  restart.xml     checkpoint (removed at the end); if present when run_md starts, the run continues from it;
  final.xml       final state;  done   speed in ns/day, written at the end.
The integrator is Langevin (1 ps^-1); with mts=True a multiple-time-step Langevin integrator with the fast forces
(force group 0: bonded terms and the MLIP) every half step, as set by create_mupe_system / set_mts_groups.
"""
import os
import struct
import time
from pathlib import Path

import openmm as mm
from openmm import app, unit

from .minimize import minimize as robust_minimize

KB_KJ = 0.00831446261815324


def _truncate_dcd(path, n_atoms, n_frames):
    """Cut a DCD after n_frames (header + fixed-size frames with unit cell) and fix the frame counters."""
    with open(path, 'r+b') as fd:
        fd.seek(92)
        comments = struct.unpack('<i', fd.read(4))[0]
        header = 104 + comments + 8
        frame = (6 * 8 + 8) + 3 * (4 * n_atoms + 8)
        fd.seek(0, os.SEEK_END)
        if (fd.tell() - header) // frame < n_frames:
            raise RuntimeError(f'{path}: fewer frames than the checkpoint')
        fd.truncate(header + n_frames * frame)
        fd.seek(8)
        fd.write(struct.pack('<i', n_frames))
        fd.seek(20)
        fd.write(struct.pack('<i', n_frames - 1))


def run_md(system, topology, positions, out_dir, ns=1.0, box=None, equil_ps=200.0, temperature=300.0, pressure=1.0,
           timestep_fs=1.0, mts=True, report_ps=1.0, checkpoint_ps=50, platform='CUDA', properties=None,
           minimize=True, seed=None, log=print):
    """Run (or continue) a simulation of `system` and write the outputs in out_dir (see the module docstring).

    ns: production length; equil_ps: equilibration before production (not saved); temperature in K; pressure in bar
    (None: NVT); timestep_fs: outer time step (with mts=True the fast forces use half of it); report_ps: interval of the
    trajectory and energy output; checkpoint_ps: interval of restart.xml (in report intervals of 1 ps by default);
    properties: platform properties (default for CUDA: mixed precision). Returns the speed in ns/day.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))     # do not modify the caller's System
    if pressure is not None:
        system.addForce(mm.MonteCarloBarostat(pressure * unit.bar, temperature * unit.kelvin, 25))
    T = temperature * unit.kelvin
    if mts:
        integ = mm.MTSLangevinIntegrator(T, 1 / unit.picosecond, timestep_fs * unit.femtosecond, [(1, 1), (0, 2)])
    else:
        integ = mm.LangevinMiddleIntegrator(T, 1 / unit.picosecond, timestep_fs * unit.femtosecond)
    if seed is not None:
        integ.setRandomNumberSeed(seed)
    plat = mm.Platform.getPlatformByName(platform)
    props = properties if properties is not None else ({'Precision': 'mixed'} if platform == 'CUDA' else {})
    ctx = mm.Context(system, integ, plat, props)
    steps_report = int(round(report_ps * 1000 / timestep_fs))
    n_out = int(round(ns * 1000 / report_ps))
    masses = [system.getParticleMass(k).value_in_unit(unit.dalton) for k in range(system.getNumParticles())]
    ndof = 3 * sum(1 for m in masses if m > 0) - system.getNumConstraints()
    restart = out / 'restart.xml'
    t0 = time.time()
    if restart.exists():
        state = mm.XmlSerializer.deserialize(restart.read_text())
        ctx.setState(state)
        k_done = int(round((state.getTime().value_in_unit(unit.picosecond) - equil_ps) / report_ps))
        lines = (out / 'energies.csv').read_text().split('\n')
        (out / 'energies.csv').write_text('\n'.join(lines[:1 + k_done]) + '\n')
        _truncate_dcd(out / 'prod.dcd', topology.getNumAtoms(), k_done)
        log(f'restart: {k_done * report_ps:g} ps of production already done (of {ns * 1000:g}); continuing')
        dcd = app.DCDFile(open(out / 'prod.dcd', 'r+b'), topology, report_ps * 1e-3, append=True)
        fcsv = open(out / 'energies.csv', 'a')
    else:
        ctx.setPositions(positions)
        if box is not None:
            ctx.setPeriodicBoxVectors(*box)
        if minimize:
            robust_minimize(ctx, 10.0, 5000, log=log)
        ctx.setVelocitiesToTemperature(T, seed or 1)
        integ.step(int(round(equil_ps * 1000 / timestep_fs)))
        ctx.setTime(equil_ps * unit.picosecond)
        app.PDBFile.writeFile(topology, ctx.getState(getPositions=True).getPositions(), open(out / 'topo.pdb', 'w'))
        dcd = app.DCDFile(open(out / 'prod.dcd', 'wb'), topology, report_ps * 1e-3)
        fcsv = open(out / 'energies.csv', 'w')
        fcsv.write('t_ps,Epot_kJmol,Ekin_kJmol,T_K,volume_nm3,density_gcm3\n')
        k_done = 0
    k_start = k_done
    mass = sum(masses)
    for k in range(k_done, n_out):
        integ.step(steps_report)
        st = ctx.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=True)
        dcd.writeModel(st.getPositions(), periodicBoxVectors=st.getPeriodicBoxVectors())
        a, b, c = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        v = a[0] * b[1] * c[2]
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        fcsv.write(f'{(k + 1) * report_ps:g},{st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.3f},'
                   f'{ke:.3f},{2 * ke / (ndof * KB_KJ):.2f},{v:.4f},{mass / v * 1.66053906660 / 1000:.5f}\n')
        fcsv.flush()
        if (k + 1) % checkpoint_ps == 0 and k + 1 < n_out:
            dcd._file.flush()
            tmp = out / 'restart.xml.tmp'
            tmp.write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True, getVelocities=True)))
            tmp.replace(restart)
    fcsv.close()
    (out / 'final.xml').write_text(mm.XmlSerializer.serialize(ctx.getState(getPositions=True, getVelocities=True)))
    restart.unlink(missing_ok=True)
    ns_run = (n_out - k_start) * report_ps / 1000 + (0 if k_start else equil_ps / 1000)
    speed = ns_run / ((time.time() - t0) / 86400)
    (out / 'done').write_text(f'{speed:.2f} ns/day\n')
    log(f'finished: {speed:.2f} ns/day')
    return speed
