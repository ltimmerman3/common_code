# SPARC imports
import sparc
from sparc.calculator import SPARC
# ASE imports
from ase.units import Hartree, eV, Bohr, Angstrom, kB, _hplanck, J
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo, CrystalThermo
# Standard lib imports
import sys
import os
from json import load, dump
import pickle
# Installed library imports
import numpy as np

conversion_factor = 658.2119 # 2 pi rad / fs to meV

import warnings
warnings.filterwarnings("ignore")

# Functions

# Constant params
temp = 473.0
natom = 120
runs = ['0.25', '0.3', '0.35', '0.5']
data = {}
with open('vdos_dict.pckl', 'rb') as f:
    pdos = pickle.load(f)
for i in range(4):
    N_eq = pdos[runs[i]]['N_eq']
    omega, vdos = pdos[runs[i]]['time_feq'], 2.0 / N_eq * np.abs(pdos[runs[i]]['facf'].real)
    normed_omega = 1E15 * omega
    e_omega = _hplanck * J * normed_omega
    # This normalization should be consistent with ASE built in -> depends on if supercell is fed
    normed_vdos =  natom * 3 / np.trapezoid(vdos[:(N_eq//8)], e_omega[:(N_eq//8)]) * vdos[:(N_eq//8)]
    thermo = CrystalThermo(normed_vdos, e_omega[:(N_eq//8)])
    ref_helm, ref_internal, ref_vib = thermo.get_helmholtz_energy(temp), thermo.get_internal_energy(temp), thermo.get_entropy(temp)
    data[runs[i]] = {'helm': ref_helm, 'internal': ref_internal, 'vib': ref_vib}
with open("./messy_inaccurate_thermo_zeolite_mesh_convergence.json", "w") as f:
    dump(data, f)
    