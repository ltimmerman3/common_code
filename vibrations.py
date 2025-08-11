from scipy.signal import fftconvolve
from scipy.fft import fft, fftfreq
import numpy as np
from ase.io import read

def autocorrelation_fftconvolve(a:np.ndarray):
    Nsteps = a.shape[0]
    Nparticles = a.shape[1]
    a_zeromean = a - np.mean(a, axis=0, keepdims=True)  # Mean over time for each atom and component
    acf_sum = np.zeros(Nsteps)
    for i in range(Nparticles):
        for j in range(3):  # Iterate over x, y, z components
            fft_vacf = fftconvolve(a_zeromean[:, i, j], a_zeromean[::-1, i, j], mode='full')[Nsteps-1:]
            acf_sum += fft_vacf.reshape(-1)

    acf_avg = acf_sum / acf_sum[0]

    return acf_avg

def vdos_fft(acf:np.ndarray, padding:float, dt):
    n = len(acf) * (1 + padding)
    vdos = fft(acf, n)
    freqs = fftfreq(n, dt)
    return vdos, freqs

def get_vacf_vdos(dt, atoms=None, filename:str=None):
    if atoms is None:
        atoms = read(filename, index=':')
    vels = np.zeros((len(atoms), len(atoms[0]), 3))
    masses = atoms[0].get_masses()
    vels *= np.sqrt(masses[np.newaxis,:,np.newaxis])
    for i in range(len(atoms)):
        vels[i] = atoms[i].get_velocities()
    vacf = autocorrelation_fftconvolve(vels)
    vdos, freqs = vdos_fft(vacf, 1, dt)
    return vacf, vdos, freqs