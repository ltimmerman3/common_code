import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import simpson as simps


# TODO: Check for absolute accuracy relative to some reference
def compute_vacf(velocities):
    """
    Computes the VACF for a specific species using FFT convolution.
    
    Parameters:
    - velocities (np.ndarray): An array of shape (T, N_i, 3) containing the velocities of N_i atoms in 3 dimensions over T timesteps. Should be sqrt mass weighted if multiple species present
    - species_indices (np.ndarray): Indices of atoms belonging to the species of interest.
    
    Returns:
    - vacf (np.ndarray): The normalized VACF for the specified species.
    """
    T = velocities.shape[0]
    N_i = velocities.shape[1]
    v_mw = velocities - np.mean(velocities, axis=0, keepdims=True)  # Mean over time for each atom and component
    vacf_sum = np.zeros(T)
    for i in range(N_i):
        for j in range(3):  # Iterate over x, y, z components
            fft_vacf = fftconvolve(v_mw[:, i, j], v_mw[::-1, i, j], mode='full')[T-1:]
            vacf_sum += fft_vacf.reshape(-1)

    vacf_avg = vacf_sum / (3 * N_i)
    vacf_avg /= vacf_avg[0]

    return vacf_avg

# TODO: Check normalization conditions: may be double normalizing
def compute_self_diffusion_coefficient(vacf, dt):
    """
    Computes the self-diffusion coefficient from the VACF using numerical integration.
    
    Parameters:
    - vacf (np.ndarray): The VACF array.
    - dt (float): The time step between velocity samples.
    
    Returns:
    - D (float): The self-diffusion coefficient.
    """
    integral = simps(vacf, dx=dt)
    D = integral / 3
    return D

if __name__ == '__main__':
    velocities = np.random.randn(1000, 10, 3)
    vacf = compute_vacf_for_species(velocities)
    D = compute_self_diffusion_coefficient(vacf, 1)
    print(D)
