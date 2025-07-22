import numpy as np
from numba import jit
from IPython.display import clear_output
from joblib import Parallel, delayed
from scipy.special import binom

@jit(nopython=True)
def compute_single_atom_histogram(atom_a, atoms, histogram, cell, hist, resolution, dr, tmp_dist):
    """
    compute the histogram of distances between two atoms
    """
    for j, _atom_b in enumerate(atoms):
        for k, _length in enumerate(cell):
            dx = abs(atom_a[k] - _atom_b[k])
            r = min(dx, abs(_length - dx))
            tmp_dist[k] = np.power(r,2)
        dist = np.sqrt(tmp_dist[0] + tmp_dist[1] + tmp_dist[2])
        if dist == 0.0:
            continue
        index = int(dist / dr)
        if index < resolution:
            hist[index] += 1.0
    return histogram

@jit(nopython=True)
def compute_pair_histogram(atoms, cell, hist, resolution, dr, tmp_dist):
    """
    compute the histogram of distances between two atoms
    """
    for i, _atom_a in enumerate(atoms):
        for j, _atom_b in enumerate(atoms[i+1:]):
            for k, _length in enumerate(cell):
                dx = abs(_atom_a[k] - _atom_b[k])
                r = min(dx, abs(_length - dx))
                tmp_dist[k] = np.power(r,2)
            dist = np.sqrt(tmp_dist[0] + tmp_dist[1] + tmp_dist[2])
            if dist == 0.0:
                continue
            index = int(dist / dr)
            if index < resolution:
                hist[index] += 1.0
    return hist

@jit(nopython=True)
def compute_traj_pair_histogram(steps, stride, data, cell, hist, resolution, dr, tmp_dist):
    """
    All atom or self rdf where all other atoms have been filtered out of data. Avoids double counting.
    """
    for step in range(steps):
        for i, _atom_a in enumerate(data[(step*stride):((step+1)*stride),:]):
            for j, _atom_b in enumerate(data[step*stride+i+1:((step+1)*stride),:]):
                for k, _length in enumerate(cell):
                    dx = abs(_atom_a[k] - _atom_b[k])
                    r = min(dx, abs(_length - dx))
                    tmp_dist[k] = np.power(r,2)
                dist = np.sqrt(tmp_dist[0] + tmp_dist[1] + tmp_dist[2])
                if np.isclose(dist, 0.0):
                    continue
                index = int(dist / dr)
                if index < resolution:
                    hist[index] += 2.0
    return hist

@jit(nopython=True)
def compute_traj_partial_histogram(steps, ref_atoms, ref_stride, target_atoms, target_stride, cell, hist, resolution, dr, tmp_dist):
    """
    Computes rdf between two unique element types
    """
    for step in range(steps):
        for i, _atom_a in enumerate(ref_atoms[(step*ref_stride):((step+1)*ref_stride),:]):
            for j, _atom_b in enumerate(target_atoms[step*target_stride:((step+1)*target_stride),:]):
                for k, _length in enumerate(cell):
                    dx = abs(_atom_a[k] - _atom_b[k])
                    r = min(dx, abs(_length - dx))
                    tmp_dist[k] = np.power(r,2)
                dist = np.sqrt(tmp_dist[0] + tmp_dist[1] + tmp_dist[2])
                if np.isclose(dist, 0.0):
                    continue
                index = int(dist / dr)
                if index < resolution:
                    hist[index] += 1.0
    return hist

def get_aa_partial_g_rs(atoms, skip=1, resolution=200):
    """
    Returns all rdfs (all atom and partial)
    Missing self interactions (self-rdfs)
    """
    g_of_rs = {}
    nsteps = len(atoms)
    natoms = len(atoms[0])
    clear_output(wait=True)
    print("Read atoms: ", nsteps)
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    compute_steps = int(nsteps / skip)
    tmp_positions = np.zeros((compute_steps*shape_tuple[0],shape_tuple[1]))
    #for idx,_traj in enumerate(atoms):
    #    tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    for idx in range(compute_steps):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = atoms[idx*skip].positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution).reshape(-1,1)
    g_of_rs['allatom'] = np.hstack((x,get_all_atom_rdf(cell, natoms, cutoff, dr, compute_steps, resolution, tmp_positions).reshape(-1,1)))

    tmp_atom_list = atoms[0].get_atomic_numbers()
    uniq_atms = np.unique(tmp_atom_list)
    n_uniq_atms = len(uniq_atms)
    n_partial_hist = binom(n_uniq_atms,2)
    for i, ref_atm in enumerate(uniq_atms):
        for j, target_atm in enumerate(uniq_atms[i+1:]):
            g_of_rs[str(ref_atm)+'-'+str(target_atm)] = np.hstack((x,get_partial_atom_rdf(cell, natoms, cutoff, dr, compute_steps, resolution, tmp_positions, tmp_atom_list, ref_atm, target_atm).reshape(-1,1)))
                    
    return g_of_rs

def get_self_rdfs(atoms, skip=1, resolution=200):
    """
    Only computes self rdf
    """
    g_of_rs = {}
    nsteps = len(atoms)
    natoms = len(atoms[0])
    clear_output(wait=True)
    print("Read atoms: ", nsteps)
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    compute_steps = int(nsteps / skip)
    tmp_positions = np.zeros((compute_steps*shape_tuple[0],shape_tuple[1]))
    #for idx,_traj in enumerate(atoms):
    #    tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    for idx in range(compute_steps):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = atoms[idx*skip].positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution).reshape(-1,1)
    tmp_atom_list = atoms[0].get_atomic_numbers()
    uniq_atms = np.unique(tmp_atom_list)
    for i, ref_atm in enumerate(uniq_atms):
        _atoms = np.where(tmp_atom_list == ref_atm)[0]
        _stride = len(_atoms)
        tmp_data = tmp_positions.reshape((compute_steps,natoms,3))
        _atoms_pos = tmp_data[:,_atoms,:].reshape(-1,3)
        g_of_rs[str(ref_atm)+'-'+str(ref_atm)] = np.hstack((x,get_all_atom_rdf(cell, _stride, cutoff, dr, compute_steps, resolution, _atoms_pos, self_rdf=str(ref_atm)).reshape(-1,1)))
    return g_of_rs
                    
def get_all_atom_rdfs(atoms, skip=1, resolution=200):
    """
    Only computes all atom rdfs. Gets them all
    
    Modified to account for possibility of wanting rdf on single image (for blocking)
    """
    g_of_rs = {}
    nsteps = len(atoms)
    natoms = len(atoms[0])
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    clear_output(wait=True)
    compute_steps = int(nsteps / skip)
    tmp_positions = np.zeros((compute_steps*shape_tuple[0],shape_tuple[1]))
    #for idx,_traj in enumerate(atoms):
    #    tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    for idx in range(compute_steps):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = atoms[idx*skip].positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution).reshape(-1,1)
    g_of_rs['allatom'] = np.hstack((x,get_all_atom_rdf(cell, natoms, cutoff, dr, compute_steps, resolution, tmp_positions).reshape(-1,1)))
    return g_of_rs

def get_partial_atom_rdfs(atoms, skip=1, resolution=200):
    """
    Only computes partial rdfs (minus self rdf)
    """
    g_of_rs = {}
    nsteps = len(atoms)
    natoms = len(atoms[0])
    clear_output(wait=True)
    print("Read atoms: ", nsteps)
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    compute_steps = int(nsteps / skip)
    tmp_positions = np.zeros((compute_steps*shape_tuple[0],shape_tuple[1]))
    #for idx,_traj in enumerate(atoms):
    #    tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    for idx in range(compute_steps):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = atoms[idx*skip].positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution).reshape(-1,1)
    tmp_atom_list = atoms[0].get_atomic_numbers()
    uniq_atms = np.unique(tmp_atom_list)
    for i, ref_atm in enumerate(uniq_atms):
        for j, target_atm in enumerate(uniq_atms[i+1:]):
            g_of_rs[str(ref_atm)+'-'+str(target_atm)] = np.hstack((x,get_partial_atom_rdf(cell, natoms, cutoff, dr, compute_steps, resolution, tmp_positions, tmp_atom_list, ref_atm, target_atm).reshape(-1,1)))
    return g_of_rs             
                
def get_all_atom_rdf(cell: list, natoms: int, cutoff: float, dr: float, nsteps: int, resolution: int, tmp_positions: np.array, self_rdf: str='', prefix: str='default'):
    """
    Returns a single all atom rdf given inputs
    """
    #norm = compute_normalization_factor(cell[0] * cell[1] * cell[2], natoms, natoms, cutoff, dr, nsteps, resolution)
    hist = np.zeros(resolution)
    tmp_dist = np.array([0,0,0],dtype=float)
    hist = compute_traj_pair_histogram(nsteps, natoms, tmp_positions, cell, hist, resolution, dr, tmp_dist)
    x = np.linspace(0,dr*resolution,resolution)
    norm = np.trapz(hist,x)
    if np.isclose(norm,0):
        norm = 1
    g_of_r = hist / norm
    """
    _save = np.hstack((np.linspace(0,dr*resolution,resolution).reshape(-1,1),g_of_r.reshape(-1,1)))
    if self_rdf:
        ref = f"{self_rdf}-{self_rdf}"
    else:
        ref = "allatom"
    np.savetxt(f'{prefix}_g_{ref}.txt',_save,delimiter=',')
    """
    return g_of_r

def get_partial_atom_rdf(cell: list, natoms: int, cutoff: float, dr: float, nsteps: int, resolution: int, tmp_positions: np.array, tmp_atom_list: np.array, ref_atm: int, target_atm: int, prefix: str='default'):
    """
    Computes a single partial rdf given inputs
    """
    ref_atoms = np.where(tmp_atom_list == ref_atm)[0]
    ref_stride = len(ref_atoms)
    target_atoms = np.where(tmp_atom_list == target_atm)[0]
    target_stride = len(target_atoms)
    #norm = compute_normalization_factor(cell[0] * cell[1] * cell[2], ref_stride, target_stride, cutoff, dr, nsteps, resolution)
    hist = np.zeros(resolution)
    tmp_data = tmp_positions.reshape((nsteps,natoms,3))
    ref_atoms_pos = tmp_data[:,ref_atoms,:].reshape(-1,3)
    target_atoms_pos = tmp_data[:,target_atoms,:].reshape(-1,3)
    tmp_dist = np.array([0,0,0],dtype=float)
    hist = compute_traj_partial_histogram(nsteps, ref_atoms_pos, ref_stride, target_atoms_pos, target_stride, cell, hist, resolution, dr, tmp_dist)
    x = np.linspace(0,dr*resolution,resolution)
    norm = np.trapz(hist,x)
    if np.isclose(norm,0):
        norm = 1
    g_of_r = hist / norm
    #_save = np.hstack((np.linspace(0,dr*resolution,resolution).reshape(-1,1),g_of_r.reshape(-1,1)))
    #np.savetxt(f'{prefix}_g_{ref_atm}-{target_atm}.txt',_save,delimiter=',')
    return g_of_r
                
def get_stability_dist(aimd_g_of_r: np.array, atoms, ref_atm: int, targ_atm: int):
    """
    Iterate through loaded data given a step size and compute rdfs and MAE for rdfs as a function of time
    for target trajectory relative to aimd trajectory. 
    """
   
    n_eval = len(atoms) - 1000
    dist = np.zeros(n_eval)
    n_data_tau = 1000
    skip = 1
    resolution = 200
    natoms = len(atoms[0])
    nsteps = len(atoms)
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    tmp_positions = np.zeros((nsteps*shape_tuple[0],shape_tuple[1]))
    for idx,_traj in enumerate(atoms):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution)
    tmp_atom_list = atoms[0].get_atomic_numbers()
    def get_rdf_dist(idx):
        data_eval = tmp_positions[idx*natoms:(n_data_tau+idx)*natoms,:]
        tmp_g = get_partial_atom_rdf(cell, natoms, cutoff, dr, n_data_tau, resolution, data_eval, tmp_atom_list, ref_atm, targ_atm)
        diff = abs(aimd_g_of_r-tmp_g)
        return np.trapz(diff,x)
    results = Parallel(n_jobs=-1)(delayed(get_rdf_dist)(i) for i in range(n_eval))
    dist = np.array(results)
    return dist    

def get_self_stability_dist(aimd_g_of_r: np.array, atoms, ref_atm: int):
    """
    Get stability metric for self rdf
    """
    
    n_eval = len(atoms) - 1000
    dist = np.zeros(n_eval)
    n_data_tau = 1000
    skip = 1
    resolution = 200
    natoms = len(atoms[0])
    nsteps = len(atoms)
    cell = atoms[0].cell.lengths()
    shape_tuple = atoms[0].positions.shape
    tmp_positions = np.zeros((nsteps*shape_tuple[0],shape_tuple[1]))
    for idx,_traj in enumerate(atoms):
        tmp_positions[idx*shape_tuple[0]:(idx+1)*shape_tuple[0],:] = _traj.positions
    cutoff = min(cell) / 2.0
    dr = cutoff / resolution
    x = np.linspace(0,dr*resolution,resolution)
    tmp_atom_list = atoms[0].get_atomic_numbers()
    _atoms = np.where(tmp_atom_list == ref_atm)[0]
    _stride = len(_atoms)
    tmp_data = tmp_positions.reshape((nsteps,natoms,3))
    _atoms_pos = tmp_data[:,_atoms,:].reshape(-1,3)
    def get_rdf_dist(idx):
        data_eval = _atoms_pos[idx*_stride:(n_data_tau+idx)*_stride,:]
        tmp_g = get_all_atom_rdf(cell, _stride, cutoff, dr, n_data_tau, resolution, data_eval)
        diff = abs(aimd_g_of_r-tmp_g)
        return np.trapz(diff,x)
    results = Parallel(n_jobs=-1)(delayed(get_rdf_dist)(i) for i in range(n_eval))
    dist = np.array(results)
    return dist