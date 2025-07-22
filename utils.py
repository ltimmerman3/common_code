import matplotlib.pyplot as plt
import re
import numpy as np
from numba import jit
from joblib import Parallel, delayed
from rdfs import get_all_atom_rdfs

def parse_sparc_output_file(file:str, parameter:str, nlines=None):
    """
    Extracts parameters from aimd or other file based on parameter name

    Inputs:
    file                : full path to aimd, log, or other file
    parameter           : parameter of interest
    nlines     

    To Do: return multiple parameters      
    """
    result = []
    escaped_parameter = re.escape(parameter)
    pattern = re.compile(rf'{escaped_parameter}')

    def file_content():
        with open(file, 'r') as f_log:
            for line in f_log:
                yield line.strip()

    f_log_content = file_content()
    for idx, line in enumerate(f_log_content):
        #if pattern.search(line):
        if pattern.match(line):
            if nlines:
                data = [process_line(next(f_log_content)) for _ in range(nlines)]
                result.append(data[0] if len(data) == 1 else data)
            else:
                if '.aimd' in file:
                    split_line = line.split(':')
                    result.append(float(split_line[0]) if len(split_line) == 1 else float(split_line[-1]))
                elif '.out' in file:
                    tmp_split_line = line.split('#')
                    split_line = tmp_split_line[-1].strip().split(' ')[0]
                    result.append(int(split_line[0:-1]))
                elif 'MLFF_RESTART' in file:
                    tmp = int(next(f_log_content).strip())
                    result.append(tmp)
                else:
                    tmp_split_line = line.split(':')[-1].strip().split(' ')[0]
                    result.append(float(tmp_split_line))
                
    
    return result[0] if len(result) == 1 else np.array(result)

def process_line(line):
    row = list(map(float, line.split(' ')))
    return row[0] if len(row) == 1 else row

def get_pearsons_corr_coeff(x:np.ndarray, y:np.ndarray):
    """
    Returns the pearsons correlation coefficient between two vectors
    """
    x_mean = np.mean(x)
    x_var = np.var(x)
    y_mean = np.mean(y)
    y_var = np.var(y)
    x -= x_mean
    y -= y_mean
    corr = np.sum(x*y)/np.sqrt(x_var*y_var)/(len(x)-1)
    return corr

def plot_2d_data(x_data: np.array, y_data: np.array, x_label="", y_label="", data_labels: list=[],
                  title="", show=True, return_fig=True, overlay=True, 
                  fill_between=False, colors: list=[]):
    """
    A flexible function for plotting 2D data with various options.

    Args:
    x_data: List or array of x-axis data.
    y_data: List or array of y-axis data.
    x_label: String for the x-axis label.
    y_label: String for the y-axis label.
    data_labels: Labels for each of the entries in y_label
    title: String for the plot title.
    show: Boolean indicating whether to display the plot immediately.
    return_fig: Boolean indicating whether to return the figure object.
    overlay: Boolean indicating whether to overlay multiple curves.
    fill_between: Boolean indicating whether to fill the area between curves.

    Returns:
    fig: matplotlib.figure.Figure object (optional).
    """

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(2.5,5),dpi=300)

    # Plot the data
    if data_labels:
        for i, data in enumerate(y_data):
            ax.plot(x_data, data, label=f"{data_labels[i]}", color=colors[i])
            
        #if y_data.shape[0] > 1:
        #    ax.legend()
    else:
        if y_data.shape[0] > 1:
            for i, data in enumerate(y_data):
                ax.plot(x_data, data, color=colors[i])
        else:
            ax.plot(x_data, y_data.reshape(-1))

    # Fill the area between curves if requested
    if fill_between and len(ax.lines) > 1:
        ax.fill_between(x_data, ax.lines[-2].get_ydata(), ax.lines[-1].get_ydata(), alpha=0.2)
        
    ax.tick_params(axis='both', labelsize=14)

    # Show the plot if requested
    if show:
        plt.show()
        
    # Optionally return the figure object
    if return_fig:
        return fig

def plot_bar_chart(data: np.array, labels: list, methods: list, title: str="", x_label: str="", y_label: str="", color_palette="viridis", style: str="", scale: str=""):
    """
    Function to create a bar plot from a list of numbers and labels.

    Args:
    data: List of numbers to be plotted on the y-axis.
    labels: List of labels for the corresponding data points.
    title: Optional title for the plot.
    x_label: Optional label for the x-axis.
    y_label: Optional label for the y-axis.
    color_palette: Optional color palette to use for the bars.

    Returns:
    fig: matplotlib.figure.Figure object.
    """

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(5,5),dpi=300)
    
    width = 0.35
      # Calculate positions for side-by-side comparison
    if style == "side-by-side":
        x_positions = np.arange(len(labels)) - width / 2
    else:
        x_positions = np.arange(len(labels))

    # Create the bar chart
    ax.bar(x_positions, data[:,0], width, label=methods[0])
    ax.bar(x_positions + width, data[:,1], width, label=methods[1])
    
    # Set x-axis tick positions and labels
    ax.set_xticks(x_positions + width / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if scale == 'log':
        ax.set_yscale('log')

    # Set grid and other visual options
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    return fig

def blocking_analysis(atoms):
    """
    Computes variance of rdf (x coordinate of first/max peak) for radial (all atom) distribution function
    
    Inputs:
    atoms               : object containing all relevant images
    
    Returns:
    data                : 2d array containing block lengths (steps) and computed variances
    
    Based on formalism laid out here: https://doi.org/10.1016%2FS1574-1400(09)00502-7
    """
    # Assume traj length is 10,000
    n = np.array([1,2,3,4,5,6,7,8,9,10,100,200,300,400,500,1000],dtype=int)
    M = 10000/n
    M = M.astype(np.int64)
    # Will want parallelization since these are independent
    def get_var_for_n(idx):
        # Stride here is just n, number of images to use for each rdf
        stride = n[idx]
        x_vals = np.zeros(M[idx])
        for _m in range(M[idx]):
            tmp_atoms = atoms[_m*stride:(_m+1)*stride]
            tmp_g = get_all_atom_rdfs(tmp_atoms, skip=1, resolution=200)
            _index = np.argmax(tmp_g['allatom'][:,1])
            x_vals[_m] = tmp_g['allatom'][_index,0]
        return np.std(x_vals)/np.sqrt(M[idx])
    results = Parallel(n_jobs=-1)(delayed(get_var_for_n)(i) for i in range(len(n)))
    dist = np.array(results)
    return dist,n