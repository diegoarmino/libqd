import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import h5py
import math
import pandas as pd
from IPython.display import display
import cupy as cp
import cupyx.scipy.sparse as csparse
import cupyx.scipy.sparse.linalg as csla
import h5py

def solve_coupled_morse_2d(m, De, a, lam, N=250, x_min=-1.0, x_max=6.0, num_states=10):
    """
    Solves the 2D coupled Morse Hamiltonian using the Colbert-Miller DVR (FGH) method.
    Returns the eigenvalues and reshaped 2D eigenvectors.
    """
    hbar = 1.0  # Assuming atomic units
    
    # 1. Define the 1D Grid
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # 2. Build 1D Kinetic Energy Matrix (Colbert-Miller formula)
    T1D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                T1D[i, j] = (np.pi**2) / 3.0
            else:
                T1D[i, j] = 2.0 * (-1)**(i - j) / (i - j)**2
    
    T1D *= (hbar**2) / (2.0 * m * dx**2)
    T1D_sparse = sparse.csr_matrix(T1D)
    I1D_sparse = sparse.eye(N, format='csr')
    
    # 3. Build 2D Kinetic Energy Matrix using Kronecker products
    # T2D = (T1D \otimes I) + (I \otimes T1D)
    T2D = sparse.kron(T1D_sparse, I1D_sparse) + sparse.kron(I1D_sparse, T1D_sparse)
    
    # 4. Build 2D Potential Energy Matrix
    # We use indexing='ij' so that reshaping aligns with the Kronecker product logic
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    y1 = 1.0 - np.exp(-a * X1)
    y2 = 1.0 - np.exp(-a * X2)
    
    V = De * (y1**2 + y2**2) - lam * y1 * y2
    
    # Flatten V and put it on the diagonal of a sparse matrix
    V_flat = V.flatten()
    V2D = sparse.diags(V_flat, format='csr')
    
    # 5. Full Hamiltonian
    H = T2D + V2D
    
    # 6. Diagonalize (Find bottom 'num_states' eigenvalues)
    # 'SA' means Smallest Algebraic eigenvalues (the lowest energies)
    print("Diagonalizing Hamiltonian... (this may take a few minutes)")
    eigenvalues, eigenvectors = sla.eigsh(H, k=num_states, which='SA')
    
    # The eigenvectors are flattened (N^2,). Reshape them back to 2D (N, N)
    wavefunctions = [eigenvectors[:, i].reshape((N, N)) for i in range(num_states)]
    
    return eigenvalues, wavefunctions, X1, X2, V

def solve_coupled_morse_2d_h5(m, De, a, lam, N=250, x_min=-1.0, x_max=6.0, num_states=10, filename=None):
    """
    Solves the 2D coupled Morse Hamiltonian using the Colbert-Miller DVR (FGH) method.
    If 'filename' is provided, saves the results to an HDF5 file.
    
    Returns:
        eigenvalues (1D array)
        wavefunctions (3D array of shape: [state_index, x1_grid, x2_grid])
        X1, X2, V (2D arrays for the grid and potential)
    """
    hbar = 1.0  # Assuming atomic units
    
    # 1. Define the 1D Grid
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    
    # 2. Build 1D Kinetic Matrix
    T1D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                T1D[i, j] = (np.pi**2) / 3.0
            else:
                T1D[i, j] = 2.0 * (-1)**(i - j) / (i - j)**2
    
    T1D *= (hbar**2) / (2.0 * m * dx**2)
    T1D_sparse = sparse.csr_matrix(T1D)
    I1D_sparse = sparse.eye(N, format='csr')
    
    # 3. Build 2D Kinetic Matrix
    T2D = sparse.kron(T1D_sparse, I1D_sparse) + sparse.kron(I1D_sparse, T1D_sparse)
    
    # 4. Build 2D Potential Matrix
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    y1 = 1.0 - np.exp(-a * X1)
    y2 = 1.0 - np.exp(-a * X2)
    
    V = De * (y1**2 + y2**2) - lam * y1 * y2
    V2D = sparse.diags(V.flatten(), format='csr')
    
    # 5. Diagonalize
    H = T2D + V2D
    print(f"Diagonalizing Hamiltonian ({N*N}x{N*N} grid)...")
    eigenvalues, eigenvectors = sla.eigsh(H, k=num_states, which='SA')
    
    # Pack wavefunctions into a single 3D NumPy array: shape (num_states, N, N)
    # This makes it much easier to save to disk and plot later.
    wavefuncs = np.array([eigenvectors[:, i].reshape((N, N)) for i in range(num_states)])
    
    # ==========================================
    # 6. Save to Disk (if requested)
    # ==========================================
    if filename is not None:
        print(f"Saving eigenstates to {filename}...")
        with h5py.File(filename, 'w') as f:
            # Save parameters as Metadata (Attributes)
            f.attrs['m'] = m
            f.attrs['De'] = De
            f.attrs['a'] = a
            f.attrs['lam'] = lam
            f.attrs['N'] = N
            f.attrs['x_min'] = x_min
            f.attrs['x_max'] = x_max
            f.attrs['num_states'] = num_states
            f.attrs['dx'] = dx
            
            # Save massive arrays as Datasets
            # 'gzip' compression drastically reduces file size for wavefunctions
            f.create_dataset('energies', data=eigenvalues)
            f.create_dataset('wavefunctions', data=wavefuncs, compression='gzip')
            f.create_dataset('X1', data=X1, compression='gzip')
            f.create_dataset('X2', data=X2, compression='gzip')
            f.create_dataset('V', data=V, compression='gzip')
            
        print("Save complete!")

    return eigenvalues, wavefuncs, X1, X2, V


#def solve_coupled_morse_2d_gpu_h5(m, De, a, lam, N=512, x_min=-2.5, x_max=12.0, num_states=20, filename=None):
#    """
#    Solves the 2D coupled Morse Hamiltonian using the Colbert-Miller DVR (FGH) method ON THE GPU.
#    If 'filename' is provided, saves the results to an HDF5 file.
#    """
#    hbar = 1.0  # Assuming atomic units
#
#    # 1. Define the 1D Grid (ON GPU)
#    x = cp.linspace(x_min, x_max, N)
#    dx = x[1] - x[0]
#
#    # 2. Build 1D Kinetic Matrix (Vectorized on GPU)
#    i, j = cp.meshgrid(cp.arange(N), cp.arange(N), indexing='ij')
#    diff = cp.abs(i - j)
#    diff[diff == 0] = 1  # Prevent division by zero
#
#    # Alternating sign: 1 if even distance, -1 if odd distance
#    sign = cp.where(diff % 2 == 0, 1.0, -1.0)
#
#    T1D = 2.0 * sign / (diff**2)
#    cp.fill_diagonal(T1D, (cp.pi**2) / 3.0)
#
#    T1D *= (hbar**2) / (2.0 * m * dx**2)
#    T1D_sparse = csparse.csr_matrix(T1D)
#    I1D_sparse = csparse.eye(N, format='csr')
#
#    # 3. Build 2D Kinetic Matrix (ON GPU)
#    T2D = csparse.kron(T1D_sparse, I1D_sparse) + csparse.kron(I1D_sparse, T1D_sparse)
#
#    # 4. Build 2D Potential Matrix (ON GPU)
#    X1, X2 = cp.meshgrid(x, x, indexing='ij')
#    y1 = 1.0 - cp.exp(-a * X1)
#    y2 = 1.0 - cp.exp(-a * X2)
#
#    V = De * (y1**2 + y2**2) - lam * y1 * y2
#    V2D = csparse.diags(V.flatten(), format='csr')
#
#    # 5. Diagonalize (ON GPU)
#    H = T2D + V2D
#    print(f"Diagonalizing Hamiltonian on GPU ({N*N}x{N*N} sparse grid)...")
#    eigenvalues_gpu, eigenvectors_gpu = csla.eigsh(H, k=num_states, which='SA')
#
#    # 6. Move data back to CPU RAM for processing and saving
#    eigenvalues = eigenvalues_gpu.get()
#    eigenvectors = eigenvectors_gpu.get()
#    X1_cpu = X1.get()
#    X2_cpu = X2.get()
#    V_cpu = V.get()
#    dx_cpu = float(dx.get()) # Convert 0D array to standard python float
#
#    # Pack wavefunctions into a single 3D NumPy array: shape (num_states, N, N)
#    wavefuncs = np.array([eigenvectors[:, idx].reshape((N, N)) for idx in range(num_states)])
#
#    # ==========================================
#    # 7. Save to Disk (ON CPU)
#    # ==========================================
#    if filename is not None:
#        print(f"Saving eigenstates to {filename}...")
#        with h5py.File(filename, 'w') as f:
#            # Save parameters as Metadata (Attributes)
#            f.attrs['m'] = m
#            f.attrs['De'] = De
#            f.attrs['a'] = a
#            f.attrs['lam'] = lam
#            f.attrs['N'] = N
#            f.attrs['x_min'] = x_min
#            f.attrs['x_max'] = x_max
#            f.attrs['num_states'] = num_states
#            f.attrs['dx'] = dx_cpu
#
#            # Save massive arrays as Datasets
#            f.create_dataset('energies', data=eigenvalues)
#            f.create_dataset('wavefunctions', data=wavefuncs, compression='gzip')
#            f.create_dataset('X1', data=X1_cpu, compression='gzip')
#            f.create_dataset('X2', data=X2_cpu, compression='gzip')
#            f.create_dataset('V', data=V_cpu, compression='gzip')
#
#        print("Save complete!")
#
#    return eigenvalues, wavefuncs, X1_cpu, X2_cpu, V_cpu

def solve_coupled_morse_2d_gpu_h5(m, De, a, lam, N=512, x_min=-2.5, x_max=12.0, num_states=50, filename=None):
    """
    Solves the 2D coupled Morse Hamiltonian using the Colbert-Miller DVR.
    Uses an O(N^2) memory LinearOperator to completely eliminate VRAM bottlenecks.
    """
    hbar = 1.0  # Assuming atomic units

    # 1. Define the 1D Grid (ON GPU)
    x = cp.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    # 2. Build 1D Kinetic Matrix (Dense, tiny: N x N)
    i, j = cp.meshgrid(cp.arange(N), cp.arange(N), indexing='ij')
    diff = cp.abs(i - j)
    diff[diff == 0] = 1

    sign = cp.where(diff % 2 == 0, 1.0, -1.0)
    T1D = 2.0 * sign / (diff**2)
    cp.fill_diagonal(T1D, (cp.pi**2) / 3.0)
    T1D *= (hbar**2) / (2.0 * m * dx**2)

    # 3. Build 2D Potential Matrix (Dense, N x N)
    X1, X2 = cp.meshgrid(x, x, indexing='ij')
    y1 = 1.0 - cp.exp(-a * X1)
    y2 = 1.0 - cp.exp(-a * X2)
    V = De * (y1**2 + y2**2) - lam * y1 * y2

    # 4. Define the Linear Operator (The Memory Saver!)
    def H_matvec(v):
        # Reshape the flattened 1D vector back into a 2D wavepacket
        Psi = v.reshape((N, N))

        # Apply Kinetic Energy: T_x(Psi) + T_y(Psi)
        # Because T1D is symmetric, T1D.T is just T1D.
        T_Psi = cp.matmul(T1D, Psi) + cp.matmul(Psi, T1D)

        # Apply Potential Energy
        V_Psi = V * Psi

        # Return the flattened result
        return (T_Psi + V_Psi).ravel()

    # Create the CuPy LinearOperator wrapper
    H_op = csla.LinearOperator(shape=(N*N, N*N), matvec=H_matvec, dtype=cp.float64)

    # 5. Diagonalize (ON GPU)
    print(f"Diagonalizing Hamiltonian on GPU using LinearOperator ({N}x{N} grid)...")
    eigenvalues_gpu, eigenvectors_gpu = csla.eigsh(H_op, k=num_states, which='SA')

    # 6. Move data back to CPU RAM
    eigenvalues = eigenvalues_gpu.get()
    eigenvectors = eigenvectors_gpu.get()
    X1_cpu = X1.get()
    X2_cpu = X2.get()
    V_cpu = V.get()
    dx_cpu = float(dx.get())

    # Pack wavefunctions
    wavefuncs = np.array([eigenvectors[:, idx].reshape((N, N)) for idx in range(num_states)])

    # 7. Save to Disk (ON CPU)
    if filename is not None:
        print(f"Saving eigenstates to {filename}...")
        with h5py.File(filename, 'w') as f:
            f.attrs['m'] = m; f.attrs['De'] = De; f.attrs['a'] = a; f.attrs['lam'] = lam
            f.attrs['N'] = N; f.attrs['x_min'] = x_min; f.attrs['x_max'] = x_max
            f.attrs['num_states'] = num_states; f.attrs['dx'] = dx_cpu

            f.create_dataset('energies', data=eigenvalues)
            f.create_dataset('wavefunctions', data=wavefuncs, compression='gzip')
            f.create_dataset('X1', data=X1_cpu, compression='gzip')
            f.create_dataset('X2', data=X2_cpu, compression='gzip')
            f.create_dataset('V', data=V_cpu, compression='gzip')
        print("Save complete!")

    return eigenvalues, wavefuncs, X1_cpu, X2_cpu, V_cpu

def plot_eigenstates_polyads(sim_data, num_polyads=5, xlim=(-1.0, 4.0), ylim=(-1.0, 4.0)):
    """
    Plots the 2D eigenstates arranged mathematically by Polyad (one polyad per row).
    Includes auto-formatting, font scaling, and dynamic grid truncation.
    """
    energies = sim_data['energies']
    wavefuncs = sim_data['wavefunctions']
    X1, X2, V = sim_data['X1'], sim_data['X2'], sim_data['V']
    De = sim_data['De']
    
    # 1. Dynamic Grid Truncation (Fixes the massive bottom white space)
    # Calculate the absolute maximum number of polyads we can plot with the loaded states
    max_possible_polyads = int(np.ceil((-1 + np.sqrt(1 + 8 * len(energies))) / 2))
    
    if num_polyads > max_possible_polyads:
        print(f"Notice: Reducing requested polyads from {num_polyads} to {max_possible_polyads} "
              f"to match the {len(energies)} available states and prevent blank space.")
        num_polyads = max_possible_polyads
    
    # 2. Create the Grid (Slightly tighter figsize multiplier)
    fig, axes = plt.subplots(num_polyads, num_polyads, 
                             figsize=(2.8 * num_polyads, 2.8 * num_polyads),
                             squeeze=False)
    
    fig.suptitle(f"Eigenstates Organized by Polyad Structure", fontsize=16, y=0.95)
    levels = np.linspace(0, De, 10)
    
    state_idx = 0
    for P in range(num_polyads):
        for col in range(num_polyads):
            ax = axes[P, col]
            
            # Polyad P has P+1 states. Plot only if col <= P
            if col <= P and state_idx < len(energies):
                # Plot potential contours
                ax.contour(X1, X2, V, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

                # Plot wavefunction density
                density = np.abs(wavefuncs[state_idx])**2
                ax.imshow(density, extent=[X1.min(), X1.max(), X2.min(), X2.max()],
                          origin='lower', cmap='magma', aspect='auto')

                # 3. Fine-tuned Typography (Fixes text and tick overlapping)
                ax.set_title(f"State {state_idx} (P={P})\nE = {energies[state_idx]:.4f}", fontsize=10)
#                ax.set_xlabel("$x_1$", fontsize=9)
#                ax.set_ylabel("$x_2$", fontsize=9)
                
                # Shrink the axis tick marks
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Apply custom zoom limits
                if xlim is not None: ax.set_xlim(xlim)
                if ylim is not None: ax.set_ylim(ylim)
                
                state_idx += 1
            else:
                # Completely deactivate empty subplots (top-right triangle)
                ax.axis('off')

    # 4. Enforce strict spacing rules
    # hspace=0.8 forces a massive vertical buffer so titles NEVER touch the X-axis above them
    fig.subplots_adjust(hspace=0.8, wspace=0.4, top=0.88, bottom=0.05, left=0.05, right=0.95)
    
    plt.show()
    return fig

def print_eigenenergies_by_polyad(sim_data, max_polyads=None):
    """
    Prints the eigenenergies to the terminal, grouped mathematically by Polyad.
    Also calculates the internal energy spread (splitting) of each polyad.
    """
    energies = sim_data['energies']
    total_states = len(energies)

    print("\n" + "="*50)
    print("      EIGENENERGIES ORGANIZED BY POLYAD")
    print("="*50)

    state_idx = 0
    P = 0  # Polyad number (P = v1 + v2)

    while state_idx < total_states:
        # Stop if user defined a maximum number of polyads to print
        if max_polyads is not None and P >= max_polyads:
            break

        states_in_polyad = P + 1

        # Check if the file ran out of states mid-polyad
        if state_idx + states_in_polyad > total_states:
            print(f"\n[ Polyad {P} ]  -- Incomplete ({total_states - state_idx} of {states_in_polyad} states available)")
        else:
            print(f"\n[ Polyad {P} ]  (v1 + v2 = {P})")

        polyad_energies =[]

        # Print all states belonging to this polyad
        for _ in range(states_in_polyad):
            if state_idx < total_states:
                e = energies[state_idx]
                polyad_energies.append(e)
                print(f"   |State {state_idx:2d}⟩ : E = {e:.6f} a.u.")
                state_idx += 1

        # Print the internal splitting of the polyad
        if len(polyad_energies) > 1:
            polyad_spread = polyad_energies[-1] - polyad_energies[0]
            print(f"   ----------------------------------------")
            print(f"   -> Polyad Splitting (ΔE) = {polyad_spread:.6f} a.u.")

        P += 1

    print("="*50 + "\n")

# 1. Helper function to load the data (Add this to your library if you haven't already)
def load_eigenstates(filename):
    """Loads a saved eigenstate .h5 file into a Python dictionary."""
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
        for key in f.attrs.keys():
            data[key] = f.attrs[key]
    return data

import pandas as pd
from IPython.display import display

def generate_energy_difference_table(sim_data, N_states=20, filename="energy_differences.html"):
    """
    Calculates the energy difference matrix and saves it as a color-coded HTML table.
    The headers contain both the state index and the eigenenergy.
    """
    energies = sim_data['energies']
    N_states = min(N_states, len(energies))
    E_subset = energies[:N_states]

    # Calculate the Difference Matrix
    diff_matrix = np.abs(E_subset[:, np.newaxis] - E_subset)

    # Mask the exact diagonal so it doesn't wash out the colormap
    np.fill_diagonal(diff_matrix, np.nan)

    # Create Labels with HTML line breaks (<br>)
    # Result:
    # |0⟩
    # E=4.9026
    labels =[f"|{i}⟩<br>E={E_subset[i]:.4f}" for i in range(N_states)]

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(diff_matrix, index=labels, columns=labels)

    # Apply styling
    styled_table = df.style.format("{:.6f}", na_rep="-").background_gradient(cmap='magma_r', axis=None)

    # Save to HTML
    styled_table.to_html(filename, escape=False) # escape=False ensures <br> is rendered as HTML!
    print(f"Saved colored table to '{filename}'. Open it in your web browser!")

    return styled_table

# ===================
# Run the calculation 
# ===================

# Run DVR calculation in GPU and store results to disk.
#------------------------------------------------------------------
# Parameters for DVR 
#m = 1.0
#De = 20.0
#a = 0.8
#lam = 0.4  # The coupling strength
#
# Run calculation
#energies, wavefuncs, X1, X2, V = solve_coupled_morse_2d_gpu_h5(
#    m, De, a, lam,
#    N=320, num_states=15, x_min=-2.5, x_max=12.0,
#    filename="morse_eigenstates_50states.h5"
#)
#------------------------------------------------------------------

# Print the Eigenenergies and check for Polyads
#print("\n--- Eigenenergies ---")
#for i, E in enumerate(energies):
#    print(f"State {i}: E = {E:.6f} a.u.")
#
#print(f"\nRabi Splitting (State 1 & 2) = {energies[2] - energies[1]:.6f} a.u.")


# ==========================================
# Load and Plot the Polyads from Disk
# ==========================================
# 1. Load the data
h5_filename = "morse_eigenstates_50states.h5"
data = load_eigenstates(h5_filename)

# 2. Plot the first 7 Polyads (this will plot 28 states in a beautiful triangle)
# You can easily change xlim and ylim here!
fig = plot_eigenstates_polyads(data, num_polyads=7, xlim=(-1.0, 4.0), ylim=(-1.0, 4.0))

# 3. Generate the Energy Differences Table (First 20 states)
table = generate_energy_difference_table(data, N_states=20, filename="energy_differences.html")

# (Optional) Display in Colab directly
#display(table)

# Print the first 10 polyads to the terminal
print_eigenenergies_by_polyad(data, max_polyads=5)
