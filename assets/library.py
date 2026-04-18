import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from IPython.display import HTML
import h5py  # For out-of-core disk storage
from scipy.signal import spectrogram
from scipy.linalg import hankel, svd, pinv
from scipy.signal import find_peaks  # <--- NEW IMPORT

# for SOFT with disk storage
import h5py
import time
import argparse
import os

# Increase the Colab animation memory limit to 100 MB
#matplotlib.rcParams['animation.embed_limit'] = 100.0

def run_soft_simulation(
    m=1.0, De=20.0, a=0.8, c_coup=0.4,
    N=256, x_min=-2.0, x_max=10.0,
    d1=1.0, d2=0.0,                  # Displacement
    dt=0.01, steps=8000, frame_interval=40,
    itp_steps=300, dtau=0.05,
    save_filename=None               # File path to save data
):
    """
    Runs a Split-Operator Fourier Transform (SOFT) simulation.
    Now includes Local Energy and Autocorrelation tracking.
    """
    print(f"--- Starting Simulation (Displacement: d1={d1}, d2={d2}) ---")
    
    # 1. Setup Grids
    dx = (x_max - x_min) / N
    x = np.linspace(x_min, x_max, N, endpoint=False)
    p = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    X1, X2 = np.meshgrid(x, x)
    P1, P2 = np.meshgrid(p, p)
    
    # 2. Setup Hamiltonian
    Y1 = (1.0 - np.exp(-a * X1))
    Y2 = (1.0 - np.exp(-a * X2))
    
    # Global Potentials
    V = De*Y1**2 + De*Y2**2 + c_coup*Y1*Y2
    T = (P1**2 + P2**2) / (2.0 * m)
    
    # --- NEW: Independent Local Operators for Expectation Values ---
    V1_local = De * Y1**2
    V2_local = De * Y2**2
    T1_local = (P1**2) / (2.0 * m)
    T2_local = (P2**2) / (2.0 * m)
    # ---------------------------------------------------------------
    
    # 3. Imaginary Time Propagation (Find Ground State)
    print("  Finding ground state via Imaginary Time Propagation...")
    psi_itp = np.exp(-((X1)**2 + (X2)**2))
    psi_itp = psi_itp / np.sqrt(np.sum(np.abs(psi_itp)**2) * dx**2)
    psi_itp = psi_itp.astype(complex)
    
    U_V_itp = np.exp(-V * dtau / 2.0)
    U_T_itp = np.exp(-T * dtau)
    
    for _ in range(itp_steps):
        psi_itp = psi_itp * U_V_itp
        psi_p = np.fft.fft2(psi_itp)
        psi_p = psi_p * U_T_itp
        psi_itp = np.fft.ifft2(psi_p) * U_V_itp
        psi_itp = psi_itp / np.sqrt(np.sum(np.abs(psi_itp)**2) * dx**2)
        
    psi_GS = psi_itp.copy()
    
    # 4. Displace the Ground State
    print(f"  Displacing ground state by ({d1}, {d2})...")
    psi_p_GS = np.fft.fft2(psi_GS)
    shift_operator = np.exp(-1j * (P1 * d1 + P2 * d2))
    psi = np.fft.ifft2(psi_p_GS * shift_operator)
    
    # --- NEW: Save Initial State for Autocorrelation ---
    psi_initial = psi.copy()
    # ---------------------------------------------------

    # 5. Real-Time Propagation
    print("  Propagating Real-Time Dynamics...")
    U_V = np.exp(-1j * V * dt / 2.0)
    U_T = np.exp(-1j * T * dt)
    
    frames =[]
    time_array = []
    exp_X1, exp_X2 = [],[]
    energy_1, energy_2 = [], []
    autocorr =[]
    
    for step in range(steps + 1):
        if step % frame_interval == 0:
            frames.append(np.abs(psi)**2)
            time_array.append(step * dt)
            
            # Expectation values for Position
            px1 = np.sum(np.abs(psi)**2, axis=0) * dx
            px2 = np.sum(np.abs(psi)**2, axis=1) * dx
            exp_X1.append(np.sum(x * px1))
            exp_X2.append(np.sum(x * px2))
            
            # --- NEW: Autocorrelation C(t) ---
            C_t = np.sum(np.conj(psi_initial) * psi) * (dx**2)
            autocorr.append(C_t)
            
            # --- NEW: Local Energy Expectation Values ---
            # Potential Energy 
            E_V1 = np.sum(np.abs(psi)**2 * V1_local) * (dx**2)
            E_V2 = np.sum(np.abs(psi)**2 * V2_local) * (dx**2)
            
            # Kinetic Energy (requires inverse FFT of T * psi_p)
            psi_p_current = np.fft.fft2(psi)
            T1_psi = np.fft.ifft2(T1_local * psi_p_current)
            T2_psi = np.fft.ifft2(T2_local * psi_p_current)
            
            # <psi | T | psi> (Only keep real part to avoid float precision errors)
            E_T1 = np.real(np.sum(np.conj(psi) * T1_psi)) * (dx**2)
            E_T2 = np.real(np.sum(np.conj(psi) * T2_psi)) * (dx**2)
            
            energy_1.append(E_T1 + E_V1)
            energy_2.append(E_T2 + E_V2)
            
        # Standard SOFT propagation
        psi = psi * U_V
        psi = np.fft.ifft2(np.fft.fft2(psi) * U_T) * U_V

    print("  Simulation Complete!\n")
    
    # Pack results into a dictionary
    results = {
        'frames': np.array(frames), # Converts list of 2D arrays to a 3D volume for compression
        'time_array': np.array(time_array),
        'exp_X1': np.array(exp_X1), 'exp_X2': np.array(exp_X2),
        'energy_1': np.array(energy_1), 'energy_2': np.array(energy_2),
        'autocorr': np.array(autocorr),
        'x': x, 'X1': X1, 'X2': X2, 'V': V,
        'dx': dx, 'De': De, 'x_min': x_min, 'x_max': x_max,
        'd1': d1, 'd2': d2
    }
    
    # --- NEW: Save to Disk ---
    if save_filename:
        print(f"  Saving data to {save_filename}...")
        np.savez_compressed(save_filename, **results)
        print(f"  Saved {(os.path.getsize(save_filename) / (1024*1024)):.2f} MB to disk.")
    
    return results


def run_soft_simulation_hdf5(
    m=1.0, De=20.0, a=0.8, c_coup=0.4,
    N=256, x_min=-2.0, x_max=10.0,
    d1=1.0, d2=0.0,
    dt=0.001, steps=500000, frame_interval=20,
    itp_steps=300, dtau=0.05,
    save_filename="simulation_data.h5",
    flush_interval=1000
):
    print(f"--- Starting HDF5-Backed Simulation ---")
    print(f"Parameters: steps={steps}, N={N}, shift=({d1}, {d2})")
    print(f"Saving to: {save_filename}\n")
    
    # 1. Setup Grids
    dx = (x_max - x_min) / N
    x = np.linspace(x_min, x_max, N, endpoint=False)
    p = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    X1, X2 = np.meshgrid(x, x)
    P1, P2 = np.meshgrid(p, p)
    
    Y1, Y2 = (1.0 - np.exp(-a * X1)), (1.0 - np.exp(-a * X2))
    V = De*Y1**2 + De*Y2**2 + c_coup*Y1*Y2
    T = (P1**2 + P2**2) / (2.0 * m)
    
    V1_local, V2_local = De * Y1**2, De * Y2**2
    T1_local, T2_local = (P1**2) / (2.0 * m), (P2**2) / (2.0 * m)
    
    # 2. Imaginary Time Propagation
    print("  Finding ground state...")
    psi = np.exp(-((X1)**2 + (X2)**2))
    psi = (psi / np.sqrt(np.sum(np.abs(psi)**2) * dx**2)).astype(complex)
    
    U_V_itp = np.exp(-V * dtau / 2.0)
    U_T_itp = np.exp(-T * dtau)
    
    for _ in range(itp_steps):
        psi = psi * U_V_itp
        psi = np.fft.ifft2(np.fft.fft2(psi) * U_T_itp) * U_V_itp
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2) * dx**2)
        
    # 3. Displace Ground State
    print("  Displacing ground state...")
    psi = np.fft.ifft2(np.fft.fft2(psi) * np.exp(-1j * (P1 * d1 + P2 * d2)))
    psi_initial = psi.copy()

    # 4. Initialize HDF5 File
    print(f"  Initializing HDF5 file...")
    h5f = h5py.File(save_filename, 'w')
    
    # Save static grids and parameters
    h5f.create_dataset('x', data=x); h5f.create_dataset('X1', data=X1)
    h5f.create_dataset('X2', data=X2); h5f.create_dataset('V', data=V)
    for k, v in [('dx', dx), ('De', De), ('x_min', x_min), ('x_max', x_max), ('d1', d1), ('d2', d2)]:
        h5f.attrs[k] = v
        
    # Dynamically resizable datasets for time-series data
    ds_frames = h5f.create_dataset('frames', shape=(0, N, N), maxshape=(None, N, N), dtype='float32', compression="gzip")
    ds_time = h5f.create_dataset('time_array', shape=(0,), maxshape=(None,), dtype='float32')
    ds_x1 = h5f.create_dataset('exp_X1', shape=(0,), maxshape=(None,), dtype='float32')
    ds_x2 = h5f.create_dataset('exp_X2', shape=(0,), maxshape=(None,), dtype='float32')
    ds_e1 = h5f.create_dataset('energy_1', shape=(0,), maxshape=(None,), dtype='float64')
    ds_e2 = h5f.create_dataset('energy_2', shape=(0,), maxshape=(None,), dtype='float64')
    ds_auto = h5f.create_dataset('autocorr', shape=(0,), maxshape=(None,), dtype='complex64')

    # 5. Real-Time Propagation
    print("  Propagating Real-Time Dynamics...")
    U_V = np.exp(-1j * V * dt / 2.0)
    U_T = np.exp(-1j * T * dt)
    
    # RAM Buffers
    buf_frames, buf_time, buf_x1, buf_x2 = [], [], [], []
    buf_e1, buf_e2, buf_auto = [], [], []
    
    def flush_buffers():
        n_new = len(buf_time)
        if n_new == 0: return
        n_old = ds_time.shape[0]
        n_total = n_old + n_new
        
        # Resize datasets on disk
        ds_frames.resize(n_total, axis=0); ds_time.resize(n_total, axis=0)
        ds_x1.resize(n_total, axis=0);     ds_x2.resize(n_total, axis=0)
        ds_e1.resize(n_total, axis=0);     ds_e2.resize(n_total, axis=0)
        ds_auto.resize(n_total, axis=0)
        
        # Write buffer to disk
        ds_frames[n_old:n_total] = np.array(buf_frames, dtype='float32')
        ds_time[n_old:n_total] = buf_time
        ds_x1[n_old:n_total] = buf_x1;     ds_x2[n_old:n_total] = buf_x2
        ds_e1[n_old:n_total] = buf_e1;     ds_e2[n_old:n_total] = buf_e2
        ds_auto[n_old:n_total] = buf_auto
        
        # Clear RAM
        buf_frames.clear(); buf_time.clear(); buf_x1.clear(); buf_x2.clear()
        buf_e1.clear(); buf_e2.clear(); buf_auto.clear()

    start_time = time.time()
    
    for step in range(steps + 1):
        if step % frame_interval == 0:
            buf_frames.append(np.abs(psi)**2)
            buf_time.append(step * dt)
            
            px1 = np.sum(np.abs(psi)**2, axis=0) * dx
            px2 = np.sum(np.abs(psi)**2, axis=1) * dx
            buf_x1.append(np.sum(x * px1))
            buf_x2.append(np.sum(x * px2))
            buf_auto.append(np.sum(np.conj(psi_initial) * psi) * (dx**2))
            
            E_V1 = np.sum(np.abs(psi)**2 * V1_local) * (dx**2)
            E_V2 = np.sum(np.abs(psi)**2 * V2_local) * (dx**2)
            psi_p = np.fft.fft2(psi)
            E_T1 = np.real(np.sum(np.conj(psi) * np.fft.ifft2(T1_local * psi_p))) * (dx**2)
            E_T2 = np.real(np.sum(np.conj(psi) * np.fft.ifft2(T2_local * psi_p))) * (dx**2)
            
            buf_e1.append(E_T1 + E_V1)
            buf_e2.append(E_T2 + E_V2)
            
            # Flush to disk periodically
            if len(buf_time) >= flush_interval:
                flush_buffers()
                print(f"    [Step {step}/{steps}] Flushed to disk. Elapsed time: {time.time()-start_time:.1f}s")
            
        # Standard SOFT propagation
        psi = psi * U_V
        psi = np.fft.ifft2(np.fft.fft2(psi) * U_T) * U_V

    # Final flush
    flush_buffers()
    h5f.close() # Close the file safely!
    
    file_size = os.path.getsize(save_filename) / (1024 * 1024)
    print(f"\n  Simulation Complete!")
    print(f"  Total time: {time.time()-start_time:.2f} seconds.")
    print(f"  Saved {file_size:.2f} MB to '{save_filename}'.")


def run_soft_simulation_hdf5_gpu(
    m=1.0, De=20.0, a=0.8, c_coup=0.4,
    N=256, x_min=-2.5, x_max=12.0,
    d1=1.0, d2=0.0,
    dt=0.001, steps=500000,
    frame_interval=100,  # Less frequent (Heavy 2D arrays)
    obs_interval=2,     # More frequent (Lightweight 1D arrays)
    itp_steps=300, dtau=0.05,
    save_filename="simulation_data_gpu.h5",
    flush_interval=1000
):
    print(f"--- Starting GPU HDF5-Backed Simulation ---")
    print(f"Parameters: steps={steps}, N={N}, shift=({d1}, {d2})")
    print(f"Intervals: frames every {frame_interval}, observables every {obs_interval}")
    print(f"Saving to: {save_filename}\n")

    # 1. Setup Grids (ON GPU)
    dx = (x_max - x_min) / N
    x = cp.linspace(x_min, x_max, N, endpoint=False)
    p = cp.fft.fftfreq(N, d=dx) * 2 * cp.pi
    X1, X2 = cp.meshgrid(x, x)
    P1, P2 = cp.meshgrid(p, p)

    Y1, Y2 = (1.0 - cp.exp(-a * X1)), (1.0 - cp.exp(-a * X2))
    V = De*Y1**2 + De*Y2**2 - c_coup*Y1*Y2
    T = (P1**2 + P2**2) / (2.0 * m)

    V1_local, V2_local = De * Y1**2, De * Y2**2
    T1_local, T2_local = (P1**2) / (2.0 * m), (P2**2) / (2.0 * m)

    # 2. Imaginary Time Propagation (ON GPU)
    print("  Finding ground state...")
    psi = cp.exp(-((X1)**2 + (X2)**2))
    psi = (psi / cp.sqrt(cp.sum(cp.abs(psi)**2) * dx**2)).astype(cp.complex128)

    U_V_itp = cp.exp(-V * dtau / 2.0)
    U_T_itp = cp.exp(-T * dtau)

    for _ in range(itp_steps):
        psi = psi * U_V_itp
        psi = cp.fft.ifft2(cp.fft.fft2(psi) * U_T_itp) * U_V_itp
        psi = psi / cp.sqrt(cp.sum(cp.abs(psi)**2) * dx**2)

    # 3. Displace Ground State (ON GPU)
    print("  Displacing ground state...")
    psi = cp.fft.ifft2(cp.fft.fft2(psi) * cp.exp(-1j * (P1 * d1 + P2 * d2)))
    psi_initial = psi.copy()

    # 4. Initialize HDF5 File (ON CPU)
    print(f"  Initializing HDF5 file...")
    h5f = h5py.File(save_filename, 'w')

    h5f.create_dataset('x', data=x.get())
    h5f.create_dataset('X1', data=X1.get())
    h5f.create_dataset('X2', data=X2.get())
    h5f.create_dataset('V', data=V.get())

    for k, v in [('dx', dx), ('De', De), ('x_min', x_min), ('x_max', x_max), ('d1', d1), ('d2', d2)]:
        h5f.attrs[k] = v

    # 2D Datasets
    ds_frames = h5f.create_dataset('frames', shape=(0, N, N), maxshape=(None, N, N), dtype='float32', compression="gzip")
    ds_time_frames = h5f.create_dataset('time_frames', shape=(0,), maxshape=(None,), dtype='float32')

    # 1D Datasets
    ds_time_obs = h5f.create_dataset('time_obs', shape=(0,), maxshape=(None,), dtype='float32')
    ds_x1 = h5f.create_dataset('exp_X1', shape=(0,), maxshape=(None,), dtype='float32')
    ds_x2 = h5f.create_dataset('exp_X2', shape=(0,), maxshape=(None,), dtype='float32')
    ds_e1 = h5f.create_dataset('energy_1', shape=(0,), maxshape=(None,), dtype='float64')
    ds_e2 = h5f.create_dataset('energy_2', shape=(0,), maxshape=(None,), dtype='float64')
    ds_auto = h5f.create_dataset('autocorr', shape=(0,), maxshape=(None,), dtype='complex128')


    # 5. Real-Time Propagation
    print("  Propagating Real-Time Dynamics...")
    U_V = cp.exp(-1j * V * dt / 2.0)
    U_T = cp.exp(-1j * T * dt)

    # RAM Buffers
    buf_frames, buf_time_frames = [], []
    buf_time_obs, buf_x1, buf_x2 = [], [], []
    buf_e1, buf_e2, buf_auto = [], [], []

    def flush_buffers():
        # --- Flush Observables ---
        n_obs_new = len(buf_time_obs)
        if n_obs_new > 0:
            n_obs_old = ds_time_obs.shape[0]
            n_obs_total = n_obs_old + n_obs_new

            ds_time_obs.resize(n_obs_total, axis=0)
            ds_x1.resize(n_obs_total, axis=0); ds_x2.resize(n_obs_total, axis=0)
            ds_e1.resize(n_obs_total, axis=0); ds_e2.resize(n_obs_total, axis=0)
            ds_auto.resize(n_obs_total, axis=0)

            ds_time_obs[n_obs_old:n_obs_total] = buf_time_obs
            ds_x1[n_obs_old:n_obs_total] = buf_x1; ds_x2[n_obs_old:n_obs_total] = buf_x2
            ds_e1[n_obs_old:n_obs_total] = buf_e1; ds_e2[n_obs_old:n_obs_total] = buf_e2
            ds_auto[n_obs_old:n_obs_total] = buf_auto

            buf_time_obs.clear(); buf_x1.clear(); buf_x2.clear()
            buf_e1.clear(); buf_e2.clear(); buf_auto.clear()

        # --- Flush Frames ---
        n_frames_new = len(buf_time_frames)
        if n_frames_new > 0:
            n_frames_old = ds_time_frames.shape[0]
            n_frames_total = n_frames_old + n_frames_new

            ds_time_frames.resize(n_frames_total, axis=0)
            ds_frames.resize(n_frames_total, axis=0)

            ds_time_frames[n_frames_old:n_frames_total] = buf_time_frames
            ds_frames[n_frames_old:n_frames_total] = np.array(buf_frames, dtype='float32')

            buf_time_frames.clear(); buf_frames.clear()

    start_time = time.time()

    for step in range(steps + 1):
        is_frame_step = (step % frame_interval == 0)
        is_obs_step = (step % obs_interval == 0)

        if is_frame_step or is_obs_step:
            # Compute probability density ONCE if needed by either step
            psi_sq = cp.abs(psi)**2

            if is_frame_step:
                buf_frames.append(psi_sq.astype(cp.float32).get())
                buf_time_frames.append(step * dt)

            if is_obs_step:
                buf_time_obs.append(step * dt)

                #-------------------------------------------------
                # DELETE
                #px1 = cp.sum(psi_sq, axis=0) * dx
                #px2 = cp.sum(psi_sq, axis=1) * dx
                #buf_x1.append(float(cp.sum(x * px1) * dx))
                #buf_x2.append(float(cp.sum(x * px2) * dx))
                #-------------------------------------------------
                # Delete px1 and px2 entirely. Do the full 2D integral natively:
                buf_x1.append(float(cp.sum(psi_sq * X1) * (dx**2)))
                buf_x2.append(float(cp.sum(psi_sq * X2) * (dx**2)))
                buf_auto.append(complex(cp.sum(cp.conj(psi_initial) * psi) * (dx**2)))

                E_V1 = cp.sum(psi_sq * V1_local) * (dx**2)
                E_V2 = cp.sum(psi_sq * V2_local) * (dx**2)
                psi_p = cp.fft.fft2(psi)

                E_T1 = cp.real(cp.sum(cp.conj(psi) * cp.fft.ifft2(T1_local * psi_p))) * (dx**2)
                E_T2 = cp.real(cp.sum(cp.conj(psi) * cp.fft.ifft2(T2_local * psi_p))) * (dx**2)

                buf_e1.append(float(E_T1 + E_V1))
                buf_e2.append(float(E_T2 + E_V2))

            # Trigger flush based on observables buffer (since it fills faster)
            if len(buf_time_obs) >= flush_interval:
                flush_buffers()
                print(f"    [Step {step}/{steps}] Flushed to disk. Elapsed time: {time.time()-start_time:.1f}s")

        # Standard SOFT propagation
        psi = psi * U_V
        psi = cp.fft.ifft2(cp.fft.fft2(psi) * U_T) * U_V

    # Final flush
    flush_buffers()
    h5f.close()

    file_size = os.path.getsize(save_filename) / (1024 * 1024)
    print(f"\n  Simulation Complete!")
    print(f"  Total time: {time.time()-start_time:.2f} seconds.")
    print(f"  Saved {file_size:.2f} MB to '{save_filename}'.")


def visualize_single(sim_data, title=None):
    """
    Generates a 3-panel dashboard for a single quantum simulation.
    Returns a Matplotlib FuncAnimation object.
    """
    # Unpack data
    frames, time_array = sim_data['frames'], sim_data['time_array']
    exp_X1, exp_X2 = sim_data['exp_X1'], sim_data['exp_X2']
    x, X1, X2, V, dx, De = sim_data['x'], sim_data['X1'], sim_data['X2'], sim_data['V'], sim_data['dx'], sim_data['De']
    x_min, x_max = sim_data['x_min'], sim_data['x_max']
    
    if title is None:
        title = f"Quantum Dynamics (Shift $x_1={sim_data['d1']}$, $x_2={sim_data['d2']}$)"
        
    fig = plt.figure(figsize=(9, 12))
    fig.suptitle(title, fontsize=16)
    
    gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4, 1.5])
    gs.update(wspace=0.05, hspace=0.25)
    
    ax_main = plt.subplot(gs[1, 0])
    ax_top = plt.subplot(gs[0, 0], sharex=ax_main)
    ax_right = plt.subplot(gs[1, 1], sharey=ax_main)
    ax_exp = plt.subplot(gs[2, :])
    
    # Init 2D
    levels = np.linspace(0, 1.5 * De, 15)
    ax_main.contour(X1, X2, V, levels=levels, colors='white', alpha=0.3, linewidths=0.8)
    ax_main.plot([x_min, x_max],[x_min, x_max], 'w--', alpha=0.4)
    im = ax_main.imshow(frames[0], extent=[x_min, x_max, x_min, x_max], origin='lower', cmap='magma', vmax=np.max(frames[0]))
    ax_main.set(xlabel="Bond 1 ($x_1$)", ylabel="Bond 2 ($x_2$)", xlim=(-1, 8), ylim=(-1, 8))
    time_text = ax_main.text(0.05, 0.95, '', transform=ax_main.transAxes, color='white', fontsize=12, va='top')

    # Init 1D
    p1 = np.sum(frames[0], axis=0) * dx
    p2 = np.sum(frames[0], axis=1) * dx
    line_top, = ax_top.plot(x, p1, color='coral', lw=2.5)
    line_right, = ax_right.plot(p2, x, color='coral', lw=2.5)
    ax_top.axis('off'); ax_right.axis('off')
    ax_top.set_ylim(0, np.max(p1) * 1.2); ax_right.set_xlim(0, np.max(p2) * 1.2)
    
    # Init Expectations
    pad = (max(max(exp_X1), max(exp_X2)) - min(min(exp_X1), min(exp_X2))) * 0.15
    ax_exp.set_ylim(min(min(exp_X1), min(exp_X2)) - pad, max(max(exp_X1), max(exp_X2)) + pad)
    ax_exp.set_xlim(0, time_array[-1])
    line_e1, = ax_exp.plot([],[], color='deepskyblue', lw=2.5, label=r'$\langle x_1 \rangle$ (Bond 1)')
    line_e2, = ax_exp.plot([],[], color='crimson', lw=2.5, label=r'$\langle x_2 \rangle$ (Bond 2)')
    ax_exp.set(xlabel="Time (a.u.)", ylabel="Expectation Value")
    ax_exp.legend(loc='upper right'); ax_exp.grid(True, alpha=0.3)
    
    def update(idx):
        den = frames[idx]
        im.set_data(den)
        im.set_clim(vmin=0, vmax=np.max(den)) 
        line_top.set_ydata(np.sum(den, axis=0) * dx)
        line_right.set_xdata(np.sum(den, axis=1) * dx)
        line_e1.set_data(time_array[:idx+1], exp_X1[:idx+1])
        line_e2.set_data(time_array[:idx+1], exp_X2[:idx+1])
        time_text.set_text(f"Time: {time_array[idx]:.1f} a.u.")
        return[im, line_top, line_right, time_text, line_e1, line_e2]

    ani = FuncAnimation(fig, update, frames=len(frames), blit=True)
    plt.close()
    return ani

def visualize_dual_comparison(sim_A, sim_B, title="Quantum Competition: Normal Mode vs. Local Mode Transition"):
    """
    Generates a 6-panel side-by-side dashboard comparing two quantum simulations.
    Syncs low-frequency 2D frames with high-frequency 1D observables.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=18)

    gs = gridspec.GridSpec(3, 4, width_ratios=[4, 1.2, 4, 1.2], height_ratios=[1.2, 4, 1.5])
    gs.update(wspace=0.15, hspace=0.25)

    # Subplot assignments
    ax_main_A = plt.subplot(gs[1, 0])
    ax_top_A = plt.subplot(gs[0, 0], sharex=ax_main_A)
    ax_right_A = plt.subplot(gs[1, 1], sharey=ax_main_A)
    ax_exp_A = plt.subplot(gs[2, 0:2])

    ax_main_B = plt.subplot(gs[1, 2])
    ax_top_B = plt.subplot(gs[0, 2], sharex=ax_main_B)
    ax_right_B = plt.subplot(gs[1, 3], sharey=ax_main_B)
    ax_exp_B = plt.subplot(gs[2, 2:4])

    def init_system(sim, ax_m, ax_t, ax_r, ax_e, color, title):
        # Load static grids into RAM
        x_arr = np.array(sim['x'])
        X1_arr = np.array(sim['X1'])
        X2_arr = np.array(sim['X2'])
        V_arr = np.array(sim['V'])

        levels = np.linspace(0, 1.5 * sim['De'], 15)
        ax_m.contour(X1_arr, X2_arr, V_arr, levels=levels, colors='white', alpha=0.3, linewidths=0.8)
        ax_m.plot([sim['x_min'], sim['x_max']], [sim['x_min'], sim['x_max']], 'w--', alpha=0.4)

        im = ax_m.imshow(sim['frames'][0], extent=[sim['x_min'], sim['x_max'], sim['x_min'], sim['x_max']],
                         origin='lower', cmap='magma', vmax=np.max(sim['frames'][0]))
        ax_m.set(xlabel="Bond 1 ($x_1$)", ylabel="Bond 2 ($x_2$)", xlim=(-1, 8), ylim=(-1, 8))
        ax_m.set_title(title, fontsize=14)

        p1 = np.sum(sim['frames'][0], axis=0) * sim['dx']
        p2 = np.sum(sim['frames'][0], axis=1) * sim['dx']

        lt, = ax_t.plot(x_arr, p1, color=color, lw=2)
        lr, = ax_r.plot(p2, x_arr, color=color, lw=2)

        ax_t.axis('off'); ax_r.axis('off')
        ax_t.set_ylim(0, np.max(p1) * 1.2); ax_r.set_xlim(0, np.max(p2) * 1.2)

        le1, = ax_e.plot([],[], color='deepskyblue', lw=2.5, label=r'$\langle x_1 \rangle$')
        le2, = ax_e.plot([],[], color='crimson', lw=2.5, label=r'$\langle x_2 \rangle$')

        # Use the frames time array to set the x-axis limits of the expectation value plot
        #ax_e.set(xlim=(0, sim['time_frames'][-1]), xlabel="Time (a.u.)")
        ax_e.set(xlim=(sim['time_frames'][0], sim['time_frames'][-1]), xlabel="Time (a.u.)")
        ax_e.legend(loc='upper right')
        ax_e.grid(True, alpha=0.3)

        # Calculate padding safely
        pad = (np.max(sim['exp_X1']) - np.min(sim['exp_X1'])) * 0.15
        if pad == 0: pad = 0.5 # Fallback if expectation value is completely flat
        ax_e.set_ylim(min(np.min(sim['exp_X1']), np.min(sim['exp_X2'])) - pad,
                      max(np.max(sim['exp_X1']), np.max(sim['exp_X2'])) + pad)

        return im, lt, lr, le1, le2

    # Initialize plots
    im_A, lt_A, lr_A, le1_A, le2_A = init_system(sim_A, ax_main_A, ax_top_A, ax_right_A, ax_exp_A,
                                                 'mediumspringgreen', f"Normal Mode ($d={sim_A['d1']}$)")
    im_B, lt_B, lr_B, le1_B, le2_B = init_system(sim_B, ax_main_B, ax_top_B, ax_right_B, ax_exp_B,
                                                 'coral', f"Local Mode ($d={sim_B['d1']}$)")

    time_text = fig.text(0.5, 0.95, '', ha='center', color='black', fontsize=14)

    # ---------------------------------------------------------
    # Pre-load 1D arrays to standard RAM to ensure fast animation
    # (1D arrays are tiny, so this won't crash your memory)
    # ---------------------------------------------------------
    t_obs_A = np.array(sim_A['time_obs'])
    exp_X1_A = np.array(sim_A['exp_X1'])
    exp_X2_A = np.array(sim_A['exp_X2'])

    t_obs_B = np.array(sim_B['time_obs'])
    exp_X1_B = np.array(sim_B['exp_X1'])
    exp_X2_B = np.array(sim_B['exp_X2'])

    def update(idx):
        # Determine the exact physical time of the current frame
        current_time = sim_A['time_frames'][idx]

        # Determine the slice index for the high-res 1D observables
        # (This finds the index in time_obs closest to current_time)
        idx_obs_A = np.searchsorted(t_obs_A, current_time, side='right')
        idx_obs_B = np.searchsorted(t_obs_B, current_time, side='right')

        # --- Sys A ---
        den_A = sim_A['frames'][idx]
        im_A.set_data(den_A)
        im_A.set_clim(vmin=0, vmax=np.max(den_A))
        lt_A.set_ydata(np.sum(den_A, axis=0) * sim_A['dx'])
        lr_A.set_xdata(np.sum(den_A, axis=1) * sim_A['dx'])

        # Plot up to the synchronized high-res index
        le1_A.set_data(t_obs_A[:idx_obs_A], exp_X1_A[:idx_obs_A])
        le2_A.set_data(t_obs_A[:idx_obs_A], exp_X2_A[:idx_obs_A])

        # --- Sys B ---
        den_B = sim_B['frames'][idx]
        im_B.set_data(den_B)
        im_B.set_clim(vmin=0, vmax=np.max(den_B))
        lt_B.set_ydata(np.sum(den_B, axis=0) * sim_B['dx'])
        lr_B.set_xdata(np.sum(den_B, axis=1) * sim_B['dx'])

        # Plot up to the synchronized high-res index
        le1_B.set_data(t_obs_B[:idx_obs_B], exp_X1_B[:idx_obs_B])
        le2_B.set_data(t_obs_B[:idx_obs_B], exp_X2_B[:idx_obs_B])

        time_text.set_text(f"Time: {current_time:.1f} a.u.")

        return[im_A, lt_A, lr_A, le1_A, le2_A, im_B, lt_B, lr_B, le1_B, le2_B, time_text]

    ani = FuncAnimation(fig, update, frames=len(sim_A['frames']), blit=True)
    plt.close()
    return ani

def plot_position_energies_spectrum(sim_data):
    t = sim_data['time_obs'][:]
    E1 = sim_data['energy_1'][:]
    E2 = sim_data['energy_2'][:]
    C_t = sim_data['autocorr'][:]
    expX1 = sim_data['exp_X1'][:]
    expX2 = sim_data['exp_X2'][:]
    print(len(E1))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))
    # =========================================================
    # Plot 0: Local Coordinates Expectation Values <X1>, <X2>
    # =========================================================

    ax0.plot(t, expX1, label=r'$\langle X_1 \rangle$ (Bond 1)', color='deepskyblue', lw=1)
    ax0.plot(t, expX2, label=r'$\langle X_2 \rangle$ (Bond 2)', color='crimson', lw=1)
    ax0.set_title("<X1> and <X2> Oscillations: Local Energy Transfer")
    ax0.set_xlabel("Time (a.u.)")
    ax0.set_ylabel("<X>")
    ax0.legend()
    ax0.grid(alpha=0.3)

    # =========================================================
    # Plot 1: Local Energies (Perfect Rabi Oscillations)
    # =========================================================
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t, E1, label=r'$\langle E_1 \rangle$ (Bond 1)', color='deepskyblue', lw=1)
    ax1.plot(t, E2, label=r'$\langle E_2 \rangle$ (Bond 2)', color='crimson', lw=1)
    ax1.set_title("Rabi Oscillations: Local Energy Transfer")
    ax1.set_xlabel("Time (a.u.)")
    ax1.set_ylabel("Local Energy")
    ax1.legend()
    ax1.grid(alpha=0.3)


    # =========================================================
    # Plot 2: Absorption Spectrum (FFT of Autocorrelation)
    # =========================================================
    # Calculate the Fourier transform of the autocorrelation
    # Remove DC component
    C_t = C_t - np.mean(C_t)

    # Window to reduce leakage
    window = np.hanning(len(C_t))
    C_t = C_t * window

    dt = t[1] - t[0]

    # Calculate standard FFT
    freqs = np.fft.fftfreq(len(C_t), dt) * 2 * np.pi
    spectrum = np.abs(np.fft.fft(C_t))**2

    # Map mathematically negative frequencies to physical positive energies
    energies = -freqs

    # Sort the arrays so Matplotlib draws a clean, continuous line
    sort_idx = np.argsort(energies)
    energies_sorted = energies[sort_idx]
    spectrum_sorted = spectrum[sort_idx]

    # Plot the sorted, positive energies
    ax2.plot(energies_sorted, spectrum_sorted, color='purple', lw=1)

    # Peak Finding and Annotation 
    # Set a threshold to avoid labeling tiny numerical noise.
    # Here, we only look at peaks with at least 5% the intensity of the maximum peak.
    max_intensity = np.max(spectrum_sorted)
    peaks, properties = find_peaks(spectrum_sorted,
                                   height=max_intensity * 0.001,   # Ignore small noise
                                   prominence=max_intensity * 0.001) # Ignore shoulders
    # Loop through the found peaks and annotate them
    print("Peak energy  |  Intensity ")
    print("--------------------------")
    for p in peaks:
        peak_energy = energies_sorted[p]
        peak_intensity = spectrum_sorted[p]

        print(f"{peak_energy:.5f} a.u. | {peak_intensity:.2e}")


        # Only annotate if the peak is inside our viewing window
        if 0.5 <= peak_energy <= 30.0:
            # Draw a small red 'x' on the peak
            ax2.plot(peak_energy, peak_intensity, "rx", markersize=6)

            # Write the frequency text just above the peak
            ax2.annotate(f"{peak_energy:.3f}",
                         xy=(peak_energy, peak_intensity),
                         xytext=(0, 6),  # Offset the text 6 points vertically
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, color='black',
                         rotation=45) # Rotation prevents overlapping text if peaks are close

    ax2.set_title("Energy Spectrum (FFT of Autocorrelation)")
    ax2.set_xlabel("Energy (a.u.)")
    ax2.set_ylabel("Spectral Intensity")
    ax2.set_xlim(0.5, 20.0) # Now your peaks will appear right here!
    # Increase the upper y-limit slightly so our new text annotations don't get cut off
    ax2.set_ylim(0, max_intensity * 1.15)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrogram(sim_data, signal_key='autocorr'):
    """
    Computes a quantum spectrogram. Fixed to display Energy (Angular Frequency)
    and handle complex quantum arrays without artifacts.
    """
    time_array = sim_data['time_array'][:]
    signal_data = sim_data[signal_key][:]

    # Calculate sampling frequency
    dt_frame = time_array[1] - time_array[0]
    fs = 1.0 / dt_frame

    # Dynamic window size (longer window = sharper horizontal lines, but more time smearing)
#    nperseg = min(256, len(signal_data) // 4)
    nperseg = min(512, len(signal_data) // 2)

    if nperseg < 16:
        nperseg = len(signal_data) #// 2

    # 1. Compute Spectrogram using the RAW COMPLEX data
    # return_onesided=False is required by SciPy when passing complex arrays
    f, times_spec, Sxx = spectrogram(
        signal_data, # Removed np.real() to preserve quantum phase direction
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg - 2,
        return_onesided=False
    )

    # 2. Convert to Angular Frequency / Energy
    omega = -f * 2 * np.pi

    # 3. Filter out negative frequencies (we only want positive energies)
    pos_mask = omega > 0
    omega_pos = omega[pos_mask]
    Sxx_pos = Sxx[pos_mask, :]

    # Sort the frequencies just in case SciPy returns them out of order
    # pcolormesh requires the y-axis to be strictly increasing, so we sort it:
    sort_idx = np.argsort(omega_pos)
    omega_pos = omega_pos[sort_idx]
    Sxx_pos = Sxx_pos[sort_idx, :]

    # Scale times back to absolute simulation time
    times_spec = times_spec + time_array[0]

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_spec, omega_pos, 10 * np.log10(Sxx_pos + 1e-10), shading='gouraud', cmap='viridis')

    plt.title(f"Quantum Spectrogram of {signal_key}")
    plt.ylabel('Energy / Frequency $\\omega$ (a.u.)')
    plt.xlabel('Time (a.u.)')
    plt.colorbar(label='Intensity (dB)')

    # Set y-limit to 20 so it visually matches your 1D FFT plot!
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()


def load_soft_simulation(filename):
    """
    Loads a previously saved quantum simulation from disk.
    Returns the exact dictionary needed for the visualization functions.
    """
    print(f"Loading data from {filename}...")
    with np.load(filename, allow_pickle=True) as data:
        # Reconstruct the dictionary exactly as it was
        results = {key: data[key] for key in data.files}
    print("Data loaded successfully!")
    return results



class LazySlicedDataset:
    """Proxy class for lazy loading of HDF5 data."""
    def __init__(self, dataset, start, stop, step):
        self.ds = dataset
        self.start = start or 0
        self.step = step or 1
        self.stop = stop if stop is not None else dataset.shape[0]
        self.length = len(range(self.start, self.stop, self.step))

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            s_start = idx.start or 0
            s_stop = idx.stop if idx.stop is not None else self.length
            s_step = idx.step or 1
            real_start = self.start + (s_start * self.step)
            real_stop = self.start + (s_stop * self.step)
            real_step = self.step * s_step
            return self.ds[real_start:real_stop:real_step]
        else:
            return self.ds[self.start + (idx * self.step)]

def load_h5_simulation(filename, start=0, stop=None, step=1, in_memory=True):
    """
    Loads an HDF5 simulation trajectory.
    Automatically syncs the 1D observables slice to match the physical time 
    of the requested 2D frames slice.
    """
    h5f = h5py.File(filename, 'r')
    sim_data = {"_file": h5f}

    # 1. Determine the slice for the 2D frames
    total_frames_psi = h5f['time_frames'].shape[0]
    total_frames_obs = h5f['time_obs'].shape[0]
    
    stop_frame = stop if stop is not None else total_frames_psi
    frame_slice = slice(start, stop_frame, step)

    # 2. Extract the exact physical time limits from the sliced frames
    time_frames_full = h5f['time_frames'][:]
    sliced_time_frames = time_frames_full[frame_slice]
    sim_data['time_frames'] = sliced_time_frames
    
    if len(sliced_time_frames) > 0:
        start_time = sliced_time_frames[0]
        stop_time = sliced_time_frames[-1]
    else:
        start_time, stop_time = 0.0, 0.0

    # 3. Find the corresponding physical boundaries in the observables arrays
    time_obs_full = h5f['time_obs'][:]
    obs_start_idx = np.searchsorted(time_obs_full, start_time, side='left')
    obs_stop_idx = np.searchsorted(time_obs_full, stop_time, side='right')
    
    # We do NOT step the observables. We load all of them in the time window 
    # so the 1D plots stay incredibly smooth!
    obs_slice = slice(obs_start_idx, obs_stop_idx, 1)

    for key in h5f.keys():
        if key == 'time_frames': continue
            
        ds = h5f[key]
        
        # Identify dataset type by dimensionality
        is_frames = (len(ds.shape) == 3) # 3D Density Matrix
        is_obs = (len(ds.shape) == 1 and ds.shape[0] == total_frames_obs) # 1D Observables

        if is_frames:
            if in_memory:
                sim_data[key] = ds[frame_slice]
            else:
                sim_data[key] = LazySlicedDataset(ds, start, stop_frame, step)
                
        elif is_obs:
            if in_memory:
                sim_data[key] = ds[obs_slice]
            else:
                sim_data[key] = LazySlicedDataset(ds, obs_start_idx, obs_stop_idx, 1)
        else:
            # Static grids (x, X1, V, etc.)
            sim_data[key] = ds[:]

    for key in h5f.attrs.keys():
        sim_data[key] = h5f.attrs[key]

    return sim_data



from scipy.linalg import hankel, svd, pinv

def svd_harmonic_inversion(sim_data, signal_key='autocorr', L_ratio=0.33, svd_tol=1e-4, max_N=6000):
    """
    Extracts exact eigenenergies and spectral weights using SVD Harmonic Inversion.
    Automatically decimates ultra-long signals to prevent RAM overflow.
    """
    time_array = sim_data['time_obs'][:]
    C_t = sim_data[signal_key][:]

    # --- 1. Auto-Decimation (MEMORY FIX) ---
    N_orig = len(C_t)
    if N_orig > max_N:
        # Calculate stride step to bring N down to max_N
        step = int(np.ceil(N_orig / max_N))
        
        dt_orig = time_array[1] - time_array[0]
        dt_new = dt_orig * step
        nyquist_E = np.pi / dt_new
        
        print(f"  [HI] Signal too large for SVD ({N_orig} pts). Auto-decimating by factor of {step}...")
        print(f"  [HI] New dt: {dt_new:.4f} a.u. | Max Resolvable Energy (Nyquist): {nyquist_E:.2f} a.u.")
        
        time_array = time_array[::step]
        C_t = C_t[::step]
        
    dt_frame = time_array[1] - time_array[0]
    N = len(C_t)

    # 2. Pencil Parameter (Rows of the Hankel Matrix)
    L = int(N * L_ratio)

    # 3. Construct the Rectangular Hankel Matrix
    H = hankel(C_t[:L], C_t[L-1:])

    # 4. Perform the Singular Value Decomposition (Now runs in seconds instead of crashing!)
    print(f"[HI] Running SVD on {H.shape[0]}x{H.shape[1]} matrix...")
    U, S, Vh = svd(H, full_matrices=False)

    # 5. Truncate the Noise (Determine number of quantum states K)
    cutoff = S[0] * svd_tol
    K = np.sum(S > cutoff)
    print(f"[HI] SVD Filter extracted {K} true quantum states.")

    Uk = U[:, :K]

    # 6. The Shift Invariance Property
    U_down = Uk[:-1, :] # Drop the last row
    U_up = Uk[1:, :]    # Drop the first row

    # 7. Solve for the Transition Matrix M
    M = pinv(U_down) @ U_up

    # 8. Extract the Signal Poles (Eigenvalues of M)
    z_k = np.linalg.eigvals(M)

    # 9. Extract the Physical Energies
    E_k = -np.imag(np.log(z_k)) / dt_frame

    # 10. Extract Amplitudes (Spectral Weights) via Linear Least Squares
    n_array = np.arange(N)[:, None]
    Phi = z_k[None, :] ** n_array  

    # Solve Phi * d = C_t
    d_k_complex, _, _, _ = np.linalg.lstsq(Phi, C_t, rcond=None)

    # Physical probabilities must be real and positive
    weights = np.abs(d_k_complex)

    # Sort the states by Energy
    sort_idx = np.argsort(E_k)
    E_k = E_k[sort_idx]
    weights = weights[sort_idx]

    return E_k, weights, S

def plot_svd_extraction(sim_data, signal_key='autocorr', svd_tol=1e-4):
    """
    Plots the SVD-extracted energies as sharp peaks overlaid on top
    of the classical Fourier Transform spectrum.
    """
    # Get standard FFT data for the background
    C_t = sim_data[signal_key][:]
    t = sim_data['time_obs'][:]
    dt = t[1] - t[0]

    # Compute FFT
    spectrum = np.abs(np.fft.fft(C_t))**2
    freqs_math = np.fft.fftfreq(len(C_t), d=dt)
    energies_fft = -freqs_math * 2 * np.pi # Convert math frequency to Physics Energy

    # Filter positive energies
    pos_mask = energies_fft > 0
    e_fft_pos = energies_fft[pos_mask]
    spec_pos = spectrum[pos_mask]

    # --- Run the SVD Harmonic Inversion ---
    E_svd, weights_svd, singular_values = svd_harmonic_inversion(sim_data, signal_key, svd_tol=svd_tol)

    # Filter for positive physical energies
    valid_mask = (E_svd > 0) & (E_svd < 30) # Keep within viewing range
    E_svd = E_svd[valid_mask]
    weights_svd = weights_svd[valid_mask]

    # Normalize SVD weights to match FFT peak height for visual comparison
    weights_svd = weights_svd / np.max(weights_svd) * np.max(spec_pos)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: SVD Values
    ax1.semilogy(singular_values, 'o-', color='purple', markersize=4)
    ax1.axhline(singular_values[0] * svd_tol, color='r', linestyle='--', label='Noise Cutoff')
    ax1.set_title("SVD Matrix Singular Values")
    ax1.set_xlabel("Singular Value Index")
    ax1.set_ylabel("Magnitude (Log Scale)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Subplot 2: Energy Extraction
    ax2.plot(e_fft_pos, spec_pos, color='lightgray', lw=2, label="Classical FFT")
    ax2.stem(E_svd, weights_svd, linefmt='crimson', markerfmt='ro', basefmt=" ",
             label="SVD Extracted States")

    ax2.set_title("Classical FFT vs SVD Harmonic Inversion")
    ax2.set_xlabel("Energy (a.u.)")
    ax2.set_ylabel("Spectral Intensity")
    ax2.set_xlim(0, 20) # Zoom in on the relevant energy range
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print the exact numerical energies extracted
    print("\n--- Extracted Exact Eigenenergies ---")
    for i, (e, w) in enumerate(zip(E_svd, weights_svd)):
        if w > np.max(weights_svd)*0.01: # Only print states with non-negligible population
            print(f"State {i+1}: Energy = {e:.5f} a.u. (Relative Weight = {w/np.max(weights_svd):.3f})")


def plot_rabi_frequency(time_array, partial_energy, title="Energy Transfer (Rabi Oscillations)"):
    """
    Takes the Fourier transform of a real observable (like partial energy)
    to extract and plot the energy transfer / Rabi frequency.

    Returns the Matplotlib figure and the calculated Rabi frequency.
    """
    dt = time_array[1] - time_array[0]
    N = len(partial_energy)

    # 1. Pre-process the signal
    # Subtract the mean to remove the massive zero-frequency (DC) peak
    energy_centered = partial_energy - np.mean(partial_energy)

    # Apply a Hanning window to prevent edge artifacts (spectral leakage)
    window = np.hanning(N)
    energy_windowed = energy_centered * window

    # 2. Perform the Real FFT
    # Since expectation values are purely real, rfft is perfect here.
    fft_result = np.fft.rfft(energy_windowed)
    power_spectrum = np.abs(fft_result)**2

    # Calculate frequencies (multiplied by 2pi for energy in a.u.)
    freqs = np.fft.rfftfreq(N, d=dt) * 2 * np.pi

    # Find the dominant peak (The Rabi Frequency)
    peak_idx = np.argmax(power_spectrum)
    rabi_freq = freqs[peak_idx]

    # 3. Create the Dashboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # Panel 1: Time Domain (The Beating)
    ax1.plot(time_array, partial_energy, color='dodgerblue', lw=2)
    ax1.set_title("Expectation Value over Time")
    ax1.set_xlabel("Time (a.u.)")
    ax1.set_ylabel(r"$\langle E_1 \rangle$ (a.u.)")
    ax1.grid(alpha=0.3)

    # Panel 2: Frequency Domain (The FFT)
    ax2.plot(freqs, power_spectrum, color='crimson', lw=2)
    ax2.axvline(rabi_freq, color='black', linestyle='--', alpha=0.6,
                label=f"Rabi Freq: {rabi_freq:.4f} a.u.")

    ax2.set_title("Fourier Transform (Spectrum)")
    ax2.set_xlabel("Energy / Frequency (a.u.)")
    ax2.set_ylabel("Spectral Power")
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(alpha=0.3)

    # Zoom in intelligently around the peak
    # (Avoids showing high-frequency numerical noise)
    if rabi_freq > 0:
        ax2.set_xlim(0, rabi_freq * 4)

    plt.tight_layout()
    return fig, rabi_freq


if __name__ == "__main__":
    # Run the simulation with arguments passed from the shell
#    print("Running SOFT simulation...")
#    run_soft_simulation_hdf5_gpu(
#        steps=500000,
#        N=512,
#        d1=0.25,
#        d2=0.00,
#        frame_interval=100, 
#        obs_interval=1,    
#        save_filename="normal_traj_N512.h5",
#        dt=0.001, 
#    )
#    print("Finished SOFT simulation...")
##
#    print("Running SOFT simulation...")
#    run_soft_simulation_hdf5_gpu(
#        steps=2000000,
#        N=512,
#        d1=2.00,
#        d2=0.00,
#        frame_interval=400,  
#        obs_interval=4,
#        save_filename="local_traj_N512_d2.0.h5",
#        dt=0.00025, 
#    )
#    print("Finished SOFT simulation...")

#---------------------------------------
# LONG DYNAMICS
#---------------------------------------
#    print("Running SOFT simulation...")
#    run_soft_simulation_hdf5_gpu(
#        steps=20000000,
#        N=512,
#        d1=2.00,
#        d2=0.00,
#        frame_interval=4000,  
#        obs_interval=40,
#        save_filename="local_traj_N512_d2.0_long.h5",
#        dt=0.00025, 
#    )
#    print("Finished SOFT simulation...")
#---------------------------------------

    # 1. Open the file in read mode
    print("Opening file...")

    # Read a slice of the full trajectory and store it in memory.
#    sim_normal = load_h5_simulation("normal_traj_N512.h5",start=0,stop=500,step=1,in_memory=True)
#    sim_local = load_h5_simulation("local_traj_N512_d2.0.h5",start=0,stop=500,step=1,in_memory=True)

    # Read trajectory directly from disk frame by frame.
#    sim_normal = load_h5_simulation("normal_traj_N512.h5",start=0,stop=None,step=1,in_memory=False)
#    sim_local = load_h5_simulation("local_traj_N512_d2.0.h5",start=0,stop=None,step=1,in_memory=False)
    sim_local = load_h5_simulation("local_traj_N512_d2.0_long.h5",start=0,stop=None,step=1,in_memory=False)

    # 3. Plot the final spectrum (using the function from your original code)
    print("Plotting...")

    # Plot expectation values of position and partial energies and FT of C(t)
    #---------------------------------------------------------------------
#    plot_position_energies_spectrum(sim_normal)
#    plot_position_energies_spectrum(sim_local)
    #---------------------------------------------------------------------

    # Spectrogram
    #---------------------------------------------------------------------
    #plot_spectrogram(sim_data, signal_key='autocorr')
    #---------------------------------------------------------------------

    # Run the SVD analysis and plot it!
    #---------------------------------------------------------------------
#    plot_svd_extraction(sim_normal, signal_key='energy_1', svd_tol=1e-4)
    plot_svd_extraction(sim_local, signal_key='energy_1', svd_tol=1e-4)
    #---------------------------------------------------------------------

    # Create and animation of the quantum trajectory.
    #---------------------------------------------------------------------
#    animation = visualize_dual_comparison(sim_normal, sim_local)
    # Save it as a GIF (uses the 'pillow' library, which is pre-installed in Colab)
    #print("Writing GIF animation...")
    #animation.save('quantum_dynamics.gif', writer='pillow', fps=10, dpi=50)
#    print("Writing MP4 animation...")
#    animation.save('quantum_dynamics.mp4', writer='ffmpeg', fps=10, dpi=100)
    #---------------------------------------------------------------------


    # Rabi frequency analysis
    #---------------------------------------------------------------------
    # Assuming you loaded your data
#    time = sim_normal['time_array']
#    energy_1 = sim_normal['energy_1']  # Replace with your actual partial energy key

    # Generate the plot and get the exact frequency
#    fig, rabi_freq = plot_rabi_frequency(time, energy_1, title="Normal Mode Energy Transfer")

#    print(f"The energy splitting (Rabi frequency) is {rabi_freq:.5f} a.u.")

    # If you want to save it and push to your git repo for Colab:
#    fig.savefig('rabi_plot.png', dpi=150)

    # Remember to close it when you are totally done!
#    print("Closing trajectory files...")
#    sim_normal["_file"].close()
#    sim_local["_file"].close()

