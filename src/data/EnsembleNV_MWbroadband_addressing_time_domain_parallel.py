# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:47:23 2024

@author: Ahmadiafsa
"""
#==============================================================================
import numpy as np
import random
import scipy.constants as cte
from scipy.integrate import trapz
from multiprocessing import Pool, cpu_count
#==============================================================================

Type = 'EnsembleNV'

SNR = 100 #signal-to-noise ratio in dB

#determine the magnetic field strength randomly
B_start = 0
B_stop = 2
step_sizes = [1, 0.1, 0.01, 1e-5]

random_selections = []
for step in step_sizes:
    series = np.arange(B_start, B_stop + step, step)  
    random_choice = random.choice(series)
    data_random = [(step, random_choice)]
    random_selections.append(data_random)
B_matrix = np.concatenate(random_selections, axis = 0)
B = B_matrix[:,1]
# print(B)

theta_B = np.pi/6.
phi_B = np.pi/3.  

theta_MW = np.pi/4.
phi_MW = np.pi/4.

# raise RuntimeError

#======================= Constants =========================================
# NV fine and hyperfine constants (in MHz)
D_0 = 2.87e3 #zero-field splitting 
Apar = -2.14
Aperp = -2.7
PQ = -4.96

# Magnetic coupling constants (in SI units)
muB = 9.274e-24
gNV = 2.0028
muN = 5.051e-27
gN = 0.404
h = 6.626e-34

# Gyromagnetic ratios (in MHz/G)
gammaNV = muB * gNV / h / 1e10 # NV gyromagnetic ratio 
gammaN = muN * gN / h / 1e10 # N gyromagnetic ratio 

# Pauli matrices
S_x = 1 / np.sqrt(2) * np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])
S_y = 1 / np.sqrt(2) * 1j * np.array([[0, 1, 0],
                                      [-1, 0, 1],
                                      [0, -1, 0]])
S_z = np.array([[1, 0, 0],
                [0, 0, 0], 
                [0, 0, -1]])
SI = np.eye(3)


S_zfs = np.dot(S_z, S_z) - 2/3 * SI # Matrix useful for definition of Hamiltonian

#MW range
n_freq = 20000 # The number of generated frequency
freq_i = 2370 #MHz
freq_f = 3370 #MHz

Linewidth = 1.
MWfreq = np.linspace(freq_i, freq_f, n_freq)

#constant for linear propagation calculation
c = cte.c #Speed of light (m/s)
f_0 = 2.87 * 1e9 #central frequency of the light (Hz)
lambda_0 = c / f_0 #wavlenghth of the light (m)
k_0 = 2 * np.pi / lambda_0
length = 50 * 1e-3 #length of a sample (m)
n = 2.41 #diamond optical refrective index

fwhm = 85 * 1e6  #Hz
sigma = fwhm / 2 * np.sqrt(np.log(2)) 

dt = 1 / (freq_f - freq_i) #time step (s)
time_range = dt * len(MWfreq) #duration of time domain signal
time = np.arange(0, time_range , dt) 
# Time = time[(time >= 8) & (time <= 12)]
Time = time[(time >= 9.98) & (time <= 10.18)]

#=========================== General functions ================================

def get_vector_cartesian(A, theta, phi):
    """ 
    Compute cartesian coordinates of a vector from its spherical coordinates:
    norm A, polar angle theta, azimutal angle phi
    """
    A_para = A * np.cos(theta)
    A_perp = A * np.sin(theta)
    vec = np.array([A_perp * np.cos(phi), 
                    A_perp * np.sin(phi),
                    A_para])
    return vec
#
# Transformation between lab frame and NV frames, cartesian coordinates
def get_rotation_matrix(idx_nv):
    """ Returns the transformation matrix from lab frame to the desired 
    NV frame, identified by idx_nv (can be 1, 2, 3 or 4) """
    if idx_nv==1:
        RNV = np.array([[1/np.sqrt(6), 1/np.sqrt(6), 2/np.sqrt(6)],
                        [1/np.sqrt(2),-1/np.sqrt(2), 0],
                        [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)]])

    elif idx_nv==2:
         RNV = np.array([[1/np.sqrt(6), -1/np.sqrt(6),  -2/np.sqrt(6)],
                         [1/np.sqrt(2),  1/np.sqrt(2),  0],
                         [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)]])
        
    elif idx_nv==3:
         RNV = np.array([[-1/np.sqrt(6), -1/np.sqrt(6),  2/np.sqrt(6)],
                         [-1/np.sqrt(2),  1/np.sqrt(2),  0],
                         [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    elif idx_nv==4:
         RNV = np.array([[-1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
                         [-1/np.sqrt(2), -1/np.sqrt(2), 0],
                         [-1/np.sqrt(3), 1/np.sqrt(3),  1/np.sqrt(3)]])
    else:
        raise ValueError('Invalid index of NV orientation')
    
    return RNV
#
def transform_vector_lab_to_NV_frame(vec_in_lab, nv_idx=1):
    """ Vector coordinates transformation from lab frame to desired NV frame.
    nv_idx can be 1, 2, 3 or 4 """
    RNV = get_rotation_matrix(nv_idx)
    vec_in_nv = np.dot(RNV, vec_in_lab)
    return vec_in_nv
#
def transform_all_frames(B0, theta, phi):
    """ 
    Compute cartesian coordinates of a vecotr in all 4 NV frames, 
    based on its spherical coordinates in lab frame
    """
    Bvec = get_vector_cartesian(B0, theta, phi)
        
    # Concise version
    Bvec_list = [transform_vector_lab_to_NV_frame(Bvec, idx)
                 for idx in range(1, 5)]
    
    return Bvec_list

#======================== Ground state Hamiltonian ============================

def NV_transitionsElevels(B):
    # Input: magnetic field, defined in NV center frame
    # Output: This function diagonalizes the Hamiltonian Hgs and return 9 eignenergies and its vector

    # Hamiltonian
    # Fine and hyperfine terms
    HZFS = D_0 * np.kron(S_zfs, SI) # Zero-field splitting
    HHFPar = Apar * np.kron(S_z, S_z) # Axial hyperfine interaction
    HHFPerp = Aperp * (np.kron(S_x, S_x) + np.kron(S_y, S_y)) # Non-axial hyperfine interaction
    HNucQ = PQ * np.kron(SI, S_zfs) # Nuclear quadrupole interaction

    # Magnetic field coupling terms
    HBEl = gammaNV * np.kron(B[0]*S_x + B[1]*S_y + B[2]*S_z, SI) # Electric Zeeman coupling
    HBNuc = gammaN * np.kron(SI, B[0]*S_x + B[1]*S_y + B[2]*S_z) # Nuclear Zeeman coupling corresponding to 14N

  
    H_total = HZFS + HBEl + HBNuc + HHFPar + HHFPerp + HNucQ
    E_I, vec_I = np.linalg.eigh(H_total)

    return E_I, vec_I

def NV_GS_Hamiltonian_MWprobe(Bmw):
    # Compute interaction Hamiltonian, with MW vector Bmw defined in NV center frame

    # Magnetic field coupling terms
    HintEl = gammaNV * np.kron(Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z, SI) # To electric spin
    HintNuc = gammaN * np.kron(SI, Bmw[0]*S_x + Bmw[1]*S_y + Bmw[2]*S_z) # To nuclear spin

    # Total interation Hamiltonian
    Hint = HintEl + HintNuc
    return Hint

#======================== Computation of ODMR spectrum ========================

def lorentzian(x, x0, fwhm):
    return 1 / (1 + (x - x0)**2 / (fwhm / 2)**2)

def ESR_singleNV(MWfreq, MWvec, Bvec, Linewidth):
    # All vectors are defined in NV frame
    nMW = len(MWfreq) # Number of frequency points
    Tstrength = np.zeros(nMW) # transition strength

    E_I, vec_I = NV_transitionsElevels(Bvec) # Eigenenergies and eigenvectors
    Hint = NV_GS_Hamiltonian_MWprobe(MWvec) # Interaction Hamiltonian

    # Calculate transition strengths
    for initS in np.arange(9): # Sweep over all initial states
        initFreq = E_I[initS] # frequency
        initVec = vec_I[:,initS]

        for finS in np.arange(initS, 9): # Sweep over all final states
            finFreq = E_I[finS] # frequency
            finVec = vec_I[:,finS] # state

            # Transition matrix element and transition amplitude
            TME = np.dot(np.dot(np.conj(finVec.transpose()),Hint), initVec)
            TA = np.abs(TME)**2
            
            # Add lorentzian lineshape
            TS = TA * lorentzian(MWfreq, abs(finFreq - initFreq), Linewidth)
            
            Tstrength += TS
                
    return Tstrength
#
def ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth):
    # All vectors are defined in lab frame (spherical coordinates)
    nMW = len(MWfreq) # Number of frequency points
    Tstrength = np.zeros(nMW) # transition strength
    
    Bvector_list = transform_all_frames(B0, thetaB, phiB)
    MWvector_list = transform_all_frames(1, thetaMW, phiMW)
    
    for MWvec, Bvec in zip(MWvector_list, Bvector_list):
        Tstrength += ESR_singleNV(MWfreq, MWvec, Bvec, Linewidth)
        
    n_NV = len(Bvector_list) # number of NV orientations in ensemble
    return Tstrength / n_NV

#================ Computation of response function ============================

def dispersion_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth):
    freq_interval = (max(MWfreq) - min(MWfreq)) / (len(MWfreq))
    I1 = np.zeros(len(MWfreq))
    I2 = np.zeros(len(MWfreq))
    absorption = ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth)
    
    for i in range(len(MWfreq)):
        tempAbsorb1 = absorption[0: i]
        tempAbsorb2 = absorption[i+1: len(absorption)]
        temp_omeg1 = MWfreq[0: i]
        temp_omeg2 = MWfreq[i+1: len(MWfreq)]
        mult1 = 1 / (temp_omeg1 - MWfreq[i])
        mult2 = 1 / (temp_omeg2 - MWfreq[i])
        I2[i] = trapz(mult1 * tempAbsorb1) + trapz(mult2 * tempAbsorb2)
        I1[i] = trapz(absorption / (MWfreq + MWfreq[i]))

    chi1 = freq_interval * (I1 + I2) / np.pi 
    return chi1
#
def absorption_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth):
    freq_interval = (max(MWfreq) - min(MWfreq)) / (len(MWfreq))
    I1 = np.zeros(len(MWfreq))
    I2 = np.zeros(len(MWfreq))
    chi1 = dispersion_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth)
    for i in range(len(MWfreq)):
        tempchi11 = chi1[0: i]
        tempchi12 = chi1[i+1: len(chi1)]
        temp_omeg1 = MWfreq[0: i]
        temp_omeg2 = MWfreq[i+1:len(MWfreq)]
        mult1 = 1 / (temp_omeg1 - MWfreq[i])
        mult2 = 1 / (temp_omeg2 - MWfreq[i])
        I2[i] = trapz(mult1 * tempchi11) + trapz(mult2 * tempchi12)
        I1[i] = trapz(chi1 * 1 / (MWfreq + MWfreq[i]))

    chi2 = freq_interval * (I1 - I2) / np.pi #Absorption 
    return chi2
#
def response_function_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth):
    chi1 = dispersion_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth)
    chi2 = ESR_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth)
    
    chi = chi1 + 1j * chi2 #full responce function 
    return chi

#=================Computation of Linear propagation============================

def gaussian(x, x0, sig):
    """Broad band MW field has a Gaussian shape""" 
    return (1 / (sig * np.sqrt(2 * np.pi))) * np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.)))
#
def time_series_signal_NVensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB,  Linewidth):
    E_0_omega = gaussian(MWfreq * 1e6,f_0, sigma) #input field = E_0 * exp(ik.r) @ L = 0
    E_0_time = np.fft.fftshift(np.fft.fft(E_0_omega)) #F.F.T of the input field 
    
    chi = response_function_ensemble(MWfreq, thetaMW, phiMW, B0, thetaB, phiB, Linewidth)
    E_L_omega = E_0_omega * np.exp(1j * length * (n * 2 * np.pi * MWfreq / c + k_0 * chi/2))
    
    #F.F.T of the output field at L = length in the response of the liner chi
    E_L_time = np.fft.fftshift(np.fft.fft(E_L_omega))
    intensity_L_time = np.abs(E_L_time) ** 2 / max(E_0_time) ** 2
    # filtered_intensity = intensity_L_time[(time >= 8) & (time <= 12)]
    filtered_intensity = intensity_L_time[(time >= 9.98) & (time <= 10.18)]

    return filtered_intensity

#=========================== Gaussian noise added==============================

def awgn(signal, desired_snr):
    """
    Add Gaussian to the input signal to achieve the desired SNR level.
    """
    #np.random.seed(64)
    
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10**(desired_snr / 10)) #in linear factor

    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    noisy_signal = signal + noise
    
    return noisy_signal

#=========================== Calculation ======================================

# def intensity_time_noisy(B):
#
#     I_time_clean = time_series_signal_NVensemble(MWfreq, theta_MW, phi_MW, B, theta_B, phi_B,  Linewidth)
#     I_time_noisy = awgn(I_time_clean, SNR)
#
#     #save data
#     title = "time(\mus) intensity(a.u.)"
#
#     DATA = np.ones((len(Time), 2))
#     DATA[:, 0] = Time
#     DATA[:, 1] = I_time_noisy
#
#     # filename = f"../Data_analyzed/EnsembleNV_MWbroadband_signal100dB_time_domain_{B:.7f}G.dat"
#     filename = f"EnsembleNV_MWbroadband_signal100dB_time_domain_{B:.7f}G.dat"
#     np.savetxt(filename, DATA, fmt='%.17g', delimiter='\t ', header=title, comments='#')
#
#     return filename
#
#
# def parallel_intensity_time_noisy(B_values):
#     num_cores = cpu_count()
#     with Pool(num_cores) as pool:
#         filenames = pool.map(intensity_time_noisy, B_values)
#     return filenames

def intensity_time_noisy(B):
    I_time_clean = time_series_signal_NVensemble(MWfreq, theta_MW, phi_MW, B, theta_B, phi_B, Linewidth)
    I_time_noisy = awgn(I_time_clean, SNR)

    DATA = np.ones((len(Time), 2))
    DATA[:, 0] = Time
    DATA[:, 1] = I_time_noisy

    return DATA


def parallel_intensity_time_noisy(B_values):
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        data_arrays = pool.map(intensity_time_noisy, B_values)
    return data_arrays

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    B_values = [0, 1, 2]  # Example B values
    data_arrays = parallel_intensity_time_noisy(B_values)
    # Now data_arrays is a list of arrays with simulated data for each B value

    data_test = data_arrays[2]
    t = data_test[:, 0]
    y = data_test[:, 1]

    plt.plot(t, y)
    # plt.ylim([-0.02, 0.51])
    # plt.xlim([9.98, 10.18])
