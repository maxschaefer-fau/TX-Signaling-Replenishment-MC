"""
author: JDRanpariya
credit: Mexy
copyright: IDC

Ts: Change this variable to increase or decrease the amount of time
kab: Rate constant for the reaction
"""

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


constants = {
        'RT': 1e-4,       # Rate Constant
        'Na': 6.022e23,   # Avogadro's Number
        'T' : 1e-4        # Time Step Constant 
        }

geometrical_params = {
        'rtx' : 80e-9,    # Nano Particle Radius
        'rrx' : 1e-6,     # Reciever Radius 
        'r0'  : 1e-3,     # Radius of Surrounding Volumn
        'l'   : 2e-6,     # Tx-Rx Distance 
        'D'   : 2.6e-12,  # Diffusion Constant 
        }

Ts = 30 # Time in seconds
t = np.arange(0,Ts, constants['T'])  # t = 0:T:500-T; 

# Calculation of Volumn and Surface Area of Tx
Vin = 4/3 * np.pi * geometrical_params['rtx']**3
A = 4 * np.pi * geometrical_params['rtx']**2
Vout = 4/3 * np.pi * geometrical_params['r0']**3 

# Initial Concentrations 
CinA0 = 0 
CinB0 = 0 
CoutB0 = 0 

NAout = 1e16 # Number of A molecules outside
CoutA0 = NAout/constants['Na']/Vout # Concentration of A outside

# permeability and reaction rates
rho0 = 7.33e-10*A # connect to area of the NP surface 
kab = 1e-1 # !!!! value has to be determined !!! 

# Initialization of solution vectors
CinA = np.zeros(len(t)) 
CinA[0] = CinA0 

CinB = np.zeros(len(t)) 
CinB[0] = CinB0 

CoutB = np.zeros(len(t)) 
CoutB[0] = CoutB0

# time dependent permeability 
rho = np.zeros(len(t))

# Switching Permeabilty | Use Ts = 10 for base case
step = Ts // 5 # Get the number of time blocks needed
for i in range(Ts):
    if i % 2 != 0:
        rho[int(len(t) * (i-1)/step):int(len(t) * i/step)] = rho0

# Eigen value calculation
lAin  = -(rho / Vin + kab) 
lBin  = -rho / Vin 
lBout = rho / Vout


#tmp = np.zeros(len(t))
#tmp[0] = CoutA0

for k in range(1, len(t)):
    CinA[k]  = np.exp(lAin[k] * constants['T']) * CinA[k-1] + constants['T'] * rho[k] / Vin * CoutA0
    CinB[k]  = np.exp(lBin[k] * constants['T']) * CinB[k-1] + constants['T'] * kab * CinA[k] 
    CoutB[k] = CoutB[k-1] + lBout[k] * constants['T'] * CinB[k]

# convert to molecules 
NinA = CinA * Vin * constants['Na']
NinB = CinB * Vin * constants['Na']
NoutB = CoutB * Vout * constants['Na'] 

# Received molecules 
b1 = ((geometrical_params['rtx'] + geometrical_params['rrx']) * (geometrical_params['rtx'] + geometrical_params['rrx'] - 2 * geometrical_params['l']) + geometrical_params['l']**2) / (4 * geometrical_params['D'])
b2 = ((geometrical_params['rtx'] - geometrical_params['rrx']) * (geometrical_params['rtx'] - geometrical_params['rrx'] + 2 * geometrical_params['l']) + geometrical_params['l']**2) / (4 * geometrical_params['D'])
ro = 1 / A
pt = (2 * ro * geometrical_params['rtx'] * geometrical_params['rrx'] / geometrical_params['l']) * np.sqrt(np.pi * geometrical_params['D'] / t) * (np.exp(-b1 / t) - np.exp(-b2 / t))
print(f"tyxpe of pt is {type(pt)}, and lenth is {len(pt)}")
pt = np.nan_to_num(pt)

print(f"Debug: pt is {pt}") # It's NAN due to some error in calculating pt

nr = fftconvolve(CoutB * Vout * constants['Na'], pt, mode='full')
Nrec = nr[:len(t)] * constants['T'] 
print(f"Debug: Nrec is {Nrec}") # It's NAN due to some error in calculating pt

# Plot Number of Molecues of Type A and B inside the Transmitter(Tx) and Number of Type B Molecules outside
plt.figure(1)
plt.grid(True)
plt.plot(t, NinA)
plt.plot(t, NinB)
plt.plot(t, NoutB)
plt.plot(t, Nrec)
plt.xlabel('Time')
plt.ylabel('# Molecules')
plt.legend()
plt.show()
