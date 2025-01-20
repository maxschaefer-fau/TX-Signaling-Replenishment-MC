"""
Ts: Change this variable to increase or decrease the amount of time
kab: Rate constant for the reaction
"""

import numpy as np
from scipy.constants import Avogadro
from models.space import AbsorbingReceiver, TransparentReceiver
from utils import get_conc_vol_for_practical

def ideal_transmitter(time_array, rho_array, config):
    '''
    time_array: Time array of simulation
    rho_array: Permibiality Vector
    config: Configuration Parameters
    return: [NinA, NinB, NoutB, Nrec] 
    '''

    # Calculation of Volumn and Surface Area of Tx
    vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(config.r_tx, config.r_out)
    A = 4 * np.pi * config.r_tx**2
    rho_array *= A
    
    # Initial Concentrations
    CinA0 = 0
    CinB0 = 0
    CoutB0 = 0
    NAout = 1e16  # Number of A molecules outside
    CoutA0 = NAout/Avogadro/vol_out  # Concentration of A outside
    
    # Initialization of solution vectors
    CinA = np.zeros(len(time_array))
    CinA[0] = CinA0
    
    CinB = np.zeros(len(time_array))
    CinB[0] = CinB0
    
    CoutB = np.zeros(len(time_array))
    CoutB[0] = CoutB0
    
    # Eigen value calculation
    lAin = -(rho_array / vol_in + config.kab)
    lBin = -rho_array / vol_in
    lBout = rho_array / vol_out
    
    for k in range(1, len(time_array)):
        CinA[k] = np.exp(lAin[k] * config.step_time) * CinA[k-1] + config.step_time * rho_array[k]  / vol_in * CoutA0
        CinB[k] = np.exp(lBin[k] * config.step_time) * CinB[k-1] + config.step_time * config.kab * CinA[k]
        CoutB[k] = CoutB[k-1] + lBout[k] * config.step_time * CinB[k]
    
    # convert to molecules
    NinA = CinA * vol_in * Avogadro
    NinB = CinB * vol_in * Avogadro
    NoutB = CoutB * vol_out * Avogadro
    
    # Received molecules
    rec = AbsorbingReceiver(config.r_rx)
    if config.receiver_type == 'AbsorbingReceiver':
        pass
    elif config.receiver_type == 'TransparentReceiver':
        rec = TransparentReceiver(config.r_rx)
    
    nr = rec.average_hits(time_array, NoutB, config.r_tx, config.D_space, config.dist)
    Nrec = nr[:len(time_array)] * config.step_time

    return  {
    'NinA': NinA,
    'NinB': NinB,
    'NoutB': NoutB,
    'Nrec': Nrec
}
