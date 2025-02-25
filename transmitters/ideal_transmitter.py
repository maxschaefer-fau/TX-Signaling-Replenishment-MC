import numpy as np
from scipy.constants import Avogadro
from models.space import AbsorbingReceiver, TransparentReceiver
from utils.utils import get_conc_vol_for_practical

def ideal_transmitter(time_array: np.ndarray, rho_array: np.ndarray, config) -> dict:
    """
    Simulates the dynamics of an ideal transmitter releasing molecules over time.

    This function calculates the concentrations of molecules A and B 
    in the system as well as the number of molecules that interact with 
    the absorbing or transparent receiver based on the provided configurations.

    Parameters:
        time_array (np.ndarray): An array of time values for the simulation.
        rho_array (np.ndarray): A permeability vector affecting the diffusion rates.
        config (object): Configuration parameters containing various constants and properties of the nanomachine.

    Returns:
        dict: A dictionary containing:
            - NinA (np.ndarray): Number of A molecules inside the transmitter over time.
            - NinB (np.ndarray): Number of B molecules inside the transmitter over time.
            - NoutB (np.ndarray): Number of B molecules outside the transmitter over time.
            - Nrec (np.ndarray): Received molecules at the receiver over time.
    """

    # Calculate Volume and Surface Area of the Transmitter
    vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(config.r_tx, config.r_out)
    A = 4 * np.pi * config.r_tx**2  # Surface area of the transmitter
    rho_array *= A  # Scale permeability by surface area

    # Initial Concentrations
    CinA0 = 0  # Initial concentration of A inside the transmitter
    CinB0 = 0  # Initial concentration of B inside the transmitter
    CoutB0 = 0  # Initial concentration of B outside the transmitter
    NAout = 1e16  # Total number of A molecules outside the transmitter
    CoutA0 = NAout / Avogadro / vol_out  # Concentration of A outside

    # Initialization of solution vectors for concentration values over time
    CinA = np.zeros(len(time_array))  # Concentration of A inside over time
    CinA[0] = CinA0

    CinB = np.zeros(len(time_array))  # Concentration of B inside over time
    CinB[0] = CinB0

    CoutB = np.zeros(len(time_array))  # Concentration of B outside over time
    CoutB[0] = CoutB0

    # Eigenvalue calculations
    lAin = -(rho_array / vol_in + config.kab)  # Decay of A due to absorption and reaction
    lBin = -rho_array / vol_in  # Decay of B inside the transmitter
    lBout = rho_array / vol_out  # Increase of B outside the transmitter

    # Time-stepping loop to calculate concentrations
    for k in range(1, len(time_array)):
        # Update concentration of A inside the transmitter
        CinA[k] = np.exp(lAin[k] * config.step_time) * CinA[k-1] + config.step_time * rho_array[k] / vol_in * CoutA0

        # Update concentration of B inside the transmitter
        CinB[k] = np.exp(lBin[k] * config.step_time) * CinB[k-1] + config.step_time * config.kab * CinA[k]

        # Update concentration of B outside the transmitter
        CoutB[k] = CoutB[k-1] + lBout[k] * config.step_time * CinB[k]

    # Convert concentrations to molecule counts
    NinA = CinA * vol_in * Avogadro  # Number of A molecules inside
    NinB = CinB * vol_in * Avogadro  # Number of B molecules inside
    NoutB = CoutB * vol_out * Avogadro  # Number of B molecules outside

    # Instantiate the appropriate receiver class
    if config.receiver_type == 'AbsorbingReceiver':
        rec = AbsorbingReceiver(config.r_rx)
    elif config.receiver_type == 'TransparentReceiver':
        rec = TransparentReceiver(config.r_rx)

    # Calculate the average hits received by the receiver
    nr = rec.average_hits(time_array, NoutB, config.r_tx, config.D_space, config.dist)

    # Calculate the number of received molecules scaled by step time
    Nrec = nr[:len(time_array)] * config.step_time

    return {
            'NinA': NinA,
            'NinB': NinB,
            'NoutB': NoutB,
            'Nrec': Nrec
            }
