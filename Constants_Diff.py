from Models.Space import Space, Particle
import numpy as np
from pathlib import Path

# Nanomachine properties
r_tx = 80e-9 # Transmitter Radius
r_rx = 1e-6 # Receiver Radius

# Diffusion Settings
p = 7.33e-10 # Membrane Permeability coefficient
p_close = 0     # Membrane closing time in seconds
D = p * r_tx # Membrane Diffusion coefficient
D_space = 2.6e-12 # Diffusion Coefficient in the space between TX and RX

# Environment Properties
l = 2e-6 # Distance between the TX and RX centers
k_d = 0.0 # Degradation rate on the space between TX and RX
r_out = 1e-3 # The environment radius, practically infinite

# Reaction Rate Constants
k1 = 3.21e3         # From E and R to ER
k_1 = 3948          # From ER to E and R
k2 = 889            # From ER to ES
#k_2 = 693           # From ES to ER
k3 = 3896           # From ES to E and S
k_3 = 4.46e3        # From E and S to ES
k_2 = k1*k2*k3/k_1/k_3

# Molecule counts on the inside
vol_in = (4*np.pi*r_tx*r_tx*r_tx)/3
conc_in = Space({
    'R': Particle(0.0, True, volume=vol_in),
    'S': Particle(0.0, True, volume=vol_in),
    'MR': Particle(2.0, False, volume=vol_in),
    'ER': Particle(0.0, False, volume=vol_in),
    'ES': Particle(0.0, False, volume=vol_in)
}, area=4*np.pi*r_tx*r_tx, volume=vol_in)

# Molecule counts on the outside
vol_out = (4*np.pi*r_out*r_out*r_out)/3
conc_out = Space({
    'R': Particle(1e16, True, volume=vol_out),
    'S': Particle(0.0, True, volume=vol_out),
    'MR': Particle(0.0, False, volume=vol_out),
    'ER': Particle(0.0, False, volume=vol_out),
    'ES': Particle(0.0, False, volume=vol_out)
}, area=4*np.pi*r_tx*r_tx, volume=vol_out)

# Simulation Parameters
step_count = int(1e5)
simulation_end = 30                   # seconds
step_time = simulation_end / step_count # seconds
steps_in_a_unit = step_count / simulation_end

# States
release_count = 2
end_part = 1
state_break = end_part / (2 * release_count + 1)
state_breaks = np.arange(state_break, end_part + state_break/2.0, state_break)
# state_breaks = [0.05, 0.2, 0.25]
# To be used to get external states
state_path = Path(__file__).parent / 'switching-patterns' / 'perm_state_data_multiple_fig6.csv'
