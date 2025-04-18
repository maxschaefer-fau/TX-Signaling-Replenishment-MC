from models.space import Space, Particle, AbsorbingReceiver
import numpy as np
import os
from pathlib import Path

# Nanomachine properties
r_tx = 80e-9        # Transmitter Radius
r_rx = 1e-6         # Receiver Radius

# Diffusion settings
p = 7.33e-10        # Membrane Permeability coefficient
p_close = 0         # Membrane closing time in seconds
D = p * r_tx        # Membrane Diffusion coefficient
D_space = 2.6e-12   # Diffusion Coefficient in the space between TX and RX

# Environment Properties
dist = 2e-6            # Distance between the TX and RX centers
k_d =  1e2          # Degradation rate on the space between TX and RX
r_out = 1e-3        # The environment radius, practically infinite

# Reaction Rate Constants
kab = 1e-1          # For Ideal Transmitter
k1 = 3.21e3         # From E and R to ER
k_1 = 3948          # From ER to E and R
k2 = 889            # From ER to ES
# k_2 = 693          # From ES to ER
k3 = 3896           # From ES to E and S
k_3 = 4.46e3        # From E and S to ES
k_2 = k1*k2*k3/k_1/k_3

# Molecule counts on the inside
vol_in = (4*np.pi*r_tx*r_tx*r_tx)/3

conc_in_ideal = Space({
    'A': Particle(0.0, True, volume=vol_in),
    'B': Particle(0.0, True, volume=vol_in),
}, area=4*np.pi*r_tx*r_tx, volume=vol_in)

conc_in = Space({
    'R': Particle(0.0, True, volume=vol_in),
    'S': Particle(0.0, True, volume=vol_in),
    'MR': Particle(2.0, False, volume=vol_in),
    'ER': Particle(0.0, False, volume=vol_in),
    'ES': Particle(0.0, False, volume=vol_in)
}, area=4*np.pi*r_tx*r_tx, volume=vol_in)

# Molecule counts on the outside
# We are considering Env a sphear including vol_in
vol_out = (4*np.pi*r_out*r_out*r_out)/3

conc_out_ideal = Space({
    'A': Particle(1e16, True, volume=vol_out),
    'B': Particle(0.0, True, volume=vol_out),
}, area=4*np.pi*r_tx*r_tx, volume=vol_out)

conc_out = Space({
    'R': Particle(1e16, True, volume=vol_out),
    'S': Particle(0.0, True, volume=vol_out),
    'MR': Particle(0.0, False, volume=vol_out),
    'ER': Particle(0.0, False, volume=vol_out),
    'ES': Particle(0.0, False, volume=vol_out)
}, area=4*np.pi*r_tx*r_tx, volume=vol_out)

# To be used to get external states
state_path = Path(__file__).parent / 'switching_patterns' / 'perm_state_data_multiple_fig6.csv'

# Simulation Parameters
receiver_type = 'AbsorbingReceiver' # Can be from ['AbsorbingReceiver', 'TransparentReceiver']
step_count = int(1e5)
simulation_end = 10                      # seconds
if os.path.isfile(state_path):
    with open(state_path,"r") as f:
        Ts = len(f.readlines())
        step_count = Ts                 # seconds
    f.close()

step_time = simulation_end / step_count  # seconds
steps_in_a_unit = step_count / simulation_end

# States Control Switchability
release_count = 0  # Number of times to swtich permibiality
end_part = 1  # fraction of simulation time to use for switching

# Time Array
time_array = np.linspace(0, simulation_end, int(simulation_end / step_time), endpoint=False)

# Save data
save_data = False
