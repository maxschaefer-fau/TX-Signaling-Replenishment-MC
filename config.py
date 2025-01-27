from pathlib import Path


class NanomachineConfig:
    def __init__(self):
        # Nanomachine properties
        self.r_tx = 80e-9        # Transmitter Radius
        self.r_rx = 1e-6         # Receiver Radius

        # Diffusion settings
        self.p = 7.33e-10        # Membrane Permeability coefficient
        self.p_close = 0         # Membrane closing time in seconds
        self.D = self.p * self.r_tx  # Membrane Diffusion coefficient
        self.D_space = 2.6e-12   # Diffusion Coefficient in the space between TX and RX

        # Environment Properties
        self.dist = 2e-6            # Distance between the TX and RX centers
        self.k_d = 0         # Degradation rate on the space between TX and RX
        self.r_out = 1e-3        # The environment radius, practically infinite

        # Reaction Rate Constants
        self.kab = 1e-1          # For Ideal Transmitter
        self.k1 = 3.21e3         # From E and R to ER
        self.k_1 = 3948          # From ER to E and R
        self.k2 = 889            # From ER to ES
        self.k3 = 3896           # From ES to E and S
        self.k_3 = 4.46e3        # From E and S to ES
        self.k_2 = self.k1 * self.k2 * self.k3 / self.k_1 / self.k_3  # Calculated

        # Simulation Parameters
        self.receiver_type = 'AbsorbingReceiver'
        self.step_count = int(1e5)
        self._simulation_end = 10
        #self.step_time = self.simulation_end / self.step_count
        #self.steps_in_a_unit = self.step_count / self.simulation_end

        # Point Transmitter Properties
        self.N = 1e16 # Number of molecules to release

        # Saving and Ploting
        self.save = True
        self.output_folder = Path(__file__).parent / 'output'
        self.plot = True

    @property
    def simulation_end(self):
        return self._simulation_end

    @simulation_end.setter
    def simulation_end(self, value):
        self._simulation_end = value

    @property
    def step_time(self):
        return self.simulation_end / self.step_count  # Calculated property

    @property
    def steps_in_a_unit(self):
        return self.step_count / self.simulation_end  # Calculated property

    def display_config(self):
        ''' Print the configuration for verification. '''
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")



