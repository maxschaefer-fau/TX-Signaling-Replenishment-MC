from pathlib import Path

class NanomachineConfig:
    """
    Configuration class for a nanomachine simulation, capturing various properties 
    related to the transmitter, receiver, diffusion, and reaction parameters.
    """

    # Constants defining settings and properties
    DEFAULT_NUM_MOLECULES = 100  # Default number of molecules to release

    def __init__(self):
        # Nanomachine properties
        self.r_tx: float = 80e-9          # Transmitter Radius in meters
        self.r_rx: float = 1e-6           # Receiver Radius in meters

        # Diffusion settings
        self.p: float = 7.33e-10          # Membrane Permeability coefficient
        self.p_close: float = 0           # Membrane closing time in seconds
        self.D: float = self.p * self.r_tx # Membrane Diffusion coefficient
        self.D_space: float = 2.6e-12     # Diffusion Coefficient between TX and RX

        # Environment Properties
        self.dist: float =  2e-6          # Distance between the TX and RX centers
        self.k_d: float = 0                            # Degradation rate in the space between TX and RX
        self.r_out: float = 1e-3                       # The environment radius, practically infinite

        # Reaction Rate Constants
        self.kab: float = 1e-1                          # For Ideal Transmitter
        self.k1: float = 3.21e3                         # From E and R to ER
        self.k_1: float = 3948                          # From ER to E and R
        self.k2: float = 889                            # From ER to ES
        self.k3: float = 3896                           # From ES to E and S
        self.k_3: float = 4.46e3                        # From E and S to ES
        self.k_2: float = self.k1 * self.k2 * self.k3 / self.k_1 / self.k_3  # Calculated

        # Simulation Parameters
        self.receiver_type: str = 'AbsorbingReceiver'
        self.step_count: int = int(1e6)
        self._simulation_end: float = 10.0

        # Point Transmitter Properties
        self.N: int = self.DEFAULT_NUM_MOLECULES        # Number of molecules to release

        # Saving and Plotting Settings
        self.save: bool = True
        self.output_folder: Path = Path(__file__).parent / 'output'
        self.plot: bool = True
        self.output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

    @property
    def simulation_end(self) -> float:
        return self._simulation_end

    @simulation_end.setter
    def simulation_end(self, value: float):
        """Setter for simulation_end"""
        if value <= 0:
            raise ValueError("Simulation end time must be greater than zero.")
        self._simulation_end = value

    @property
    def step_time(self) -> float:
        return self.simulation_end / self.step_count  # Calculated property

    @property
    def steps_in_a_unit(self) -> float:
        return self.step_count / self.simulation_end  # Calculated property

    def display_config(self) -> None:
        """Print the configuration for verification."""
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
