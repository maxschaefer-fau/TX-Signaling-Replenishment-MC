# R-S Mandelate Diffusion Model with Reaction

# Table of Contents

* [General Layout](#general-layout)
* [Package Installation](#package-installation)
* [Running the Code](#running-the-code)
* [Settings](#settings)

# General Layout
The code is implemented in the following parts:
1) `diffusion.py`: Runs the simulation and generates the results in a directory called `diffusion_res`
2) `config.py`: Includes all of the settings and connstants required for the model.
3) `diff_combine.py`: Combines the results in the folder `Results_Combine` and saves them to `Combined_Plots`
4) `create_state_values.py`: Not to be run, helper to create states when no external input is given.

# Package Installation
First of all, one needs to have [Python](https://www.python.org/downloads/) to run the code. The version used to test the code is 3.8.5. The Python directory along with the scripts directory need to be added to the environment variables to directly run commands from the terminal.

Then, the package dependencies need to be installed. The versions that the codes were run with are also provided, but they do not need to be exactly the same. These are the package dependencies:

 - numpy - v1.22.1
 - matplotlib - v3.3.4
 - tqdm - v0.0.1
 - scipy - v1.5.2

To install a package, use the command `pip install <package>=<version>` on your terminal.

# Running the Code

To run the code:

 1) Change settings on `config.py` if needed.
 2) Run `diffusion.py`, output will be generated on `diffusion_res` directory.
 3) Create a directory `Results_Combine` if not created yet.
 4) Move the `diffusion_res` directory to `Results_Combine`, rename `diffusion_res` to your scenario name to be seen in the plots.
 5) Repeat this process from steps 1-4 until all required scenario results are generated.
 6) Run `diff_combine.py`, the comparison results will be on `Combined_Plots`.

# Settings

The settings can be found in the file `config.py`. The file includes the settings on receiver, transmitter, the environment and the molecule counts. The simulation precision can be modified from the simulation parameters. The way the membrane states are initialized can be modified, or the membrane states can be provided externally.

1) **Nanomachine Properties**: In this part, the TX and RX radii are defined.
2) **Diffusion Settings**: In this part, the diffusion and permeability coefficients on the medium and the membrane are defined. The opening and closing of the membrane can be made noninstantaneous if the parameter `p_close` is set to a positive value.
3) **Environment Properties**: These are the properties for the environment or the channel, such as degradation and TX-RX distance.
4) **Reaction Rate Constants**: These constants are the experimentally found reaction rate constants for the (R)-Mandelate to (S)-Mandelate reaction. Not to be modified.
5) **Molecule Counts**: Defined as an object for the inside and outside. Changing the first parameter of the `Particle` objects for the inside or the outside will change the corresponding initial molecule counts.
6) **Simulation Parameters**: `step_count` and `simulation_end` can be changed. `step_count` will determine the number of total steps in the simulation, so it defines the precision. `simulation_end` determines the total simulation time.
7) **States**: The states are an array with the length of `step_count`. In the settings, a way to define this array is provided by the parameters. `release_count` is the number of releases, `end_part` is the part after which the TX stays closed. The releases are distributed evenly, but that can be changed by changing `state_breaks`. Note that in this part the simulation is considered of length 1, so `end_part=0.5` means the TX will stay closed in the second half of the simulation.
