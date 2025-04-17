# Molecular Communication Experiements

# Table of Contents

* [General Layout](#general-layout)
* [Environment Setup](#package-installation)
* [Running the Code](#running-the-code)
* [Settings](#settings)

# General Layout
The code is implemented in the following parts:
1) `main.py`: Main file to run.
2) `transmitters/ideal_transmitter.py`: Code for Ideal Transmitter.
3) `transmitters/practical_transmitter.py`: Code for Practical Transmitter.
4) `transmitters/point_transmitter.py`: Code for Point Transmitter.
5) `models/space.py`: Code for Receivers, Particle and Space.
6) `utils/config.py`: Includes all of the settings and constants required for the model.
7) `utils/utils.py`: Contains code for utility functions.

# Environment Setup

Create Conda Environment from `conda_env.yml` file

```
conda env create -f environment.yml
conda activate idealTx
```


# Running the Code
1) Configure Simulation Settings
   - Modify simulation parameters as necessary in `config.py`. You can customize settings such as the nanomachine configuration, simulation duration, and other essential properties specific to your scenarios.

2) Run the Simulation
   - Execute the main script to run the simulation:
     ```bash
     python main.py
     ```

# Settings

The settings can be found in the file `config.py`. The file includes the settings on receiver, transmitter, the environment and the molecule counts. The simulation precision can be modified from the simulation parameters. The way the membrane states are initialized can be modified, or the membrane states can be provided externally.

1) **Nanomachine Properties**: In this part, the TX and RX radii are defined.
2) **Diffusion Settings**: In this part, the diffusion and permeability coefficients on the medium and the membrane are defined. The opening and closing of the membrane can be made noninstantaneous if the parameter `p_close` is set to a positive value.
3) **Environment Properties**: These are the properties for the environment or the channel, such as degradation and TX-RX distance.
4) **Reaction Rate Constants**: These constants are the experimentally found reaction rate constants for the (R)-Mandelate to (S)-Mandelate reaction. Not to be modified.
5) **Molecule Counts**: Defined as an object for the inside and outside. Changing the first parameter of the `Particle` objects for the inside or the outside will change the corresponding initial molecule counts.
6) **Simulation Parameters**: `step_count` and `simulation_end` can be changed. `step_count` will determine the number of total steps in the simulation, so it defines the precision. `simulation_end` determines the total simulation time.

# Building Docs

```
cd docs/ && make html
```

- Run generated docs

    ```
    run docs/_build/html/index.html
    ```
