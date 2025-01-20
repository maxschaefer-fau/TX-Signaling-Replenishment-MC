import numpy as np
from config import NanomachineConfig
from utils import generate_switching_pattern, get_conc_vol_for_practical, save_to_csv
from ideal_transmitter import ideal_transmitter
from practical_transmitter import practical_transmitter

# Set Config
conf = NanomachineConfig()
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

# Set Switching pattern
switching_pattern = [1,0,1,0,1]
Ts = 2
print(conf.step_time)
conf.simulation_end = len(switching_pattern) * Ts
print(conf.step_time)
rho = generate_switching_pattern(switching_pattern, time_interval=Ts, config=conf)

# Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

# IdealTx
results_ideal = ideal_transmitter(rho=rho,
                                  time_array=time_array,
                                  config=conf)

# PracticalTx
results_practicle = practical_transmitter(rho=rho,
                                          time_array=time_array,
                                          conc_in=conc_in,
                                          conc_out=conc_out,
                                          config=conf)

if conf.save:
   file_name = 'test_ideal1'
   save_to_csv(results_ideal, file_name, conf)

if conf.plot:
   pass

