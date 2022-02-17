import sys
from MARM.analysis import read_settings, run_and_store_simulation

settings = read_settings(sys.argv)
run_and_store_simulation(settings, 'trainingdata')
run_and_store_simulation(settings, 'trainingdata',
                         channel_specific_dusp_binding=True)
