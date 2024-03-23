from kmcTE.run import Run
from kmcTE.input import read_input

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
params = read_input('kmc.in')
Run(params)