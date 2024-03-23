from __future__ import unicode_literals
from __future__ import print_function

__default_input = {
    "size": (50, 50, 50),
    "dcut": 3,
    "unit_length": (1.0, 1.0, 1.0),
    "chg_num": 1000,
    "injection_mode": "fermi",
    "calc_fermi": False,
    "fermi_energy": -0.0,
    "sigma": 0.0,
    "niu0": 1.0E12,
    "alpha": 1.0,
    "temperature": 300.0,
    "elec_field": (0.0, 0.0, 0.0),
    "warm_step": 0,
    "max_step": 0,
    "wait_step": 0,
    "analyze_step": 10000,
    "continue": False,
    "calc_dos": False,
    "outfile": "kmcout.dat",
    "transport_file": "transport.dat",
    "error_file": "error.dat",
}


keywords_description = {
    "size": "the number of sites along x,y,z directions (Nx, Ny, Nz).",
    "dcut": "threshold distance for hopping, must be int, in nm.",
    "unit_length": "unit length along x,y,z directions (a, b, c), in nm.",
    "chg_num": "the number of charges in the box.",
    "injection_mode": "the way to inject carrier, either 'random' or 'fermi'.",
    "calc_fermi": "'True' to calculate; 'False' for given values.",
    "fermi_energy": "set the fermi energy if 'calc_fermi = False', in eV.",
    "sigma": "control the width of Gauss-type DOS, in eV.",
    "niu0": "attempt-to-jump frequency controled by phonon modes, in s^-1.",
    "alpha": "control the tendency of MA-type wave-package, in usual 'alpha = a', in nm.",
    "temperature": "the temperature of bath, in Kelvin.",
    "elec_field": "electric field along x,y,z directions (Fx, Fy, Fz), in V/m",
    "warm_step": "the step needed to pre-simulation for better equilibrium.",
    "max_step": "the maximum number of kMC steps.",
    "wait_step": "interval steps between two measurements.",
    "analyze_step": "interval steps between two transport and DOS calculations.",
    "continue": "'True' to continue a finished calculation.",
    "calc_dos": "whether to calculate DOS; 'occ' or 'tot' or 'both'.", 
    "outfile": "restore the intermediate quantities of kMC simulations.",
    "transport_file": "real-time transport coefficients.",
    "error_file": "real-time errors of transport coefficients.",
}

__trans_dict = {
    int: lambda x: int(x),
    float: lambda x: float(x),
    bool: lambda x: "T" in x,
    str: lambda x: str(x)
}

kmc_outputs = {
    "t": 0, # 模拟时间
    "state": [], # 占据状态[0,0,1,0,...,1,0,0,1...]
    "tau_list": [], # 每个载流子的等待时间
    "k_tot_list": [], # 每个载流子的总跳跃速率
    "hop_vector": [0,0,0], # 输出的平均位移
    "displace_vector": [0,0,0], # 输出的总平方位移
    "energy_vector": [0,0,0], # 输出的位移加权的能量，用于计算E_trans
    "Markov_energy": [] # 每一步模拟过程中的占据能量，用于计算态密度
    }

def read_input(file_input):
    input_dict = {}
    try:
        file_in = open(file_input, 'r')
  
        for line in file_in:
            line = line.strip()
            if len(line) > 0:
                if "file" in line or "dir" in line:
                    terms = line.replace('=', ' ').split()
                    terms[0] = terms[0].lower()
                else:
                    terms = line.lower().replace('=', ' ').split()
                    # print(terms)

                if len(terms) > 1:
                    input_dict[terms[0]] = terms[1:]

        file_in.close()
    except FileNotFoundError:
        print("-"*100)
        print("Warning: 'kmc.in' is missing, using defult parameters...")

    for key in __default_input:
        value = input_dict.get(key)

        if value == None:
            value = __default_input[key]
        else:
            value_type = type(__default_input[key])
            if value_type in __trans_dict:
                value = __trans_dict[value_type](value[0])
            elif value_type == tuple:
                value_type = type(__default_input[key][0])
                value = tuple([__trans_dict[value_type](x) for x in value])
        input_dict[key] = value

    keys = __default_input.keys()

    return input_dict

