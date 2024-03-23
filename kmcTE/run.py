from __future__ import unicode_literals
from __future__ import print_function
import time
import numpy as np
from kmcTE.kmc import ImportMarkov, ExportMarkov, init_box, single_kmc_step
from kmcTE.analyze import calc_dos, calc_occ_dos, calc_transport, calc_transport_onthefly
from kmcTE.input import keywords_description, read_input

def Run(params):

    begin_time = time.time()

    if params["continue"] == True: # continue a finished or stoped calculation
        print("Continue to calculate based on the old Markov chain...")
        kmc_outputs = ImportMarkov("kmc_outputs")
        params = ImportMarkov("params")
        
    else: # start a new calculation
        from kmcTE.input import kmc_outputs
        kmc_outputs, params = init_box(kmc_outputs, params, keywords_description)
        
        outfile = open(params["outfile"],"w")
        transport_file = open(params["transport_file"],"w")
        error_file = open(params["error_file"],"w")
        
        outfile.write("%15s  %10s  %10s  %10s  %15s  %15s  %15s  %20s  %20s  %20s\n" 
                    %("t(s)",
                        "x_avg(nm)","y_avg(nm)","z_avg(nm)",
                        "x2_avg(nm^2)","y2_avg(nm^2)","z2_avg(nm^2)",
                        "Ex_avg(eV*nm)","Ey_avg(eV*nm)","Ez_avg(eV*nm)"))

        transport_file.write("%15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s\n"
                            %("t(s)",
                            "sigma_x(S/cm)","sigma_y","sigma_z",
                            "miu_x(cm^2/Vs)", "miu_y","miu_z",
                            "S_x(uV/K)","S_y","S_z"))

        error_file.write("%15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s\n"
                            %("t(s)",
                            "sigma_x(S/cm)","sigma_y","sigma_z",
                            "miu_x(cm^2/Vs)", "miu_y","miu_z",
                            "S_x(uV/K)","S_y","S_z"))
        
        outfile.close()
        transport_file.close()
        error_file.close()
        
        # pre-equilibrium box
        if params["warm_step"] > 0:
            print("Warming box for %d steps..." %(params["warm_step"]))
            warm_step = 1
            while warm_step <= params["warm_step"]:
                kmc_outputs = single_kmc_step(kmc_outputs, params)
                warm_step += 1

            kmc_outputs["t"] = 0
            kmc_outputs["hop_vector"] = [0,0,0]
            kmc_outputs["displace_vector"] = [0,0,0]
            kmc_outputs["energy_vector"] = [0,0,0]
            kmc_outputs["Markov_energy"] = []

    t_list = []
    transport = {
        "hop_x": [],
        "hop_y": [],
        "hop_z": [],
        "displace_x": [],
        "displace_y": [],
        "displace_z": [],
        "energy_x": [],
        "energy_y": [],
        "energy_z": [],
        }

    transport_results = []  # [[sigma_x,sigma_y,sigma_z,...],[],...,[]]

    # begin simulation
    print("Runing KMC simulations, max_step: %d..." %(params["max_step"]))
    print("%10s  %15s  %10s  %15s  %20s" %("kmc_step", "t(s)", "x_avg(nm)", "x2_avg(nm^2)", "Ex_avg(eV*nm)"))

    kmc_step = 1
    while kmc_step <= params["max_step"]:
        kmc_outputs = single_kmc_step(kmc_outputs, params)
        if kmc_step % params["wait_step"] == 0:
            outfile = open(params["outfile"],"a")
            outfile.write("%15.6e  %10d  %10d  %10d  %15.6e  %15.6e  %15.6e  %20.6f  %20.6f  %20.6f\n"
                        %(kmc_outputs["t"], 
                        kmc_outputs["hop_vector"][0], kmc_outputs["hop_vector"][1], kmc_outputs["hop_vector"][2],
                        kmc_outputs["displace_vector"][0], kmc_outputs["displace_vector"][1], kmc_outputs["displace_vector"][2],
                        kmc_outputs["energy_vector"][0], kmc_outputs["energy_vector"][1], kmc_outputs["energy_vector"][2]))
            outfile.close()
            
            t_list.append(kmc_outputs["t"])
            transport["hop_x"].append(kmc_outputs["hop_vector"][0])
            transport["hop_y"].append(kmc_outputs["hop_vector"][1])
            transport["hop_z"].append(kmc_outputs["hop_vector"][2])
            transport["displace_x"].append(kmc_outputs["displace_vector"][0])
            transport["displace_y"].append(kmc_outputs["displace_vector"][1])
            transport["displace_z"].append(kmc_outputs["displace_vector"][2])
            transport["energy_x"].append(kmc_outputs["energy_vector"][0])
            transport["energy_y"].append(kmc_outputs["energy_vector"][1])
            transport["energy_z"].append(kmc_outputs["energy_vector"][2])
            
            print("%10d %15.6e %10d %15.6e %20.6f" 
                %(kmc_step, kmc_outputs["t"], kmc_outputs["hop_vector"][0], kmc_outputs["displace_vector"][0], kmc_outputs["energy_vector"][0]))
        
        if kmc_step % params["analyze_step"] == 0:

            if params["calc_dos"] == "tot":
                calc_dos(kmc_step, kmc_outputs["Markov_energy"])
            if params["calc_dos"] == "occ":
                calc_occ_dos(kmc_step, kmc_outputs, params)
            if params["calc_dos"] == "both":
                calc_dos(kmc_step, kmc_outputs["Markov_energy"])
                calc_occ_dos(kmc_step, kmc_outputs, params)
                
            transport_results.append(calc_transport_onthefly(params, t_list, transport))
            results = np.mean(np.asarray(transport_results), axis = 0)
            results_std = np.std(np.asarray(transport_results), axis = 0)

            transport_file = open(params["transport_file"],"a")
            error_file = open(params["error_file"], "a")
            
            transport_file.write("%15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e\n"
                                %(kmc_outputs["t"], 
                                results[0], results[1], results[2],
                                results[3], results[4], results[5],
                                results[6], results[7], results[8]))
            error_file.write("%15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e\n"
                                %(kmc_outputs["t"], 
                                results_std[0], results_std[1], results_std[2],
                                results_std[3], results_std[4], results_std[5],
                                results_std[6], results_std[7], results_std[8]))
            transport_file.close()
            error_file.close()

            t_list = []
            for key in transport.keys():
                transport[key] = []

            # save check point files
            ExportMarkov(params, "params")
            ExportMarkov(kmc_outputs, "kmc_outputs")
            
        kmc_step += 1
        
    print('-'*100)
    print("Calculating physical quantities...")
    calc_transport(params)
    print('-'*100)
    print("All Done! :)")
    end_time = time.time()
    print("Run time: %.2fs." %(end_time - begin_time))
    print('-'*100)

if __name__ == '_main__':
    params = read_input("kmc.in")
    Run(params)
