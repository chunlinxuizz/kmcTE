#!/bin/env python
from __future__ import unicode_literals
from __future__ import print_function
import copy
import math
import numpy as np
from kmcTE.constants import CONST, UNIT
from kmcTE.kmc import ImportMarkov


def calc_occ_dos(kmc_step, kmc_outputs, params, center = 0, num_bins = 100):
    '''计算占据态密度'''
    state = kmc_outputs["state"]
    energy_list = params["energy_list"]
    fermi_energy = params["fermi_energy"]
    sigma = params["sigma"]
    temperature = params["temperature"]
    
    filename = "occ_dos" + str(kmc_step) + ".dat"
    occ_dos_file = open(filename,"w")
    
    occ_energy = []
    for site in range(len(energy_list)):
        if state[site]:
            occ_energy.append(energy_list[site]/(CONST["K_B"]*temperature*UNIT["J-eV"]))
    site_energy = list(copy.copy(energy_list))
    site_energy = [energy/(CONST["K_B"]*temperature*UNIT["J-eV"]) for energy in site_energy]

    site_energy.sort()
    occ_energy.sort()
    bin_w = (site_energy[-1]-site_energy[0])/num_bins
    Nsite = len(site_energy)
    
    bins = np.arange(site_energy[0], site_energy[-1]+1.0E-10, bin_w)
    bin_center = np.arange(site_energy[0]+0.5*bin_w, site_energy[-1], bin_w)
    
    n_energy_list = np.zeros(num_bins)
    n_occ_energy = np.zeros(num_bins)
    
    i = 1
    for i in range(np.shape(bins)[0]):
        for energy in site_energy:
            if bins[i-1] < energy < bins[i]:     
                n_energy_list[i-1] += 1                
        i += 1
    
    i = 1
    for i in range(np.shape(bins)[0]):
        for energy in occ_energy:
            if bins[i-1] < energy < bins[i]:     
                n_occ_energy[i-1] += 1
        i += 1
    
    n_energy_list /= (Nsite*bin_w)
    n_occ_energy /= (Nsite*bin_w)   
    
    tot_dos = (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((bin_center - center)/sigma)**2)
    partial_FD =  1/(np.exp(bin_center-fermi_energy)+1)*(1-1/(np.exp(bin_center-fermi_energy)+1))
    occ_dos = tot_dos*partial_FD

    #plot(bin_center,[n_occ_energy,occ_dos],'E','DOS')
    occ_dos_file.write("%15s  %15s  %15s\n" %("E(eV)", "occ_dos", "FD*(-df/dE)"))      
    for i in range(np.shape(bin_center)[0]):
        occ_dos_file.write("%15.6f  %15.6f  %15.6f\n" 
                           %(bin_center[i], n_occ_energy[i], occ_dos[i]))        
    
    occ_dos_file.close()

    return True

def calc_dos(kmc_step, Markov_energy, num_bins = 100):

    filename = "tot_dos" + str(kmc_step) + ".dat"
    tot_dos_file = open(filename,"w")
    
    Markov_energy.sort()
    bin_w = (Markov_energy[-1]-Markov_energy[0])/num_bins
    bins = np.arange(Markov_energy[0], Markov_energy[-1]+1.0E-10, bin_w)
    bin_center = np.arange(Markov_energy[0]+0.5*bin_w, Markov_energy[-1], bin_w)
    
    n_Markov_energy = np.zeros(num_bins)
    i = 1
    for i in range(np.shape(bins)[0]):
        for energy in Markov_energy:
            if bins[i-1] < energy < bins[i]:     
                n_Markov_energy[i-1] += 1                
        i += 1
    n_Markov_energy /= (len(Markov_energy)*bin_w)
    
    #plot(bin_center,[n_Markov_energy],'E','DOS')
    tot_dos_file.write("%15s  %15s\n" %("E(eV)", "tot_dos"))      
    for i in range(np.shape(bin_center)[0]):
        tot_dos_file.write("%15.6f  %15.6f\n" %(bin_center[i], n_Markov_energy[i]))        
    
    tot_dos_file.close()
    
    return True

def calc_transport(params,file_in = 'kmcout.dat'):
    '''calculate physical quantities of a finished kMC simulation'''    
    print("%15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s"
                         %("t(s)",
                           "sigma_x(S/cm)","sigma_y","sigma_z",
                           "miu_x(cm^2/Vs)", "miu_y","miu_z",
                           "S_x(uV/K)","S_y","S_z"))

    analyze_step = params["analyze_step"]
    wait_step = params["wait_step"]
    
    file_in = open(file_in,'r')
    file_in.readline()
    
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
    transport_results = []
    
    i = 0
    while True:
        line = file_in.readline()
        i += 1
        if not line:
            break
        line = line.split()
        t_list.append(float(line[0]))
        transport["hop_x"].append(float(line[1]))
        transport["hop_y"].append(float(line[2]))
        transport["hop_z"].append(float(line[3]))
        transport["displace_x"].append(float(line[4]))
        transport["displace_y"].append(float(line[5]))
        transport["displace_z"].append(float(line[6]))
        transport["energy_x"].append(float(line[7]))
        transport["energy_y"].append(float(line[8]))
        transport["energy_z"].append(float(line[9]))
        
        if i % int(analyze_step/wait_step) == 0:
            transport_results.append(calc_transport_onthefly(params, t_list, transport))
            results = np.mean(np.asarray(transport_results), axis = 0)
            print("%15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e"
                             %(t_list[-1], 
                               results[0], results[1], results[2],
                               results[3], results[4], results[5],
                               results[6], results[7], results[8]))
            t_list = []
            for key in transport.keys():
                transport[key] = []
            
    results_std = np.std(np.asarray(transport_results), axis = 0)
    print("Standard error:")
    print("%15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s"
                         %("sigma_x(S/cm)","sigma_y","sigma_z",
                           "miu_x(cm^2/Vs)", "miu_y","miu_z",
                           "S_x(uV/K)","S_y","S_z"))
    print("%15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e"
                         %(results_std[0], results_std[1], results_std[2],
                           results_std[3], results_std[4], results_std[5],
                           results_std[6], results_std[7], results_std[8]))
    file_in.close()

    return None

def calc_transport_onthefly(params, t_list, transport):
    '''calculate physical quantities when performing kMC simulation'''
    elec_field = params["elec_field"]
    n_site = params["n_site"]
    n_carrier = params["n_carrier"]
    temperature = params["temperature"]
    chg_num = params["chg_num"]    

    if n_carrier > n_site + 1.0E-10:
        n_carrier = 2*n_site - n_carrier
    
    t_list = [t - t_list[0] for t in t_list]
    t_list = np.asarray(t_list)[:,np.newaxis]
    
    transport["displace_x"] = [transport["displace_x"][i]/chg_num 
                               - (transport["hop_x"][i]/chg_num)**2 for i in range(len(t_list))]
    transport["displace_y"] = [transport["displace_y"][i]/chg_num 
                               - (transport["hop_y"][i]/chg_num)**2 for i in range(len(t_list))]
    transport["displace_z"] = [transport["displace_z"][i]/chg_num 
                               - (transport["hop_z"][i]/chg_num)**2 for i in range(len(t_list))]
    
    fit_results = {}
    for key in transport.keys():
        transport[key] = [x - transport[key][0] for x in transport[key]]
        transport[key] = np.asarray(transport[key])
        a,_,_,_ = np.linalg.lstsq(t_list, transport[key], rcond = None)
        fit_results[key] = float(a[0])
    
    results = {
        "sigma_x": 0.0, # in S/cm
        "sigma_y": 0.0, # in S/cm
        "sigma_z": 0.0, # in S/cm
        "miu_x": 0.0, # in cm^2/(Vs)
        "miu_y": 0.0, # in cm^2/(Vs)
        "miu_z": 0.0, # in cm^2/(Vs)
        "Seebeck_x": 0.0, # in uV/K
        "Seebeck_y": 0.0, # in uV/K
        "Seebeck_z": 0.0, # in uV/K
        }
    
    if elec_field[0] != 0:
        results["sigma_x"] = abs(fit_results["hop_x"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"])
        results["Seebeck_x"]= -fit_results["energy_x"]/fit_results["hop_x"]/temperature*1.0E6
        
    if elec_field[1] != 0:
        results["sigma_y"] = abs(fit_results["hop_y"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"])
        results["Seebeck_y"]= -fit_results["energy_y"]/fit_results["hop_y"]/temperature*1.0E6
        
    if elec_field[2] != 0:
        results["sigma_z"] = abs(fit_results["hop_z"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"])
        results["Seebeck_z"]= -fit_results["energy_z"]/fit_results["hop_z"]/temperature*1.0E6
    
    results["miu_x"] = (fit_results["displace_x"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)
    results["miu_y"] = (fit_results["displace_y"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)
    results["miu_z"] = (fit_results["displace_z"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)

    return [results[key] for key in results.keys()]

def calc_transport_linearfit(params,file_in = 'kmcout.dat'):
    print("%15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s  %15s"
                         %(" ",
                           "sigma_x(S/cm)","sigma_y","sigma_z",
                           "miu_x(cm^2/Vs)", "miu_y","miu_z",
                           "S_x(uV/K)","S_y","S_z"))
    elec_field = params["elec_field"]
    n_site = params["n_site"]
    n_carrier = params["n_carrier"]
    temperature = params["temperature"]
    chg_num = params["chg_num"]
    
    if n_carrier > n_site + 1.0E-10:
        n_carrier = 2*n_site - n_carrier
    
    file_in = open(file_in,'r')
    
    file_in.readline()
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
    results = {
        "sigma_x": 0.0, # in S/cm
        "sigma_y": 0.0, # in S/cm
        "sigma_z": 0.0, # in S/cm
        "miu_x": 0.0, # in cm^2/(Vs)
        "miu_y": 0.0, # in cm^2/(Vs)
        "miu_z": 0.0, # in cm^2/(Vs)
        "Seebeck_x": 0.0, # in uV/K
        "Seebeck_y": 0.0, # in uV/K
        "Seebeck_z": 0.0, # in uV/K
        }
    
    while True:
        line = file_in.readline()
        if not line:
            break
        line = line.split()
        
        t_list.append(float(line[0]))
        transport["hop_x"].append(float(line[1]))
        transport["hop_y"].append(float(line[2]))
        transport["hop_z"].append(float(line[3]))
        transport["displace_x"].append(float(line[4])/chg_num - (float(line[1])/chg_num)**2)
        transport["displace_y"].append(float(line[5])/chg_num - (float(line[2])/chg_num)**2)
        transport["displace_z"].append(float(line[6])/chg_num - (float(line[3])/chg_num)**2)
        transport["energy_x"].append(float(line[7]))
        transport["energy_y"].append(float(line[8]))
        transport["energy_z"].append(float(line[9]))
    file_in.close()
    
    t_list = np.asarray(t_list)[:,np.newaxis]
    fit_results = {}
    for key in transport.keys():
        transport[key] = np.asarray(transport[key])
        a,_,_,_ = np.linalg.lstsq(t_list, transport[key], rcond = None)  # fit y = a*x
        fit_results[key] = float(a[0])
        
    if elec_field[0] != 0:
        results["sigma_x"] = fit_results["hop_x"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"]
        results["Seebeck_x"]= -fit_results["energy_x"]/fit_results["hop_x"]/temperature*1.0E6
        
    if elec_field[1] != 0:
        results["sigma_y"] = fit_results["hop_y"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"]
        results["Seebeck_y"]= -fit_results["energy_y"]/fit_results["hop_y"]/temperature*1.0E6
        
    if elec_field[2] != 0:
        results["sigma_z"] = fit_results["hop_z"]*n_carrier*CONST["E_CHARGE"]*UNIT["nm-cm"]/elec_field[0]*UNIT["m-cm"]
        results["Seebeck_z"]= -fit_results["energy_z"]/fit_results["hop_z"]/temperature*1.0E6
    
    results["miu_x"] = (fit_results["displace_x"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)
    results["miu_y"] = (fit_results["displace_y"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)
    results["miu_z"] = (fit_results["displace_z"]*UNIT["nm-cm"]**2/2*chg_num)*CONST["E_CHARGE"]/(CONST["K_B"]*temperature)

    print("%15s  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e  %15.6e"
                         %(" ",kmc_outputs["t"],abs(results["sigma_x"]), abs(results["sigma_y"]), abs(results["sigma_z"]),
                           results["miu_x"], results["miu_y"], results["miu_z"],
                           results["Seebeck_x"], results["Seebeck_y"], results["Seebeck_z"]))
    return None

#--------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    kmc_outputs = ImportMarkov("kmc_outputs")
    params = ImportMarkov("params")

    # calc_occ_dos(0, kmc_outputs, params)
    calc_transport(params)
    calc_transport_linearfit(params)

