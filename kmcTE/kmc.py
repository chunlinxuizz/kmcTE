import itertools
import math
import numpy as np
import copy
import random
import time
import pickle
import sympy
from kmcTE.constants import CONST, UNIT
from kmcTE.input import kmc_outputs
import matplotlib.pyplot as plt

#--------------------------------格点操作函数-----------------------------------------
def __symmetric_sites_cubic(i, j, k):
    '''简单立方{ijk}的所有对称格点'''
    R = math.sqrt(i**2+j**2+k**2)
    group = list(set(list(itertools.permutations([i,j,k],3))))
    sign = list(itertools.product([-1,1],repeat=3))
    tmp = list(itertools.product(np.array(group),np.array(sign)))
    sym_list = [tuple(tmp[i][0]*tmp[i][1]) for i in range(len(tmp))]
    sym_list = list(set(sym_list))
    sym_list = [list([R,sym_list[i]]) for i in range(len(sym_list))]

    return R, sym_list, len(sym_list)

def __get_pair_vector(dcut):
    '''在dcut范围内所有近邻格点的矢量'''
    npair = 0
    pair_vector = []
    for i in range(dcut+1):
        for j in range(i+1):
            for k in range(j+1):
                R, symmetric_sites, n_sym_sites = __symmetric_sites_cubic(i,j,k)
                if 0 < R <= dcut:
                    pair_vector += symmetric_sites
                    npair += n_sym_sites
    print("%d neighbors are found within the distance cutoff %d nm." %(npair,dcut))
    return pair_vector, npair

def __site_num_2_position(site_num, size):
    '''简单立方从格点序号转换为位置矢量'''
    nx, ny, nz = size
    x = site_num % (nx*ny) % nx
    y = site_num % (nx*ny) // nx
    z = site_num // (nx*ny)
    
    return (x, y, z)

def __get_pair_list(site_num, pair_vector, size, dcut):
    '''求距离指定格点的所有在dcut范围内的格点'''
    nx, ny, nz = size
    site_position = __site_num_2_position(site_num,size)
    npair = len(pair_vector)
    pair_list = []
    for i in range(npair):
        pair_position = [[],[],[]]
        for j in range(3):
            pair_position[j] = pair_vector[i][1][j] + site_position[j]
            if pair_position[j] < 0:
                pair_position[j] += size[j]
            if pair_position[j] > (size[j]-1):
                pair_position[j] -= size[j]
        pair_list.append(pair_position[0]+pair_position[1]*nx+pair_position[2]*nx*ny)
        
    return pair_list

def __gen_ene_gauss(Nsite, sigma, center = 0):
    print("Generating site energy according to Gauss distribution...")
    energy_list = np.random.normal(loc=center, scale=sigma, size=Nsite)
    energy_list = tuple(energy_list)
    
    return energy_list

#--------------------------------载流子操作函数----------------------------------------
def __inject_carrier(site,state):
    state[site] = 1
    
    return state

def __is_occupied(site,state):
    #read the bit(0 or 1) on the (site)th position of (state) 
    return state[site]

def __hopping_action(i,j,state):
    '''i -> j'''
    state[i] = 0
    state[j] = 1

    return state

def __calc_Efermi(chg_num,ene_list,temperature):
    print("Calculating Fermi energy...")
    E_min, E_max = min(ene_list), max(ene_list)   
 
    for itr in range(10000):
        n, fermi_energy = 0.0, (E_min+E_max)/2.0
        
        for energy in ene_list:
            n += 1/(math.exp((energy-fermi_energy)/(CONST["K_B"]*temperature*UNIT["J-eV"]))+1)
        
        if abs(n - float(chg_num)) < 1.0E-3:
            break
        elif n > chg_num:
                E_max = fermi_energy
        else:
            E_min = fermi_energy
    
    print("Fermi energy is: %.3f eV, %.3f*kT." %(fermi_energy,fermi_energy/(CONST["K_B"]*temperature*UNIT["J-eV"])))
    return fermi_energy

def __fermi_injection(chg_num, energy_list, temperature, fermi_energy = 0.0, calc_fermi = None):
    Nsite = len(energy_list)
    if calc_fermi == True:
        fermi_energy = __calc_Efermi(chg_num, energy_list, temperature)
    else:
        print("Fermi energy is set to: %s eV, %.3f kT." %(fermi_energy,fermi_energy/(CONST["K_B"]*temperature*UNIT["J-eV"])))

    print("Injecting charges according to Fermi-Dirac distribution...")
    FD_list = []
    FD = 0
    for i in range(Nsite):
        FD = 1/(math.exp((energy_list[i]-fermi_energy)/(CONST["K_B"]*temperature*UNIT["J-eV"]))+1)
        FD_list.append(FD)
        
    ini_state = [0 for i in range(Nsite)]
    injected_carrier = 0
    niter = 0
    while injected_carrier < chg_num:
        try_site = random.randint(0,Nsite-1)
        if not __is_occupied(try_site, ini_state):
            final_state = ini_state
            Acceptance = FD_list[try_site]
            r = random.random()
            if r < Acceptance:
                final_state =  __inject_carrier(try_site, ini_state)
                injected_carrier += 1
                ini_state = final_state
            niter += 1
    print("Injection finished after %d steps." %niter)
    
    return final_state, fermi_energy

def __random_injection(chg_num, energy_list, temperature, fermi_energy = 0.0, calc_fermi = None):
    if calc_fermi == True:
        fermi_energy = __calc_Efermi(chg_num, energy_list, temperature)
    else:
        print("Fermi energy is set to: %s eV; %.3f kT." %(fermi_energy,fermi_energy/(CONST["K_B"]*temperature*UNIT["J-eV"])))
    
    print("Injecting charges randomly...")
    Nsite = len(energy_list)
    ini_state = [0 for i in range(Nsite)]
    injected_carrier = 0
    niter = 0
    while injected_carrier < chg_num:
        try_site = random.randint(0,Nsite-1)
        final_state =  __inject_carrier(try_site, ini_state)
        injected_carrier += 1
        ini_state = final_state
        niter += 1
    
    return final_state, fermi_energy

def __calc_concentration(Nsite, chg_num, unit_length, temperature, fermi_energy, energy_list, calc_fermi, sigma, mode = None):

    n_site = 1/(unit_length[0]*unit_length[1]*unit_length[2]*(UNIT["nm-cm"]**3))  # cm^-3


    if calc_fermi:
        n_carrier = chg_num/Nsite*n_site
        print("The charge concentration is %.3e cm^-3" %(n_carrier))
    else:
        if mode == "continuous":
            n_carrier = integrate.quad(lambda x: \
                                n_site/sigma/(math.sqrt(2*math.pi))*math.exp(-0.5*x**2/sigma**2)*\
                                    (1/(math.exp((x-fermi_energy)/(CONST["K_B"]*temperature*UNIT["J-eV"]))+1)),\
                                     -230*CONST["K_B"]*temperature*UNIT["J-eV"], 230*CONST["K_B"]*temperature*UNIT["J-eV"])[0]
            print("The charge concentration is %.3e cm^-3." %(n_carrier))
            
        if mode == None:
            chg_num = 0
            for i in range(Nsite):
                chg_num += 1/(math.exp((energy_list[i]-fermi_energy)/(CONST["K_B"]*temperature*UNIT["J-eV"]))+1)
            if chg_num > Nsite/2 + 1.0E-10:
                chg_num = Nsite - chg_num
            n_carrier = chg_num/Nsite*n_site
            
            print("The charge concentration is %.3e cm^-3." %(n_carrier))

    return n_site, float(n_carrier)

#-----------------------------储存和读取Markov链-------------------------------------------    
def ImportMarkov(lable):
	try:
		File_Markov = open(lable+'.pklz','rb')
		MarkovChain = pickle.load(File_Markov)
		File_Markov.close()
	except IOError:
		print('Missing') 
	
	return MarkovChain

def ExportMarkov(MarkovChain, lable):
	File_Markov = open(lable +'.pklz','wb')
	pickle.dump(MarkovChain,File_Markov)
	File_Markov.close()
	return None

#-----------------------------计算KMC所需参数------------------------------------------  
def __calc_rate_MA(Ei,Ej, R, vec, niu0, alpha, temperature, fermi_energy, elec_field, mode = None):
    '''根据MA公式计算速率常数，适用于电子。能量的单位为eV'''
    k0 = niu0 * math.exp(-2 * R / alpha)
    W = abs(Ei-Ej) + abs(Ei-fermi_energy) + abs(Ej-fermi_energy)

    if mode == "mobility":
        W = Ej-Ei + abs(Ei-Ej)
    for i in range(3):
        W += vec[i] * elec_field[i]*UNIT["nm-m"]

    k = k0 * math.exp(-0.5*W/(CONST["K_B"]*temperature*UNIT["J-eV"]))

    return k

def __calc_tau_MA(size, dcut, pair_vector, site_num, energy_list, niu0,alpha, temperature, fermi_energy, elec_field):
    '''根据MA公式计算位于site_num的载流子的等待时间'''
    '''k_list: [[pair,k],[[pair,k]],...,[[pair,k]]]'''
    '''k_tot_list: [[pair,k_tot],[[pair,k_tot]],...,[[pair,k_tot]]]，用于选择跳跃'''
    
    pair_list = __get_pair_list(site_num,pair_vector,size,dcut)
    npair = len(pair_vector)
    k_list = []
    k_tot_list = []
    k_tot = 0
    for j in range(npair):
        pair_num = pair_list[j]
        R = pair_vector[j][0]
        vec = pair_vector[j][1]
        
        k_ij = __calc_rate_MA(energy_list[site_num],energy_list[pair_num],R,vec,niu0,alpha,temperature, fermi_energy, elec_field)
        k_tot += k_ij
        k_list.append([pair_num,k_ij])
        k_tot_list.append([pair_num,k_tot])
       
    tau = 1/k_tot
    for i in range(npair):
        k_tot_list[i][1] /= k_tot
        
    return k_list, k_tot_list, tau, k_tot

def __calc_tau_list(size, dcut, state, Nsite, pair_vector, energy_list, niu0, alpha, temperature, fermi_energy, elec_field):
    '''tau_list: [[site,tau],[,],...,[,]]'''
    '''k_tot_list: [[site,k_tot],[],...,[]]'''
    '''计算所有载流子的等待时间，用于选择跳跃的载流子'''
    print("Calculating rate constants using Miller-Abraham equation...")

    k_tot_list= []
    tau_list = []
    for site in range(Nsite):
        if __is_occupied(site, state):
           _,_, tau, k_tot = __calc_tau_MA(size, dcut, pair_vector, site, energy_list, niu0, alpha,  temperature,fermi_energy, elec_field)
           k_tot_list.append([site,k_tot])
           tau_list.append([site,tau])
           # tau_list.sort(key=lambda x:x[1])  # set for the first reaction principle

    return tau_list, k_tot_list

def __get_k_tot_list(k_list):
    '''k_list: [[site,k or k_tot],[site,k or k_tot],[site,k or k_tot]]
       k_tot_list: an accumulation list of k_tot of each carrier to chose 
       the hop carrier, replace the tau_list in the first reaction principle'''

    k_tot = 0
    k_tot_list = []
    for i in range(len(k_list)):
        site = k_list[i][0]
        k_tot += k_list[i][1]
        k_tot_list.append([site,k_tot])
    
    for i in range(len(k_tot_list)):
        k_tot_list[i][1] /= k_tot
        
    return k_tot_list

def __random_walker(k_tot_list):
    '''摇骰子机器：产生随机数，利用二叉树法查找满足条件的载流子或site'''
    n_candidate = len(k_tot_list)
    if n_candidate == 1:
        try_site = k_tot_list[0][0]
        mid = 0
        
    else:
        r = random.random()
        try_site = 0 
        start, end = 0, n_candidate-1
        while start < end:
            mid = (start + end) // 2
            if k_tot_list[mid-1][1] < r < k_tot_list[mid][1]:
                try_site = k_tot_list[mid][0]        
                break
            elif mid == 0:
                try_site = k_tot_list[mid][0]
                break
            elif (mid == end-1) and (mid == start):
                mid = end
                try_site = k_tot_list[mid][0]
                break
            elif r > k_tot_list[mid][1]:
                start = mid
            else: 
                end = mid
            
    return mid, try_site

#-------------------------------正式KMC功能函数------------------------------------------
def init_box(kmc_outputs, params, keywords_description):
    size = params["size"]
    dcut = params["dcut"]
    unit_length = params["unit_length"]
    chg_num = params["chg_num"]
    sigma = params["sigma"]
    niu0 = params["niu0"]
    alpha = params["alpha"]
    temperature = params["temperature"]
    elec_field = params["elec_field"]
    injection_mode = params["injection_mode"]
    calc_fermi = params["calc_fermi"]
    fermi_energy = params["fermi_energy"]
    
    nx, ny, nz = size
    Nsite = nx*ny*nz
    print('-'*100)
    print("%dx%dx%d cubic box with %d sites and %d electrons." %(nx,ny,nz,Nsite,chg_num))
    pair_vector, npair = __get_pair_vector(dcut)
    energy_list = __gen_ene_gauss(Nsite, sigma)
    if injection_mode == "fermi":
        ini_state, params["fermi_energy"] = __fermi_injection(chg_num, energy_list, temperature, fermi_energy, calc_fermi)
    elif injection_mode == "random":
        ini_state, params["fermi_energy"] = __random_injection(chg_num, energy_list, temperature, fermi_energy, calc_fermi)
    else:
        print("Unexpected carrier injection mode, only 'fermi' and 'random' are supported!")
        
    n_site, n_carrier = __calc_concentration(Nsite, chg_num, unit_length, temperature, fermi_energy, energy_list, calc_fermi, sigma)
    
    tau_list, k_tot_list = __calc_tau_list(
        size,dcut,ini_state,Nsite,pair_vector,energy_list,niu0,alpha,temperature,fermi_energy, elec_field)
    
    print("Finish to initialize box.")
    print('-'*100)
    for key in params.keys():
        print("| %-20s  %-25s  # %20s" %(key,params[key],keywords_description[key]))
    print('-'*100)

    params["energy_list"] = energy_list
    params["pair_vector"] = pair_vector
    params["n_site"] = n_site
    params["n_carrier"] = n_carrier
    kmc_outputs["state"] = ini_state
    kmc_outputs["tau_list"] = tau_list
    kmc_outputs["k_tot_list"] = k_tot_list
    
    return kmc_outputs, params

def single_kmc_step(kmc_outputs, params):
    # 1. select an electron to hop according to the first reaction principle
    # 2. select a neighbor site to hop according to the k_ij
    # 3. if try_side is not occ., hop succeed; else, hop failed
    # 4. recalculate rates (here we won't do this)
    # 5. update tau_list (written in step 3)
    
    size = params["size"]
    dcut = params["dcut"]
    niu0 = params["niu0"]
    alpha = params["alpha"]
    temperature = params["temperature"]
    elec_field = params["elec_field"]
    pair_vector = params["pair_vector"]
    energy_list = params["energy_list"]
    fermi_energy = params["fermi_energy"]
    
    ini_state = kmc_outputs["state"]
    tau_list = kmc_outputs["tau_list"]
    k_tot_list = kmc_outputs["k_tot_list"]
    
    k_tot_tot_list = __get_k_tot_list(k_tot_list)
    site_num, try_carrier = __random_walker(k_tot_tot_list)
    tau = tau_list[site_num][1] 
    
    _,k_tot_site_list,_,_ = __calc_tau_MA(
        size, dcut, pair_vector, try_carrier, energy_list, niu0, alpha, temperature, fermi_energy, elec_field)
    
    pair_num, try_site = __random_walker(k_tot_site_list)

    if __is_occupied(try_carrier, ini_state) and not __is_occupied(try_site, ini_state):
        final_state = __hopping_action(try_carrier, try_site, ini_state)
        
        _,_,tau_try_site,k_tot_try_site = __calc_tau_MA(
            size, dcut, pair_vector, try_site, energy_list, niu0, alpha, temperature, fermi_energy, elec_field)
        
        tau_list[site_num] = [try_site,tau_try_site]  #更新等待时间列表
        k_tot_list[site_num] = [try_site,k_tot_try_site]  #更新速率列表
        
        #计算物理量
        E_transport = (energy_list[try_site] + energy_list[try_carrier]) / 2.0 - fermi_energy
        kmc_outputs["Markov_energy"].append(energy_list[try_site])
        hop_vector = pair_vector[pair_num][1]
        
        for i in range(3):
            kmc_outputs["hop_vector"][i] += hop_vector[i]
            kmc_outputs["displace_vector"][i] += hop_vector[i]**2
            kmc_outputs["energy_vector"][i] += (hop_vector[i] * E_transport)

    else:
        final_state = ini_state
        tau_list = tau_list
        k_tot_list = k_tot_list
        try_site = try_carrier
    
    r = random.random()
    kmc_outputs["t"] -= tau*math.log(r)   
    kmc_outputs["state"] = final_state
    kmc_outputs["tau_list"] = tau_list
    kmc_outputs["k_tot_list"] = k_tot_list
    
    return kmc_outputs

