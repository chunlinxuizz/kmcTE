#!/bin/env python
# Define constants

# CONST is for storing constants
CONST = {}
# math constants
CONST["PI"] = 3.14159265357  # circumference ratio

# basic physical constants
CONST["H"] = 6.62607004081E-34  # J*s, Planck constant
CONST["K_B"] = 1.3806485279E-23   # J/K, Boltzmann constant
CONST["E_CHARGE"] = 1.602176620898E-19  # C, elementary charge
CONST["C_LIGHT"] = 299792458          # m/s, speed of light
CONST["M_E"] = 9.1093835611E-31   # kg, rest mass of the electron
# H/m, vacuum permeability (P.S. H(henry) = ohm*s is the SI unit of electrical inductance)
CONST["MIU_0"] = 4.0E-7 * CONST["PI"]

# derived physical constants
# J*s, reduced Planck constant
CONST["H_BAR"] = CONST["H"] / (2.0 * CONST["PI"])
# F/m, vacuum permittivity (P.S. F(farad) = C/V is the SI unit of electrical capacitance)
CONST["EPSILON_0"] = 1.0 / (CONST["MIU_0"] * CONST["C_LIGHT"]**2)
CONST["RYDBERG"] = CONST["M_E"] * CONST["E_CHARGE"]**4 / \
    (8.0 * CONST["EPSILON_0"]**2 * CONST["H"] **
     3 * CONST["C_LIGHT"])  # m^-1, Rydberg constant

# UNIT is for unit transfermation
UNIT = {}
# length
__temp_list = {}
__temp_list["m"] = 1.0E+00
__temp_list["cm"] = 1.0E-02
__temp_list["mm"] = 1.0E-03
__temp_list["mium"] = 1.0E-06  # micrometer
__temp_list["nm"] = 1.0E-09
__temp_list["A"] = 1.0E-10
__temp_list["pm"] = 1.0E-12

for key1 in __temp_list.keys():
    for key2 in __temp_list.keys():
        if key1 != key2:
            UNIT[key1 + '-' + key2] = __temp_list[key1] / __temp_list[key2]

# energy
__temp_list = {}
__temp_list["J"] = 1.0E+00
__temp_list["eV"] = CONST["E_CHARGE"]
__temp_list["meV"] = 1.0E-3 * CONST["E_CHARGE"]
__temp_list["cm^-1"] = CONST["H"] * CONST["C_LIGHT"] / UNIT["cm-m"]
__temp_list["Ry"] = CONST["H"] * CONST["C_LIGHT"] * CONST["RYDBERG"]  # Rydberg
__temp_list["Ha"] = 2.0 * CONST["H"] * \
    CONST["C_LIGHT"] * CONST["RYDBERG"]  # Hartree(a.u.)

for key1 in __temp_list.keys():
    for key2 in __temp_list.keys():
        if key1 != key2:
            UNIT[key1 + '-' + key2] = __temp_list[key1] / __temp_list[key2]

