# -*- coding: utf-8 -*-
"""
24/02/2026
@author: callum
"""
import os
import glob
import itertools
import pickle
import zipfile
import numpy as np
import pandas as pd
import constants
import solvers
import stats
import light
import utils
import genetic_algorithm as ga

file_dir = os.path.join("out", "anox_rc_1_configs")
filelist = glob.glob(os.path.join(file_dir, "configs_*"))
solutions = np.full(1000 * len(filelist), np.nan, dtype=ga.gt)
start_index = 0
for ii, file in enumerate(filelist):
    with open(file, "rb") as f:
        tmp = pickle.load(f)
    end_index = start_index + tmp.shape[0]
    solutions[start_index:end_index] = tmp
    start_index = end_index

# ugly but i do not care
solutions = solutions[:end_index]
de_uniq = np.unique(solutions['dE0'])
ip_uniq = np.unique(solutions['i_p'])
de_uniq_nm = utils.ev_nm(de_uniq)

print(f"unique energies: {de_uniq}")
print(f"energies in nm: {de_uniq_nm}")
print(f"unique ionisation potentials: {ip_uniq}")

param_dict = {}
for ue in de_uniq:
    mask = solutions['dE0'] == ue
    this = solutions[mask.flat]
    uniqs = np.unique(this['i_p'])
    param_dict[ue] = {}
    for ui in uniqs:
        imask = this['i_p'] == ui
        param_dict[ue][ui] = np.sort(this[imask.flat], order='output')

with open(os.path.join(file_dir, "for_analysis.txt"), "w") as f:
    for ue, un in zip(de_uniq, de_uniq_nm):
        for ui in ip_uniq:
            if ui in param_dict[ue]:
                f.write(f"dE0 = {ue} ({un} nm), i_p = {ui}\n")
                for nt_row, out in zip(param_dict[ue][ui]['e_t'],
                                       param_dict[ue][ui]['output']):
                    f.write(f"e_t = {nt_row} , output = {out}\n")
                f.write("\n")
