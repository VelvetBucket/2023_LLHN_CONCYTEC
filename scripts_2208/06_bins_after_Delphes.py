import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import pandas as pd
from scipy.interpolate import interp1d
from my_funcs import isolation, get_mass_width
from pathlib import Path
import json

ph_eff_zo= pd.read_csv(f'./data/z0_eff_points.csv', delimiter=',', header=None).set_axis(['zorigin','eff'], axis=1)
mu_eff_pt = pd.read_csv('./data/muon_eff_pt.csv',header=0)
el_eff_pt = pd.read_csv('./data/electron_eff_pt.csv',header=0)
el_eff_eta = pd.read_csv('./data/electron_eff_eta.csv',header=0)
zorigin_res= pd.read_csv(f'./data/z0_res_points.csv', delimiter=',', header=None).set_axis(['zorigin','res'], axis=1)
reltof_res= pd.read_csv(f'./data/z0_res_points.csv', delimiter=',', header=None).set_axis(['zorigin','res'], axis=1)
cutflow_path = "./data/clean/cutflow/"

scales = pd.read_csv("./data/cross_section.dat",delimiter="\t",index_col=0,header=None,squeeze=True)

np.random.seed(0)

## For photonresolution
photon_eff_zo = interp1d(ph_eff_zo.zorigin, ph_eff_zo.eff, fill_value=tuple(ph_eff_zo.eff.iloc[[0,-1]]),
                        bounds_error=False)
## For muon efficiency
mu_func = interp1d(mu_eff_pt.pt,mu_eff_pt.eff, fill_value=tuple(mu_eff_pt.eff.iloc[[0,-1]]), bounds_error=False)
## For electron efciency
el_pt_func = interp1d(el_eff_pt.BinLeft,el_eff_pt.Efficiency, fill_value=tuple(el_eff_pt.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')
el_eta_func = interp1d(el_eff_eta.BinLeft,el_eff_eta.Efficiency, fill_value=tuple(el_eff_eta.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')
el_normal_factor = 1/0.85
## For comparing with the Z mass
m_Z = 91.1876 #GeV
Ecell_factor = 0.35
## For photon's z origin resolution
zorigin_res_func = interp1d(zorigin_res.zorigin, zorigin_res.res, fill_value=tuple(zorigin_res.res.iloc[[0,-1]]),
                        bounds_error=False)

## For photon's relative tof resolution
p0_h = 1.962
p1_h = 0.262

p0_m = 3.650
p1_m = 0.223

def t_res(ecell):
    if ecell >= 25:
        resol= np.sqrt((p0_m/ecell)**2 + p1_m**2)
    else:
        resol= min(np.sqrt((p0_h / ecell) ** 2 + p1_h ** 2), 0.57)

    return resol

# For bin classification
z_bins = [0,50,100,200,300,2000.1]
t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,12.1]}
met_bins = [0, 30, 50, np.inf]

destiny_info = f'./data/clean/'
origin = f"./data/clean/"
destiny = f"./data/clean/"
types = ['ZH', "WH", "TTH"]
tevs = [13]

for tev in tevs[:]:

    files_in = glob.glob(origin + f"complete*{types[0]}*{tev}*photons.pickle")
    bases = sorted(list([re.search(f'/.*{types[0]}_(.+)_photons', x).group(1) for x in files_in]))
    #print(bases)
    #sys.exit()
    for base_out in bases[:]:

        bin_matrix = dict()
        for key, t_bin in t_bins.items():
            bin_matrix[key] = np.zeros((len(z_bins) - 1, len(t_bin) - 1, len(met_bins) - 1)).tolist()
        #print(bin_matrix)
        #sys.exit()
        for type in types[:]:

            folder_txt = f"./cases/{tev}/{type}/"
            cutflows = dict()
            scale = scales[type + '_' + base_out]  * 1000 * 0.2 * 139 / 10000
            #print(scale)
            #sys.exit()

            print(f'RUNNING: {base_out} - {type}')

            input_file = origin + f"complete_{type}_{base_out}_photons.pickle";
            photons = pd.read_pickle(input_file)
            leptons = pd.read_pickle(input_file.replace('photons', 'leptons'))
            jets = pd.read_pickle(input_file.replace('photons', 'jets'))
            #print(photons)
            #sys.exit()

            if leptons.size == 0 or photons.size == 0:
                continue
            #print(photons.shape[0])
            ### Aplying resolutions

            ## Z Origin
            photons['zo_smeared'] = \
                photons.apply(lambda row:
                              np.abs(row['z_origin'] + zorigin_res_func(row['z_origin']) * np.random.normal(0, 1)),
                            axis=1)

            ## relative time of flight
            photons['rt_smeared'] = \
                photons.apply(lambda row: row['rel_tof'] + t_res(Ecell_factor * row['E']) * np.random.normal(0, 1), axis=1)

            ### Applying efficiencies

            ## leptons
            leptons.loc[(leptons.pdg==11),'eff_value'] = \
                leptons[leptons.pdg==11].apply(lambda row:
                                               el_normal_factor*el_pt_func(row.pt)*el_eta_func(row.eta), axis=1)
            leptons.loc[(leptons.pdg == 13), 'eff_value'] = \
                leptons[leptons.pdg == 13].apply(lambda row: mu_func(row.pt), axis=1)

            leptons['detected'] = leptons.apply(lambda row: np.random.random_sample() < row['eff_value'], axis=1)

            leptons = leptons[leptons.detected]

            ## photons
            photons['detected'] = \
                photons.apply(lambda row: np.random.random_sample() < photon_eff_zo(row['zo_smeared']), axis=1)
            # print(df[['zo_smeared','detected']])

            photons = photons[photons['detected']]
            #print(photons.shape[0])
            #sys.exit()

            ## Overlapping
            ### Primero electrones
            leptons.loc[(leptons.pdg==11),'el_iso_ph'] = isolation(leptons[leptons.pdg==11],photons,'pt',same=False,dR=0.4)
            leptons = leptons[(leptons.pdg==13)|(leptons['el_iso_ph']==0)]

            ## Luego jets
            jets['jet_iso_ph'] = isolation(jets,photons,'pt',same=False,dR=0.4)
            jets['jet_iso_e'] = isolation(jets, leptons[leptons.pdg==11], 'pt', same=False, dR=0.2)
            jets = jets[jets['jet_iso_e'] + jets['jet_iso_ph']==0]

            ## Electrones de nuevo
            leptons.loc[(leptons.pdg == 11), 'el_iso_j'] = isolation(leptons[leptons.pdg == 11], jets, 'pt', same=False,
                                                                      dR=0.4)
            leptons = leptons[(leptons.pdg == 13) | (leptons['el_iso_j'] == 0)]

            ## Finalmente, muones
            jets['jet_iso_mu'] = isolation(jets, leptons[leptons.pdg == 13], 'pt', same=False, dR=0.01)
            jets = jets[jets['jet_iso_mu'] == 0]

            leptons.loc[(leptons.pdg == 13), 'mu_iso_j'] = isolation(leptons[leptons.pdg == 13], jets, 'pt', same=False,
                                                                     dR=0.4)
            leptons.loc[(leptons.pdg == 13), 'mu_iso_ph'] = isolation(leptons[leptons.pdg == 13], photons, 'pt', same=False,
                                                                     dR=0.4)
            leptons = leptons[(leptons.pdg == 11) | ((leptons['mu_iso_j'] + leptons['mu_iso_ph']) == 0)]

            ##### De ahí leptones con pt > 27
            leptons = leptons[leptons.pt > 27]
            #print(leptons)

            ### Invariant mass
            photons0 = photons.groupby(['N']).nth(0)
            photons0['px'] = photons0.pt * np.cos(photons0.phi)
            photons0['py'] = photons0.pt * np.sin(photons0.phi)
            photons0['pz'] = photons0.pt / np.tan(2 * np.arctan(np.exp(photons0.eta)))
            photons0 = photons0[['E', 'px', 'py', 'pz']]

            leptons0 = leptons.groupby(['N']).nth(0)
            leptons0['px'] = leptons0.pt * np.cos(leptons0.phi)
            leptons0['py'] = leptons0.pt * np.sin(leptons0.phi)
            leptons0['pz'] = leptons0.pt / np.tan(2 * np.arctan(np.exp(leptons0.eta)))
            leptons0['E'] = np.sqrt(leptons0.mass**2 + leptons0.pt**2 + leptons0.pz**2)
            leptons0 = leptons0[['E','px','py','pz','pdg']]

            final_particles = photons0.join(leptons0,how='inner',lsuffix='_ph',rsuffix='_l')
            final_particles['M_eg'] = np.sqrt((final_particles.E_ph + final_particles.E_l) ** 2 -
                            ((final_particles.px_ph + final_particles.px_l) ** 2 +
                             (final_particles.py_ph + final_particles.py_l) ** 2 +
                             (final_particles.pz_ph + final_particles.pz_l) ** 2))
            final_particles = final_particles[(final_particles.pdg == 13) | (np.abs(final_particles.M_eg - m_Z) > 15)]

            photons = photons[photons.index.get_level_values(0).isin(
                    list(final_particles.index.get_level_values(0)))]

            ### CLaasifying in channels
            ph_num = photons.groupby(['N']).size()
            dfs = {'1': photons.loc[ph_num[ph_num == 1].index], '2+': photons.loc[ph_num[ph_num > 1].index]}

            for channel, phs in dfs.items():
                ## Keeping the most energetic
                phs = phs.groupby(['N']).nth(0)
                ## Filtering Ecell, zorigin and reltof
                phs = phs[(Ecell_factor * phs['E']) > 10]
                phs = phs[phs['zo_smeared'] < 2000]
                phs = phs[(0 < phs['rt_smeared']) & (phs['rt_smeared'] < 12)]
                ## Classifying in bins
                phs['z_binned'] = np.digitize(phs['zo_smeared'], z_bins) - 1
                phs['t_binned'] = np.digitize(phs['rt_smeared'], t_bins[channel]) - 1
                phs['met_binned'] = np.digitize(phs['MET'], met_bins) - 1
                #print(phs[phs['z_binned'] == 5])
                ixs, tallies = np.unique(phs[['z_binned','t_binned','met_binned']].values,
                                return_counts=True, axis=0)
                #print(ixs, tallies)
                for ix, tally in zip(ixs, tallies):
                    z, t, met = ix
                    bin_matrix[channel][z][t][met] += tally * scale

        with open(destiny + f'bin_matrices-{base_out}.json', 'w') as file:
            json.dump(bin_matrix, file)