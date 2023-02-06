import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import pandas as pd
from scipy.interpolate import interp1d
from my_funcs import isolation, get_mass_width

from pathlib import Path

mu_eff_pt = pd.read_csv('./data/muon_eff_pt.csv',header=0)
el_eff_pt = pd.read_csv('./data/electron_eff_pt.csv',header=0)
el_eff_eta = pd.read_csv('./data/electron_eff_eta.csv',header=0)
cutflow_path = "./data/clean/cutflow/"
el_normal_factor = 1/0.85
m_Z = 91.1876 #GeV

np.random.seed(0)
mu_func = interp1d(mu_eff_pt.pt,mu_eff_pt.eff, fill_value=tuple(mu_eff_pt.eff.iloc[[0,-1]]), bounds_error=False)
el_pt_func = interp1d(el_eff_pt.BinLeft,el_eff_pt.Efficiency, fill_value=tuple(el_eff_pt.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')
el_eta_func = interp1d(el_eff_eta.BinLeft,el_eff_eta.Efficiency, fill_value=tuple(el_eff_eta.Efficiency.iloc[[0,-1]]),
                      bounds_error=False,kind='zero')

types = ['ZH', "WH", "TTH"]
tevs = [13]

for tev in tevs[:]:
    mismatches = []
    for type in types[:]:
        origin = f"./data/bins/{tev}/{type}/"
        destiny = f"./data/bins/{tev}/{type}/"
        files_in = glob.glob(origin + f"*{type}*{tev}*_photons.pickle")
        print(files_in)
        bases = sorted(list(set([re.search(f'/.*({type}.+)-df', x).group(1) for x in files_in])))
        cutflows = dict()

        for base_out in bases[:]:

            destiny_info = f'./data/clean/'
            folder_txt = f"./cases/{tev}/{type}/"
            mismatch_data = {'1_total': 0, '1_to_2+': 0,'2+_total': 0, '2+_to_1': 0}
            dfs = {'1': '','2+': ''}
            scale = 1.

            print(f'RUNNING: {base_out}')

            for file_in in sorted(glob.glob(origin + f'*{base_out}*_photons.pickle'))[:]:
                photons = pd.read_pickle(file_in)
                leptons = pd.read_pickle(file_in.replace('photons', 'leptons'))
                jets = pd.read_pickle(file_in.replace('photons', 'jets'))
                delphesCuts = pd.read_pickle(file_in.replace('photons', 'DelphesCuts'))

                #print(photons)

                ### ver cuales no cumplen con su label
                if 'df1' in file_in:
                    nums = photons.groupby(['N']).size()
                    mismatch_data['1_total'] += nums.size
                    mismatch_data['1_to_2+'] += nums[nums > 1].size
                elif 'df2+' in file_in:
                    nums = photons.groupby(['N']).size()
                    mismatch_data['2+_total'] += nums.size
                    mismatch_data['2+_to_1'] += nums[nums == 1].size

                #print(mismatch_data)

                ##### ADD Delphes to the cutflow
                if 'All' in file_in:

                    prev_cutflow = pd.read_excel(cutflow_path + f'cutflow_{type}_part2.xlsx', sheet_name=base_out,
                                                 index_col=0)
                    cutflow = prev_cutflow.to_dict(orient='records')
                    row_titles = list(prev_cutflow.index)
                    # First Photons
                    for dlph_col, dlph_value in delphesCuts.loc[:,'Photon'].iteritems():
                        row_titles.append('Dlph -  Photon ' + dlph_col)
                        event_flow = dlph_value[dlph_value > 0].size
                        cutflow.append({'# events': event_flow,
                                        '% of total': 100 * event_flow / cutflow[0]['# events'],
                                        '% of last': 100 * event_flow / cutflow[-1]['# events']})
                    # Now, leptons
                    delphesCuts = delphesCuts[dlph_value > 0]
                    delphesCuts = delphesCuts.loc[:,['Electron','Muon']]
                    # Because we are not using the Isolation modules with Muons....
                    delphesCuts[('Muon','Isolation')] = delphesCuts[('Muon','Efficiency')]
                    delphesCuts[('Muon', 'CaloIsolation')] = delphesCuts[('Muon','Efficiency')]
                    dlphCutsLeps = delphesCuts.sum(level=1, axis=1)

                    for dlph_col, dlph_value in dlphCutsLeps.iteritems():
                        row_titles.append('Dlph -  Lepton ' + dlph_col)
                        event_flow = dlph_value[dlph_value > 0].size
                        cutflow.append({'# events': event_flow,
                                        '% of total': 100 * event_flow / cutflow[0]['# events'],
                                        '% of last': 100 * event_flow / cutflow[-1]['# events']})
                    #print(row_titles)
                    #print(cutflow)

                ### Keeping only the events with leptons AND photons
                leptons = leptons[leptons.index.get_level_values(0).isin(list(photons.index.get_level_values(0)))]

                if 'All' in file_in:
                    event_flow = leptons.index.get_level_values(0).unique().size
                    #print(event_flow)
                    #sys.exit()
                    row_titles.append('Delphes Lepton/Photon')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})

                if leptons.size == 0:
                    continue
                #sys.exit()
                ### Applying efficiencies
                leptons.loc[(leptons.pdg==11),'eff_value'] = \
                    leptons[leptons.pdg==11].apply(lambda row:
                                                   el_normal_factor*el_pt_func(row.pt)*el_eta_func(row.eta), axis=1)

                leptons.loc[(leptons.pdg == 13), 'eff_value'] = \
                    leptons[leptons.pdg == 13].apply(lambda row: mu_func(row.pt), axis=1)

                leptons['detected'] = leptons.apply(lambda row: np.random.random_sample() < row['eff_value'], axis=1)

                leptons = leptons[leptons.detected]
                #print(leptons)

                ##### ADDto  the cutflow
                if 'All' in file_in:
                    event_flow = len(
                        np.intersect1d(photons.index.get_level_values(0), leptons.index.get_level_values(0)))
                    row_titles.append('Lepton eff.')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})
                    print(leptons.groupby('pdg').size())
                ## Overlapping

                ### Primero electrones
                leptons.loc[(leptons.pdg==11),'el_iso_ph'] = isolation(leptons[leptons.pdg==11],photons,'pt',same=False,dR=0.4)
                leptons = leptons[(leptons.pdg==13)|(leptons['el_iso_ph']==0)]

                ##### ADDto  the cutflow
                if 'All' in file_in:
                    event_flow = leptons.index.get_level_values(0).unique().size
                    row_titles.append('Overlap removal 1')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})

                ## Luego jets
                jets['jet_iso_ph'] = isolation(jets,photons,'pt',same=False,dR=0.4)
                jets['jet_iso_e'] = isolation(jets, leptons[leptons.pdg==11], 'pt', same=False, dR=0.2)
                jets = jets[jets['jet_iso_e'] + jets['jet_iso_ph']==0]

                ## Electrones de nuevo
                leptons.loc[(leptons.pdg == 11), 'el_iso_j'] = isolation(leptons[leptons.pdg == 11], jets, 'pt', same=False,
                                                                          dR=0.4)
                leptons = leptons[(leptons.pdg == 13) | (leptons['el_iso_j'] == 0)]
                ##### ADDto  the cutflow
                if 'All' in file_in:
                    event_flow = leptons.index.get_level_values(0).unique().size
                    row_titles.append('Overlap removal 2')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})

                ## Finalmente, muones
                jets['jet_iso_mu'] = isolation(jets, leptons[leptons.pdg == 13], 'pt', same=False, dR=0.01)
                jets = jets[jets['jet_iso_mu']== 0]

                leptons.loc[(leptons.pdg == 13), 'mu_iso_j'] = isolation(leptons[leptons.pdg == 13], jets, 'pt', same=False,
                                                                         dR=0.4)
                leptons.loc[(leptons.pdg == 13), 'mu_iso_ph'] = isolation(leptons[leptons.pdg == 13], photons, 'pt', same=False,
                                                                         dR=0.4)
                leptons = leptons[(leptons.pdg == 11) | ((leptons['mu_iso_j'] + leptons['mu_iso_ph']) == 0)]

                #sys.exit()
                ##### ADDto  the cutflow
                if 'All' in file_in:
                    print(leptons.groupby('pdg').size())
                    event_flow = leptons.index.get_level_values(0).unique().size
                    row_titles.append('Overlap removal 3')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})

                ##### De ahí leptones con pt > 27
                leptons = leptons[leptons.pt > 27]
                #print(leptons)

                ##### ADD to  the cutflow
                if 'All' in file_in:
                    event_flow = leptons.index.get_level_values(0).unique().size
                    row_titles.append('lep pt > 27')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})
    
                ### Invariant mass
                photons = photons.groupby(['N']).nth(0)
                photons['px'] = photons.pt * np.cos(photons.phi)
                photons['py'] = photons.pt * np.sin(photons.phi)
                photons['pz'] = photons.pt / np.tan(2 * np.arctan(np.exp(photons.eta)))
                photons = photons[['E', 'px', 'py', 'pz']]

                leptons = leptons.groupby(['N']).nth(0)
                leptons['px'] = leptons.pt * np.cos(leptons.phi)
                leptons['py'] = leptons.pt * np.sin(leptons.phi)
                leptons['pz'] = leptons.pt / np.tan(2 * np.arctan(np.exp(leptons.eta)))
                leptons['E'] = np.sqrt(leptons.mass**2 + leptons.pt**2 + leptons.pz**2)
                leptons = leptons[['E','px','py','pz','pdg']]

                final_particles = photons.join(leptons,how='inner',lsuffix='_ph',rsuffix='_l')
                final_particles['M_eg'] = np.sqrt((final_particles.E_ph + final_particles.E_l) ** 2 -
                                ((final_particles.px_ph + final_particles.px_l) ** 2 +
                                 (final_particles.py_ph + final_particles.py_l) ** 2 +
                                 (final_particles.pz_ph + final_particles.pz_l) ** 2))
                #print(final_particles.groupby('pdg').size())
                final_particles = final_particles[(final_particles.pdg==13)|(np.abs(final_particles.M_eg - m_Z) > 15)]
                #print(final_particles.groupby('pdg').size())
                ##### ADD to  the cutflow
                if 'All' in file_in:
                    event_flow = len(final_particles)
                    row_titles.append('|M_eg - M_Z| > 15 GeV')
                    cutflow.append({'# events': event_flow,
                                    '% of total': 100 * event_flow / prev_cutflow.iloc[0]['# events'],
                                    '% of last': 100 * event_flow / cutflow[-1]['# events']})
                    cutflow3 = pd.DataFrame(cutflow, index=row_titles)
                    cutflows[base_out] = cutflow3

            stats = get_mass_width(base_out)

            mismatches.append({'type':type, 'mass': stats['M'], 'width': stats['W'],
                '1_total': mismatch_data['1_total'],
                '1_to_2+': np.around(mismatch_data['1_to_2+']/mismatch_data['1_total'],4),
                '2+_total': mismatch_data['2+_total'],
                '2+_to_1': np.around(mismatch_data['2+_to_1']/mismatch_data['2+_total'],4)})

            #print(mismatches)
            #sys.exit()

        print('hey')
        with pd.ExcelWriter(cutflow_path + f'cutflow_{type}_part3.xlsx') as writer:
            for base, cuts in cutflows.items():
                cuts.to_excel(writer, sheet_name=base)

    pd.DataFrame(mismatches).to_excel(cutflow_path + 'photon_df_mismataches.xlsx')
