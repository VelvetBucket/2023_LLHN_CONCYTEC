import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
import glob
from my_funcs import my_arctan
from scipy.interpolate import interp1d
from my_funcs import get_mass_width, isolation

points = pd.read_csv(f'./data/z0_res_points.csv', delimiter=',', header=None).values
linear = interp1d(points[:, 0], points[:, 1])

points_eff = pd.read_csv(f'./data/z0_eff_points.csv', delimiter=',', header=None).values
linear_eff = interp1d(points_eff[:, 0], points_eff[:, 1])


def z_res(z0):
    z = abs(z0)
    if z < points[0, 0]:
        resol = points[0, 1]
    elif z < points[-1, 0]:
        resol = linear(z)
    else:
        resol = points[-1, 1]

    return resol


def zo_eff(z0):
    z = abs(z0)
    if z < points_eff[0, 0]:
        eff = points_eff[0, 1]
    elif z < points_eff[-1, 0]:
        eff = linear_eff(z)
    else:
        eff = points_eff[-1, 1]

    return eff

np.random.seed(0)
destiny = f'./data/clean/'
types = ['ZH', "WH", "TTH"]
tevs = [13]
cutflow_path = "./data/clean/cutflow/"
Path(cutflow_path).mkdir(parents=True, exist_ok=True)

for tev in tevs[:]:
    lepton_cut = []
    for type in types[:]:

        folder_txt = f"./cases/{tev}/{type}/"
        folder_ims = f"./cases/{tev}/{type}/ims/"
        Path(folder_ims).mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(cutflow_path + f'cutflow_{type}_part1.xlsx') as writer:

            for photons_in in sorted(glob.glob(f"./data/clean/photon_df-{type}*{tev}.pickle"))[:]:

                base_out = re.search(f'({type}.+)\.', photons_in).group(1)
                Path(folder_txt + base_out).mkdir(parents=True, exist_ok=True)

                print(f'RUNNING: {base_out}')

                '''
                ########################
                ######### START Leptom distributions
                #########################
                photons = pd.read_pickle(destiny + f'photon_df-{base_out}.pickle')
                ixevents = np.unique(photons.index.get_level_values(0))
    
                leptons = pd.read_pickle(destiny + f'lepton_df-{base_out}.pickle')
                leptons = leptons.loc[np.intersect1d(ixevents, leptons.index.get_level_values(0))]
    
                #### Setting surround particles
                jets = pd.read_pickle(destiny + f'jets-{base_out}.pickle')
    
                jets_surr = jets[jets.pt > 20]
                leps_surr = leptons[leptons.pt > 1]
                phs_surr = photons[(photons.r < 1) & (np.abs(photons.z) < 1) & (photons.pt > 1)]
    
                ## Defining dicts
                notisodict = dict()
                isodict = dict()
    
                for pdg, color in [(11, 'yellow'), (13, 'red')][:]:
                    flavourlep = leptons[(np.abs(leptons.pdg) == pdg)]
                    leading = flavourlep.groupby(['Event']).nth(0)
                    subleading = flavourlep.groupby(['Event']).nth(1)
                    for col in ['pt', 'eta']:
                        binmax = np.amax(np.concatenate((leading[col].values, subleading[col].values)))
                        binmin = np.amin(np.concatenate((leading[col].values, subleading[col].values)))
                        bins = np.linspace(binmin, binmax, 25)
                        fig, axs = plt.subplots(ncols=2, sharey=True)
                        axs[0].hist(leading[col], bins=bins, color=color)
                        axs[0].set_title('leading')
                        axs[1].hist(subleading[col], bins=bins, color=color)
                        axs[1].set_title('subleading')
                        plt.suptitle(col)
                        fig.savefig(folder_txt + f'{base_out}/{col}Dist_lep{pdg}_allPhotons-preDelphes.png')
                        #plt.show()
                        plt.close()
                    ### Making DeltaR plot
                    print(leading.shape)
                    leading = leading.loc[subleading.index]
                    print(leading.shape)
                    plt.hist(np.sqrt(((leading.phi - subleading.phi) ** 2 +
                                      (leading.eta - subleading.eta) ** 2).astype(np.float64)),
                             bins=25, color=color)
                    plt.suptitle('Delta R')
                    plt.savefig(folder_txt + f'{base_out}/deltaRDist_lep{pdg}_allPhotons-preDelphes.png')
                    #plt.show()
                    plt.close()
    
                    flavourlep = flavourlep[flavourlep.pt > 10]
                    ### leptons without isolation
                    notisolep = flavourlep.groupby(['Event']).size()
                    notisolep = notisolep.reindex(sorted(ixevents), fill_value=0)
                    notisodict[pdg] = notisolep
                    # print(notisolep)
    
                    ### leptons with isolation
                    isolep = isolation(flavourlep, leps_surr, "pt", "pt_l", photons=True)
                    isolep = isolation(isolep, jets_surr, "pt", "pt_j")
                    isolep = isolation(isolep, phs_surr, "pt", "pt_p")
                    isolep = isolep[(isolep['pt_l'] + isolep['pt_j'] + isolep['pt_p']) == 0.0]
                    isolep = isolep.groupby(['Event']).size()
                    isolep = isolep.reindex(sorted(ixevents), fill_value=0)
                    isodict[pdg] = isolep
                    # print(isolep)
    
                pieces = [(notisodict, 'leptons pt > 10 without isolation', 'WOutIso'),
                          (isodict, 'leptons pt > 10 with isolation', 'Wiso')]
                for lepsdict, title, filename in pieces:
                    maxlep = np.max([np.max(x) for x in lepsdict.values()]) + 2
    
                    fig, axs = plt.subplots(ncols=2, sharey=True)
                    axs[0].hist(lepsdict[11], bins=range(maxlep), color='yellow')
                    axs[0].set_title('# electrons')
                    axs[1].hist(lepsdict[13], bins=range(maxlep), color='red')
                    axs[1].set_title('# muons')
                    plt.setp(axs, xticks=np.arange(0.5, maxlep + 0.5, 1), xticklabels=np.arange(0, maxlep, 1))
                    plt.suptitle(title)
                    fig.savefig(folder_txt + f"{base_out}/numberOfLeptons{filename}_allPhotons-preDelphes.png")
                    #plt.show()
                    plt.close()
    
                ########################
                ######### END Leptom distributions
                #########################
                '''
                leptons_in = f'./data/clean/lepton_df-{base_out}.pickle'
                jets_in = f'./data/clean/jets-{base_out}.pickle'

                df_phs = pd.read_pickle(photons_in)
                df_jets = pd.read_pickle(jets_in)
                df_leps = pd.read_pickle(leptons_in)

                row_titles = ['initial']
                cutflow = [{'# events':len(df_phs.index.get_level_values(0).unique()),
                           '% of total': np.nan, '% of last': np.nan}]

                ##### See lepton trigger
                stats = get_mass_width(base_out)
                lepcut_row = {'type':type, 'mass': stats['M'], 'width': stats['W'],
                             'initial #': len(df_phs.index.get_level_values(0).unique())}
                lepcut_row['lepton trigger'] = len(df_leps[df_leps.pt > 27].index.get_level_values(0).unique())
                lepcut_row['lepton trigger %'] = 100 * lepcut_row['lepton trigger']/lepcut_row['initial #']
                lepton_cut.append(lepcut_row)
                #sys.exit()
                ###### ENd lepton trigger

                ## Separating photon objects and photons for surroundings
                phs_main = df_phs.copy()
                phs_surr = df_phs.copy()

                ### photon pt >= 10 GeV{
                phs_main = phs_main[phs_main.pt > 10]
                evflows = len(phs_main.index.get_level_values(0).unique())
                row_titles.append('pt > 10')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                ########### CONTAINED ################
                phs_main = phs_main[(phs_main.r < 1) & (np.abs(phs_main.z) < 1)]

                evflows = len(phs_main.index.get_level_values(0).unique())
                row_titles.append('contained')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                ### photon eta not in 1.37 < eta < 1.52 nor eta > 2.37
                phs_main = phs_main[np.abs(phs_main.eta) < 2.37]
                phs_main = phs_main[(np.abs(phs_main.eta) < 1.37) | (np.abs(phs_main.eta) > 1.52)]

                evflows = len(phs_main.index.get_level_values(0).unique())
                row_titles.append('eta cuts')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                ### Now, photon isolation

                #### First, only elements with pt > 0.1 GeV (20 GeV for jets)
                df_jets = df_jets[df_jets.pt > 20]
                df_leps = df_leps[df_leps.pt > 0.1]
                phs_surr = phs_surr[phs_surr.pt > 0.1]

                #### THen, keep thr pbservables ofonly the objects that are in Delta R < 0.2 of photon objects
                df = phs_main.copy()
                ##### First, the energy of electron and photons
                phs_main["pt_ph"] = isolation(phs_main, phs_surr, "pt", True)
                phs_main["pt_e"] = isolation(phs_main, df_leps[np.abs(df_leps.pdg) == 11], "pt")

                #phs_main = phs_main[(phs_main["pt_ph"] + phs_main["pt_e"]) < 0.065 * phs_main.pt]

                evflows = len(phs_main.index.get_level_values(0).unique())
                row_titles.append('caloisolation')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                ##### Then, the pt of leptons and jets
                df_leps = df_leps[df_leps.pt > 1]
                phs_main["pt_j"] = isolation(phs_main, df_jets, "pt")
                phs_main["pt_l"] = isolation(phs_main, df_leps, "pt")

                phs_main = phs_main[(phs_main["pt_j"] + phs_main["pt_l"]) < 0.05 * phs_main.pt]

                evflows = len(phs_main.index.get_level_values(0).unique())
                row_titles.append('isolation')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                df = phs_main.copy()

                ########## z_origin
                df['zo_smeared'] = \
                    df.apply(lambda row: row['z_origin'] + z_res(row['z_origin']) * np.random.normal(0, 1), axis=1)

                plt.hist(df['z_origin'], bins=60, color=f'C2')
                plt.xlabel('z_origin [mm]')
                plt.savefig(folder_ims + f'{base_out}_All_z_origin.png')
                # plt.show()
                plt.close()

                plt.hist(df['zo_smeared'], bins=50, color=f'C2')
                plt.xlabel('z_origin Smeared [mm]')
                plt.savefig(folder_ims + f'{base_out}_All_z_origin_smeared.png')
                plt.close()
                # plt.show()

                df = df[np.abs(df['zo_smeared']) < 2000]
                # lost_ev[base_out][key].append(dfs[key].shape[0])

                plt.hist(df['zo_smeared'], bins=50, color=f'C2')
                plt.xlabel('z_origin Smeared Cutted [mm]')
                plt.savefig(folder_ims + f'{base_out}_All_z_origin_smeared_cutted.png')
                # plt.show()
                plt.close()

                evflows = len(df.index.get_level_values(0).unique())
                row_titles.append('|zo smeared| < 2000')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                ### Apply efficiency dependent ofz_origin
                #df['detected'] = True
                df['detected'] = \
                    df.apply(lambda row: np.random.random_sample() < zo_eff(row['zo_smeared']), axis=1)
                #print(df[['zo_smeared','detected']])

                df = df[df['detected']]

                evflows = len(df.index.get_level_values(0).unique())
                row_titles.append('photon efficiency')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                # Photons not in the barrel
                ph_num = df.groupby(['Event']).size()
                dfs = {'1': df.loc[ph_num[ph_num == 1].index], '2+': df.loc[ph_num[ph_num > 1].index]}

                evflows = 0
                for key, df_ in dfs.items():
                    df_ = df_[1.52 > np.abs(df_.eta)]
                    dfs[key] = df_
                    evflows += len(df_.index.get_level_values(0).unique())

                row_titles.append('no photon in barrel')
                cutflow.append({'# events': evflows,
                                '% of total': 100 * evflows / cutflow[0]['# events'],
                                '% of last': 100 * evflows / cutflow[-1]['# events']})

                #print(pd.DataFrame(cutflow, index=row_titles))
                #sys.exit()
                pd.DataFrame(cutflow, index=row_titles).to_excel(writer, sheet_name=base_out)
                # print(dfs['2+'][['pt','pz']])
                dfs['2+'] = dfs['2+'].loc[dfs['2+'].groupby('Event')['pt'].idxmax()]

                for key, df_ in dfs.items():
                    df_.to_pickle(destiny + f'df_photon_{key}-{base_out}.pickle')
                print('dfs saved!')

    pd.DataFrame(lepton_cut).to_excel(cutflow_path + 'lepton_trigger_cut.xlsx')
