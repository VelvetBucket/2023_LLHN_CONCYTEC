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
destiny = f'./data_test/clean/'
types = ['ZH', "WH", "TTH"]
tevs = [13]
cutflow_path = "./data_test/clean/cutflow/"
img_path = "./data_test/clean/graphs/"

Path(cutflow_path).mkdir(parents=True, exist_ok=True)
Path(img_path).mkdir(parents=True, exist_ok=True)

for tev in tevs[:]:
    lepton_cut = []
    for type in types[:]:

        folder_txt = f"./cases/{tev}/{type}/"
        folder_ims = f"./cases/{tev}/{type}/ims/"
        Path(folder_ims).mkdir(parents=True, exist_ok=True)

        for photons_in in sorted(glob.glob(f"./data_test/clean/photon_df-{type}*.pickle"))[:]:

            base_out = re.search(f'({type}.+)\.', photons_in).group(1)
            Path(folder_txt + base_out).mkdir(parents=True, exist_ok=True)

            print(f'RUNNING: {base_out}')

            leptons_in = f'./data_test/clean/lepton_df-{base_out}.pickle'
            #jets_in = f'./data_test/clean/jets-{base_out}.pickle'

            df_phs = pd.read_pickle(photons_in)
            #df_jets = pd.read_pickle(jets_in)
            df_leps = pd.read_pickle(leptons_in)

            row_titles = ['initial']
            cutflow = [{'# events':len(df_phs.index.get_level_values(0).unique()),
                       '% of total': np.nan, '% of last': np.nan}]

            ##### See lepton trigger
            #stats = get_mass_width(base_out)
            lepcut_row = {'type':base_out,
                         'initial #': len(df_phs.index.get_level_values(0).unique())}
            lepcut_row['lepton trigger'] = len(df_leps[df_leps.pt > 27].index.get_level_values(0).unique())
            lepcut_row['lepton trigger %'] = 100 * lepcut_row['lepton trigger']/lepcut_row['initial #']
            lepton_cut.append(lepcut_row)
            #sys.exit()

            # LEADING LEPTON PT
            suplim = 99
            events = df_leps.loc[df_leps.groupby('Event')['pt'].idxmax()]
            # print(events.pt)
            color = events[events.pt > 27]
            black = events[events.pt <= 27]

            bins_color = np.linspace(27, suplim, 10)
            binwidth = bins_color[1] - bins_color[0]
            #sys.exit()
            bins_black = np.flip(np.arange(27, min(black.pt) - binwidth, - binwidth))

            plt.hist(color.pt.clip(upper=suplim), bins=bins_color, color=f'blue')
            plt.hist(black.pt.clip(upper=suplim), bins=bins_black, color='k')
            plt.title(f'{base_out}\nLeading lepton PT')
            plt.savefig(img_path + 'leadingLeptonPT_'+base_out+'.png')
            plt.close()

            # LEADING PHOTON PT
            suplim = 99
            events = df_phs.loc[df_phs.groupby('Event')['pt'].idxmax()]

            color = events[events.pt > 10]
            black = events[events.pt <= 10]
            bins_color = np.linspace(10, suplim, 10)
            binwidth = bins_color[1] - bins_color[0]
            #print(black)
            bins_black = np.flip(np.arange(10, min(black.pt,default=10) - binwidth, - binwidth))

            plt.hist(color.pt.clip(upper=suplim), bins=bins_color, color=f'orange')
            plt.hist(black.pt.clip(upper=suplim), bins=bins_black, color='k')
            plt.title(f'{base_out}\nLeading photon PT')
            #plt.show()
            #sys.exit()
            plt.savefig(img_path + 'leadingPhotonPT_'+base_out+'.png')
            plt.close()

            # LEADING PROMPT PHOTON PT
            suplim = 99
            events = df_phs[np.abs(df_phs.z_origin) < 1]
            events = events.loc[events.groupby('Event')['pt'].idxmax()]

            color = events[events.pt > 10]
            black = events[events.pt <= 10]
            bins_color = np.linspace(10, suplim, 10)
            binwidth = bins_color[1] - bins_color[0]
            # print(black)
            bins_black = np.flip(np.arange(10, min(black.pt, default=10) - binwidth, - binwidth))

            plt.hist(color.pt.clip(upper=suplim), bins=bins_color, color=f'green')
            plt.hist(black.pt.clip(upper=suplim), bins=bins_black, color='k')
            plt.title(f'{base_out}\nLeading prompt photon PT')
            # plt.show()
            # sys.exit()
            plt.savefig(img_path + 'leadingPromptPhotonPT_' + base_out + '.png')
            plt.close()
            ###### ENd lepton trigger
            '''
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
            phs_main = isolation(phs_main, phs_surr, "pt", "pt_ph", True)
            phs_main = isolation(phs_main, df_leps[np.abs(df_leps.pdg) == 11], "pt", "pt_e")

            phs_main = phs_main[(phs_main["pt_ph"] + phs_main["pt_e"]) < 0.065 * phs_main.pt]

            ##### Then, the pt of leptons and jets
            df_leps = df_leps[df_leps.pt > 1]
            phs_main = isolation(phs_main, df_jets, "pt", "pt_j")
            phs_main = isolation(phs_main, df_leps, "pt", "pt_l")

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

            df = df[np.abs(df['zo_smeared']) < 2000]

            evflows = len(df.index.get_level_values(0).unique())
            row_titles.append('|zo smeared| < 2000')
            cutflow.append({'# events': evflows,
                            '% of total': 100 * evflows / cutflow[0]['# events'],
                            '% of last': 100 * evflows / cutflow[-1]['# events']})
            
            ### Apply efficiency dependent ofz_origin
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
            '''

    pd.DataFrame(lepton_cut).to_excel(cutflow_path + 'lepton_trigger_cut.xlsx')
