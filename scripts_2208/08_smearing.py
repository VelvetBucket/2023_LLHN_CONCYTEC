from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from my_funcs import get_scale
import sys
import re
import glob

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

def isolation(phs_o, surr, obs, label, photons=False):
    phs = phs_o.copy()
    phs[label] = 0
    for ix in phs.index.get_level_values(0).unique()[:]:
        event_ph = phs.loc[ix]
        try:
            event_surr = surr.loc[ix]
            # print(event_surr)
            for index_ph, row_ph in event_ph.iterrows():
                # print(row_ph)
                cone = 0
                for index_d, row_d in event_surr.iterrows():
                    dr = np.sqrt((row_d.phi - row_ph.phi) ** 2 + (row_d.eta - row_ph.eta) ** 2)
                    if photons and (index_ph == index_d):
                        dr = 1000
                    if dr < 0.2:
                        cone += row_d[obs]
                phs.at[(ix, index_ph), label] = cone
        except KeyError:
            continue

    return phs

types = ['ZH', "WH", "TTH"]
tevs = [13]

z_bins = [0,50,100,200,300,2000.1]
t_bins = {'1': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1], '2+': [0,0.2,0.4,0.6,0.8,1.0,1.5,12.1]}

bin_matrix = dict()
bin_matrix_neg = dict()
lost_ev = dict()
cutflow_path = "./data/clean/cutflow/"

np.random.seed(0)

for tev in tevs[:]:
    for type in types[:]:
        files_in = glob.glob(f"./data/clean/df_photon_*-{type}*{tev}.pickle")
        bases = sorted(list(set([re.search(f'({type}.+)\.', x).group(1) for x in files_in])))
        #print(bases)

        with pd.ExcelWriter(cutflow_path + f'cutflow_{type}_part2.xlsx') as writer:

            for base_out in bases[:]:

                row_titles = []
                cutflow = []

                for key, tbin in t_bins.items():
                    bin_matrix[key] = np.zeros((len(z_bins) - 1, len(tbin) - 1))
                    bin_matrix_neg[key] = np.zeros((len(z_bins) - 1, len(tbin) - 1))

                lost_ev[base_out] = dict()

                folder_ims = f"./cases/{tev}/{type}/ims/"
                #paper_ims = f"./paper/{tev}/{type}/{card}/"
                folder_txt = f"./cases/{tev}/{type}/"
                #paper_txt = f"./paper/{tev}/{type}/"
                destiny_info = f'./data/clean/'
                dfs = {'1': '','2+': ''}
                dfs_neg = {'1': '', '2+': ''}

                Path(folder_ims).mkdir(parents=True, exist_ok=True)
                Path(folder_txt).mkdir(parents=True, exist_ok=True)
                Path(folder_txt + base_out).mkdir(parents=True, exist_ok=True)
                #Path(paper_txt).mkdir(parents=True, exist_ok=True)
                #Path(paper_ims).mkdir(parents=True, exist_ok=True)

                print(f'RUNNING: {base_out}')
                scale = 1
                ix = 0
                for key in dfs.keys():

                    lost_ev[base_out][key] = []

                    dfs[key] = pd.read_pickle(f'./data/clean/df_photon_{key}-{base_out}.pickle')

                    ########## z_origin

                    plt.hist(dfs[key]['z_origin'], bins=60, color=f'C{ix}')
                    plt.xlabel('z_origin (with efficiency) [mm]')
                    plt.savefig(folder_ims + f'{base_out}_{key}_z_origin.png')
                    #plt.show()
                    plt.close()

                    plt.hist(dfs[key]['zo_smeared'], bins=50, color=f'C{ix}')
                    plt.xlabel('z_origin Smeared Cutted (with efficiency) [mm]')
                    plt.savefig(folder_ims + f'{base_out}_{key}_z_origin_smeared_cutted.png')
                    # plt.show()
                    plt.close()

                    ########### ret_tof ##############
                    dfs[key]['rt_smeared'] = \
                        dfs[key].apply(lambda row: row['rel_tof'] + t_res(0.35 * row['E']) * np.random.normal(0, 1), axis=1)

                    plt.hist(dfs[key]['rel_tof'], bins=50, color=f'C{ix}')
                    plt.xlabel('t_gamma [ns]')
                    plt.savefig(folder_ims + f'{base_out}_{key}_rel_tof.png')
                    # plt.show()
                    plt.close()

                    plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                    plt.xlabel('t_gamma Smeared [ns]')
                    plt.savefig(folder_ims + f'{base_out}_{key}_rel_tof_smeared.png')
                    # plt.show()
                    plt.close()

                    dfs_neg[key] = dfs[key][(0 > dfs[key]['rt_smeared']) & (dfs[key]['rt_smeared'] >= -12)]
                    dfs[key] = dfs[key][(0 <= dfs[key]['rt_smeared']) & (dfs[key]['rt_smeared'] <= 12)]
                    lost_ev[base_out][key].append(dfs[key].shape[0])

                    plt.hist(dfs[key]['rt_smeared'], bins=50, color=f'C{ix}')
                    plt.xlabel('t_gamma Smeared Cutted [ns]')
                    plt.savefig(folder_ims + f'{base_out}_{key}_rel_tof_smeared_cutted.png')
                    plt.close()
                    ############

                    dfs[key]['t_binned'] = np.digitize(dfs[key]['rt_smeared'], t_bins[key])
                    dfs[key]['z_binned'] = np.digitize(np.abs(dfs[key]['zo_smeared']),z_bins)

                    dfs_neg[key]['t_binned'] = np.digitize(np.abs(dfs_neg[key]['rt_smeared']), t_bins[key])
                    dfs_neg[key]['z_binned'] = np.digitize(np.abs(dfs_neg[key]['zo_smeared']), z_bins)

                    for ind, row in dfs[key].iterrows():
                        #print(ind)
                        bin_matrix[key][row['z_binned'] - 1, row['t_binned'] - 1] +=1
                        #print(row['z_binned'], row['t_binned'])

                    for ind, row in dfs_neg[key].iterrows():
                        #print(ind)
                        bin_matrix_neg[key][row['z_binned'] - 1, row['t_binned'] - 1] +=1
                    #print(bin_matrix_neg)
                    ix += 1
                '''
                ixevents = np.unique(pd.concat(dfs.values()).index.get_level_values(0))
                leptons = pd.read_pickle(destiny_info+f'lepton_df-{base_out}.pickle')
                leptons = leptons.loc[np.intersect1d(ixevents,leptons.index.get_level_values(0))]
    
                #### Setting surround particles
                jets = pd.read_pickle(destiny_info + f'jets-{base_out}.pickle')
                photons = pd.read_pickle(destiny_info + f'photon_df-{base_out}.pickle')
    
                jets_surr = jets[jets.pt > 1]
                leps_surr = leptons[leptons.pt > 1]
                phs_surr = photons[(photons.r < 1) & (np.abs(photons.z) < 1) & (photons.pt > 1)]
    
                ## Defining dicts
                notisodict = dict()
                isodict = dict()
    
                for pdg, color in [(11, 'blue'), (13, 'purple')][:]:
                    flavourlep = leptons[(np.abs(leptons.pdg) == pdg)]
                    leading = flavourlep.groupby(['Event']).nth(0)
                    subleading = flavourlep.groupby(['Event']).nth(1)
                    for col in ['pt','eta']:
                        binmax = np.amax(np.concatenate((leading[col].values, subleading[col].values)))
                        binmin = np.amin(np.concatenate((leading[col].values, subleading[col].values)))
                        bins = np.linspace(binmin, binmax, 25)
                        fig, axs = plt.subplots(ncols=2, sharey=True)
                        axs[0].hist(leading[col], bins=bins,color=color)
                        axs[0].set_title('leading')
                        axs[1].hist(subleading[col],bins=bins, color=color)
                        axs[1].set_title('subleading')
                        plt.suptitle(col)
                        fig.savefig(folder_txt + f'{base_out}/{col}Dist_lep{pdg}_preDelphes.png')
                        #plt.show()
                        plt.close()
                    ### Making DeltaR plot
                    #print(leading.shape)
                    leading = leading.loc[subleading.index]
                    #print(leading.shape)
                    plt.hist(np.sqrt(((leading.phi - subleading.phi)**2 +
                                      (leading.eta - subleading.eta)**2).astype(np.float64)),
                             bins=25,color=color)
                    plt.suptitle('Delta R')
                    plt.savefig(folder_txt + f'{base_out}/deltaRDist_lep{pdg}_preDelphes.png')
                    #plt.show()
                    plt.close()
    
                    flavourlep = flavourlep[flavourlep.pt > 10]
                    ### leptons without isolation
                    notisolep = flavourlep.groupby(['Event']).size()
                    notisolep = notisolep.reindex(sorted(ixevents),fill_value=0)
                    notisodict[pdg] = notisolep
                    #print(notisolep)
    
                    ### leptons with isolation
                    isolep = isolation(flavourlep, leps_surr, "pt", "pt_l", photons=True)
                    isolep = isolation(isolep, jets_surr, "pt", "pt_j")
                    isolep = isolation(isolep, phs_surr, "pt", "pt_p")
                    isolep = isolep[(isolep['pt_l'] + isolep['pt_j'] + isolep['pt_p']) == 0.0]
                    isolep = isolep.groupby(['Event']).size()
                    isolep = isolep.reindex(sorted(ixevents), fill_value=0)
                    isodict[pdg] = isolep
                    #print(isolep)
    
                pieces = [(notisodict, 'leptons pt > 10 without isolation', 'WOutIso'),
                          (isodict, 'leptons pt > 10 with isolation', 'Wiso')]
                for lepsdict, title, filename in pieces:
                    maxlep = np.max([np.max(x) for x in lepsdict.values()]) + 2
    
                    fig, axs = plt.subplots(ncols=2,sharey=True)
                    axs[0].hist(lepsdict[11], bins=range(maxlep), color='blue')
                    axs[0].set_title('# electrons')
                    axs[1].hist(lepsdict[13], bins=range(maxlep),color='purple')
                    axs[1].set_title('# muons')
                    plt.setp(axs,xticks=np.arange(0.5,maxlep + 0.5,1), xticklabels=np.arange(0,maxlep,1))
                    plt.suptitle(title)
                    fig.savefig(folder_txt + f"{base_out}/numberOfLeptons{filename}_preDelphes.png")
                    #plt.show()
                    plt.close()
                #sys.exit()
                '''
                #print(bin_matrix)
                fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
                plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
                ymax = ymin = []
                for key in bin_matrix.keys():
                    nbins = np.array(range(bin_matrix[key].shape[1] + 1)) + 0.5
                    ix = int(key[0])-1
                    ir = 0
                    for row in axs:
                        row[ix].hist(nbins[:-1], bins=nbins, weights=bin_matrix[key][ir]*scale, histtype='step')
                        row[ix].set_yscale('log')
                        row[ix].set_xticks(np.array(range(bin_matrix[key].shape[1])) + 1)
                        row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                        ymax.append(row[ix].get_ylim()[1])
                        ymin.append(row[ix].get_ylim()[0])
                        ir+=1
                plt.setp(axs, ylim=(min(ymin), max(ymax)))
                fig.savefig(folder_ims + f'{base_out}_zbins_tbins_pos.png')
                #fig.savefig(paper_ims + f'{type}_{base_out}_zbins_tbins_BeforeDelphes_pos.png')
                #plt.show()
                plt.close()

                fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
                plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
                ymax = ymin = []
                for key in bin_matrix_neg.keys():
                    nbins = np.array(range(bin_matrix_neg[key].shape[1] + 1)) + 0.5
                    ix = int(key[0]) - 1
                    ir = 0
                    for row in axs:
                        row[ix].hist(nbins[:-1], bins=nbins, weights=bin_matrix_neg[key][ir] * scale, histtype='step',
                                     color='C1')
                        row[ix].set_yscale('log')
                        row[ix].set_xticks(np.array(range(bin_matrix_neg[key].shape[1])) + 1)
                        row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                        ymax.append(row[ix].get_ylim()[1])
                        ymin.append(row[ix].get_ylim()[0])
                        ir += 1
                plt.setp(axs, ylim=(min(ymin), max(ymax)))
                fig.savefig(folder_ims + f'{base_out}_zbins_tbins_neg.png')
                #fig.savefig(paper_ims + f'{type}_{base_out}_zbins_tbins_BeforeDelphes_neg.png')
                # plt.show()
                plt.close()
                #sys.exit()

                ##### ADD rt cutflow
                cutflow1 = pd.read_excel(cutflow_path + f'cutflow_{type}_part1.xlsx', sheet_name=base_out, index_col=0)

                rt_smeared = sum([len(x.index.get_level_values(0).unique()) for x in dfs.values()])
                row_titles.append('0 < rt_smeared < 12')
                cutflow.append({'# events': rt_smeared,
                                '% of total': 100 * rt_smeared / cutflow1.iloc[0]['# events'],
                                '% of last': 100 * rt_smeared / cutflow1.iloc[-1]['# events']})
                cutflow2 = cutflow1.append(pd.DataFrame(cutflow, index=row_titles))
                cutflow2.to_excel(writer, sheet_name=base_out)
                #sys.exit()

                for key in dfs.keys():
                    print(f'{key} dataset: {dfs[key].shape}')
                    dfs[key].to_pickle(f'./data/clean/df_photon_smeared_{key}-{base_out}_pos.pickle')
                for key in dfs_neg.keys():
                    #print(dfs_neg[key])
                    dfs_neg[key].to_pickle(f'./data/clean/df_photon_smeared_{key}-{base_out}_neg.pickle')
                print('dfs saved!')






