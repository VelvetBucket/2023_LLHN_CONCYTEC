import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import pandas as pd
from pathlib import Path

types = ['ZH', "WH", "TTH"]
tevs = [13]

for tev in tevs[:]:
    for type in types[:2]:
        files_in = glob.glob(f"./data/clean/df_photon_*-{type}*{tev}.pickle")
        bases = sorted(list(set([re.search(f'({type}.+)\.', x).group(1) for x in files_in])))
        print(bases)

        for base_out in bases[:3]:

            origin = f"./data/bins/{tev}/{type}/"
            destiny = f"./data/bins/{tev}/{type}/"
            destiny_info = f'./data/clean/'
            folder_txt = f"./cases/{tev}/{type}/"
            dfs = {'1': '','2+': ''}

            print(f'RUNNING: {base_out}')

            df = pd.read_pickle(glob.glob(origin + f'*{base_out}*All*_photons.pickle')[0])
            leptons = pd.read_pickle(glob.glob(origin + f'*{base_out}*All*_leptons.pickle')[0])
            lens = pd.read_pickle(glob.glob(origin + f'*{base_out}*All*_succesion2.pickle')[0])
            #print(glob.glob(origin + f'*{base_out}*All*_leptons.pickle')[0])

            addStr = 'V3test1'
            color = 'red'
            Path(folder_txt + f'{base_out}/electronFlow/').mkdir(parents=True, exist_ok=True)
            for ix, col in enumerate(lens.columns[:]):
                plt.hist(lens[col], bins=range(np.amax(lens[col])+2), color=color)
                plt.title(col)
                plt.savefig(folder_txt + f'{base_out}/electronFlow/{addStr}_{ix}_{col}.png')
                plt.show()
                plt.close()
                #print(col)
            sys.exit()

            ixevents = np.unique(df.index.get_level_values(0))
            leptons = leptons.loc[np.intersect1d(ixevents, leptons.index.get_level_values(0))]

            ## Defining dicts
            notisodict = dict()
            #isodict = dict()

            for pdg, color in [(11, 'blue'), (13, 'purple')][:]:
                flavourlep = leptons[(np.abs(leptons.pdg) == pdg)]
                leading = flavourlep.groupby(['N']).nth(0)
                subleading = flavourlep.groupby(['N']).nth(1)
                for col in ['pt', 'eta']:
                    binmax = np.amax(np.concatenate((leading[col].values, subleading[col].values)))
                    binmin = np.amin(np.concatenate((leading[col].values, subleading[col].values)))
                    bins = np.linspace(binmin, binmax, 25)
                    fig, axs = plt.subplots(ncols=2, sharey=True)
                    axs[0].hist(leading[col], bins=bins,  histtype = 'step', color=color)
                    axs[0].set_title('leading')
                    axs[1].hist(subleading[col], bins=bins, histtype = 'step',  color=color)
                    axs[1].set_title('subleading')
                    plt.suptitle(col)
                    fig.savefig(folder_txt + f'{base_out}/{col}Dist_lep{pdg}_pstDelphes.png')
                    #plt.show()
                    plt.close()
                ### Making DeltaR plot
                leading = leading.loc[subleading.index]
                plt.hist(np.sqrt(((leading.phi - subleading.phi) ** 2 +
                                  (leading.eta - subleading.eta) ** 2).astype(np.float64)),
                         bins=25, color=color,  histtype = 'step')
                plt.suptitle('Delta R')
                plt.savefig(folder_txt + f'{base_out}/deltaRDist_lep{pdg}_pstDelphes.png')
                # plt.show()
                plt.close()

                flavourlep = flavourlep[flavourlep.pt > 10]
                ### leptons without isolation
                notisolep = flavourlep.groupby(['N']).size()
                notisolep = notisolep.reindex(sorted(ixevents), fill_value=0)
                notisodict[pdg] = notisolep

            pieces = [(notisodict, 'leptons with Delphes isolation', 'WIso')]
            for lepsdict, title, filename in pieces:
                maxlep = np.max([np.max(x) for x in lepsdict.values()]) + 2

                fig, axs = plt.subplots(ncols=2, sharey=True)
                axs[0].hist(lepsdict[11], bins=range(maxlep),  histtype = 'step', color='blue')
                axs[0].set_title('# electrons')
                axs[1].hist(lepsdict[13], bins=range(maxlep), histtype = 'step',color='purple')
                axs[1].set_title('# muons')
                plt.setp(axs, xticks=np.arange(0.5, maxlep + 0.5, 1), xticklabels=np.arange(0, maxlep, 1))
                plt.suptitle(title)
                fig.savefig(folder_txt + f"{base_out}/numberOfLeptons{filename}_pstDelphes.png")
                # plt.show()
                plt.close()
            # sys.exit()

