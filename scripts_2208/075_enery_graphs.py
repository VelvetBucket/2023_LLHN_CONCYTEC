import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
import glob
from my_funcs import my_arctan
from scipy.interpolate import interp1d
from my_funcs import get_mass_width

destiny = f'./data/clean/'
types = ['ZH','WH','TTH']
tevs = [13]
cutflow_path = "./data/clean/cutflow/"
Path(cutflow_path).mkdir(parents=True, exist_ok=True)

for tev in tevs[:]:
    for type in types[:]:

        folder_txt = f"./cases/{tev}/{type}/"
        folder_ims = f"./cases/{tev}/{type}/ims/"
        Path(folder_ims).mkdir(parents=True, exist_ok=True)

        for photons_in in sorted(glob.glob(f"./data/clean/photon_df-{type}*{tev}.pickle"))[:1]:
            base_out = re.search(f'({type}.+)\.', photons_in).group(1)
            Path(folder_txt + base_out).mkdir(parents=True, exist_ok=True)

            print(f'RUNNING: {base_out}')

            leptons_in = f'./data/clean/lepton_df-{base_out}.pickle'
            df_leps = pd.read_pickle(leptons_in)
            df_leps = df_leps.loc[df_leps.groupby('Event')['pt'].idxmax()]

            fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(12,6))
            for ix, lepton in enumerate([11,-11,13,-13]):
                events = df_leps[(df_leps.pdg == lepton)]
                suplim = 99
                #print(events.pt)
                color = events[events.pt>27]
                black = events[events.pt <= 27]

                bins_color=np.linspace(27,suplim,10)

                bins_black=np.flip(np.arange(27,min(black.pt) - 8,-8))
                axs[0,ix].hist(color.pt.clip(upper=suplim),bins=bins_color,color=f'C{ix}')
                axs[0, ix].hist(black.pt.clip(upper=suplim),bins=bins_black, color='k')
                axs[0, ix].set_title(f'{lepton} PT')

                bins_black = np.flip(np.arange(27, min(black.E) - 8, -8))
                axs[1, ix].hist(color.E.clip(upper=suplim), bins=bins_color, color=f'C{ix}')
                axs[1, ix].hist(black.E.clip(upper=suplim), bins=bins_black, color='k')
                axs[1, ix].set_title(f'{lepton} E')
            plt.suptitle(f'{type} - leading lepton distribution')
            #plt.show()
            #sys.exit()
            fig.savefig(folder_txt + f'{type}-l1_dist.png')
            plt.close()



