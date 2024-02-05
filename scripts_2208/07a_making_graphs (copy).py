import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import numpy as np

destiny = f"./data/"

deltas = ['01','15']
met_labels = ['BKG', 'CR', 'SR']
vround = np.vectorize(round)
colores = {'60':'r','50':'g','40':'b','30':'m'}
k_factors = {'ZH':1.491,'WH':1.253,'TTH':1.15}

for delta in deltas[:]:
    origin = f"./data/matrices_{delta}GeV/"
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)

    for mass in ['5','8']:
        burrito = []
        input_files = list(reversed(sorted(glob.glob(origin + f"bin_*M{mass}_*Alpha5_*.json"))))
        for input_file in input_files:
            process = re.search(f'/.*_13-(.*).json', input_file).group(1)
            with open(input_file, 'r') as file:
                cofre = json.load(file)
            cofre = {key: k_factors[process]*np.asarray(matrix) for key, matrix in cofre.items()}
            if burrito == []:
                burrito = cofre
            else:
                burrito = {key: burrito[key] + cofre[key] for key in cofre.keys()}
            #print(burrito)
        #sys.exit()
        norm = sum([x[:,:,-1].sum() for x in burrito.values()])
        burrito = {key: value[:,:,-1]/norm for key, value in burrito.items()}
        #print(sum([x.sum() for x in burrito.values()]))
        #sys.exit()

        ymax_p = []
        ymin_p = []

        for key, matrix in burrito.items():
            nbins = np.array(range(matrix.shape[1] + 1)) + 0.5
            ix = int(key[0]) - 1
            ir = 0
            #print(nbins)
            #print(matrix[ir][:,-1])
            #sys.exit()
            for row in axs:
                row[ix].hist(nbins[:-1],
                             bins=nbins, weights=matrix[ir], histtype='step', label=f'Mass {mass}')
                row[ix].set_yscale('log')
                row[ix].set_xticks(np.array(range(matrix.shape[1])) + 1)
                row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
                row[ix].legend()
                row[ix].secondary_yaxis("right")
                ymax_p.append(row[ix].get_ylim()[1])
                ymin_p.append(row[ix].get_ylim()[0])
                ir += 1

    plt.setp(axs, ylim=(10**(-4),0.4))
    plt.suptitle(f'Alpha 5 - Delta {delta}')
    #plt.show()
    fig.savefig(destiny + f'validation_graphs-Delta{delta}.png')
    plt.close()