import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import numpy as np

origin = f"./data/clean/"
destiny = f"./cases/"

met_labels = ['BKG', 'CR', 'SR']
vround = np.vectorize(round)

for input_file in sorted(glob.glob(origin + f"bin_*.json"))[:]:

    ymax_p = []
    ymin_p = []

    base_out = re.search(f'/.*bin_matrices-(.+)\.json', input_file).group(1)
    print(base_out)

    with open(input_file, 'r') as file:
        burrito = json.load(file)

    burrito = {key: np.asarray(matrix) for key, matrix in burrito.items()}

    for key, matrix in burrito.items():
        nbins = np.array(range(matrix.shape[1] + 1)) + 0.5
        for values in matrix:
            plt.hist(np.tile(nbins[:-1], (matrix.shape[-1], 1)).transpose(),
                     bins=nbins, weights=values, histtype='step',
                     stacked=False, color=['C3', 'C2', 'C0'], label=met_labels)
            plt.yscale('log')
            plt.legend()
            ymax_p.append(plt.gca().get_ylim()[1])
            ymin_p.append(plt.gca().get_ylim()[0])
            # plt.show()
            plt.close()

    events = sum([np.sum(vround(np.sum(item, axis=1))) for item in burrito.values()])
    # sys.exit()
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))
    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.95, wspace=None, hspace=0.3)
    for key, matrix in burrito.items():
        nbins = np.array(range(matrix.shape[1] + 1)) + 0.5
        ix = int(key[0]) - 1
        ir = 0
        for row in axs:
            row[ix].hist(np.tile(nbins[:-1], (matrix.shape[-1], 1)).transpose(),
                         bins=nbins, weights=matrix[ir], histtype='step',
                         stacked=False, color=['C3', 'C2', 'C0'], label=met_labels)
            row[ix].set_yscale('log')
            row[ix].set_xticks(np.array(range(matrix.shape[1])) + 1)
            row[ix].set_title(f'Dataset {key} ph - bin z {ir + 1}')
            row[ix].legend()
            # ymax_p.append(row[ix].get_ylim()[1])
            # ymin_p.append(row[ix].get_ylim()[0])
            ir += 1
    plt.setp(axs, ylim=(min(ymin_p), max(ymax_p)))
    plt.suptitle(f'{base_out} GeV\n{events} events')
    #plt.show()
    fig.savefig(destiny + f'graphs-{base_out}.png')
    plt.close()