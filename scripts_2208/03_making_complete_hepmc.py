from pathlib import Path
import sys
import glob
import re
import pandas as pd
from multiprocessing import Pool


def main(parameters):

    file_in, type = parameters

    base_out = re.search(f'({type}.+)\.', file_in).group(1)
    file_out = destiny + f'complete_{base_out}.hepmc'

    print(f'\nRUNNING: {base_out}')

    new_observs = pd.read_pickle(f'./data/clean/photon_df-{base_out}.pickle')

    keys = new_observs[['Event','pid']].values.tolist()
    new_observs = new_observs.set_index(['Event','pid'])

    hepmc = open(file_in, 'r')
    new_hepmc = open(file_out, 'w')

    event = -1
    for sentence in hepmc:
        zorigin = 0.0
        relt = 0.0
        line = sentence.split()
        if len(line) > 0:
            if line[0] == 'E':
                event += 1
                if (event % 100) == 0:
                    print(f'{base_out}: Event {event}')
                #print(event)
            elif line[0] == 'P':
                pid = int(line[1])
                if (abs(int(line[2])) == 22) and (int(line[11]) == 0) and keys.count([event, pid]) == 1:
                    #print([event, pid])
                    this = new_observs.loc[(event,pid)]
                    zorigin = this.z_origin
                    relt = this.rel_tof
                line.insert(13, str(relt))
                line.insert(13, str(zorigin))
                sentence = ' '.join(line) + '\n'
        new_hepmc.write(sentence)

    hepmc.close()
    new_hepmc.close()
    return


destiny = "./data/raw/"
types = ["ZH","WH","TTH"]
tevs = [13]

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(f"./data/raw/run_{typex}*{tevx}.hepmc"))[:]:
            allcases.append([file_inx, typex])

if __name__ == '__main__':
    with Pool() as pool:
        pool.map(main, allcases)