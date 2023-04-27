from pathlib import Path
import sys
import glob
import re
import pandas as pd
import psutil
import tracemalloc
tracemalloc.start()

destiny = "./data/raw/"
types = ["ZH","WH","TTH"]
tevs = [13]

for type in types[:]:
    for tev in tevs[:]:
        for file_in in sorted(glob.glob(f"./data/raw/run_{type}*{tev}.hepmc"))[1:]:
            # Programming Parameters

            base_out = re.search(f'({type}.+)\.', file_in).group(1)
            file_out = destiny + f'complete_{base_out}.hepmc'

            print(f'\nRUNNING: {base_out}')

            new_observs = pd.read_pickle(f'./data/clean/photon_df-{base_out}.pickle')

            keys = new_observs[['Event','pid']].values.tolist()
            events = new_observs.Event.values
            pids = new_observs.pid.values
            #print(new_observs)

            new_observs = new_observs.set_index(['Event','pid'])
            #print()
            #sys.exit()
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
                            print(str(psutil.virtual_memory().percent) + " %")
                            snapshot = tracemalloc.take_snapshot()
                            top_stats = snapshot.statistics('lineno')
                            print("[ Top 5 ]")
                            for stat in top_stats[:5]:
                                print(stat)
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

