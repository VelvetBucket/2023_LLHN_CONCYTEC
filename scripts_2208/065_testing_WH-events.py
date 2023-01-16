import json
import numpy as np
from pathlib import Path
import gc
import glob
import re
destiny = './data/raw/'
types = ['ZH','WH']
tevs = [13]

Path(destiny).mkdir(exist_ok=True, parents=True)

for type in types[:]:
    for tev in tevs[:]:
        for file_in in sorted(glob.glob(f"./data/raw/run_{type}*{tev}.hepmc"))[:1]:

            # Programming Parameters

            base_out = re.search(f'({type}.+)\.', file_in).group(1)

            # Action
            file_out = open(destiny + f'events_wo_leppt27-{base_out}.hepmc', 'w')
            df = open(file_in, "r")
            it = 0

            while it <= 2:
                file_out.write(df.readline())
                it += 1

            ix = 0
            n_events = 0
            skip = True
            event_holder = []
            for sentence in df:
                # while i<(limit+20):
                # sentence = df.readline()
                # print(sentence)
                line = sentence.split()
                if line[0] == 'E':

                    if ix > 0:
                        if not skip:
                            file_out.write(''.join(event_holder))
                            n_events += 1

                    line[1] = str(ix)
                    sentence = ' '.join(line) + '\n'
                    print(f'RUNNING: {base_out} ' + f'Event {ix}')
                    ix += 1
                    event_holder = []
                    skip = False
                    
                elif line[0] == 'P':
                    pdg = int(line[2])
                    in_vertex = int(line[11])
                    if (abs(pdg) in [11, 13]) and (line[8] == '1'):
                        pt = np.sqrt(float(line[3])**2 + float(line[4])**2)
                        if pt > 27:
                            skip = True

                event_holder.append(sentence)

            if not skip:
                file_out.write(''.join(event_holder))
                n_events += 1

            print(n_events)

            df.close()
            file_out.close()
