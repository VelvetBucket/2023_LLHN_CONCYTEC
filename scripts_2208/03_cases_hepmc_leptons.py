import gc
import json
import numpy as np
from pathlib import Path
import glob
import re

destiny = "./data/clean/"
types = ['ZH', "WH", "TTH"]
tevs = [13]

# Particle Parameters
neutralinos = [9900016, 9900014, 9900012]
neutrinos = [12, 14, 16, 18]
cleptons = [11, 13]

for type in types[:]:
    for tev in tevs[:]:
        for file_in in sorted(glob.glob(f"./data/raw/run_{type}*{tev}.hepmc")):

            base_out = re.search(f'({type}.+)\.', file_in).group(1)
            file_out = destiny + "recollection_leptons-" + base_out + ".json"
            # Programming Parameters

            it = 0
            i = 0
            limit = 2

            it_start = 0
            batch = 500
            corte_inf = it_start * batch
            corte_sup = corte_inf + batch * 2
            final_part_active = True

            # Action
            df = open(file_in, "r")
            Path(destiny).mkdir(exist_ok=True, parents=True)

            while it < 2:
                df.readline()
                it += 1

            # Initializing values
            data = dict()
            codes = []
            num = 0
            p_scaler = None
            d_scaler = None

            for sentence in df:
            #while i<(limit+1000):
            #    sentence = df.readline()
                # print(sentence)
                line = sentence.split()
                if num <= corte_inf:
                    holder = {'v': dict(), 'l': []}
                    if line[0] == 'E':
                        if (num % 500) == 0:
                            print(f'RUNNING: {base_out} ' + f'Event {num}')
                            print(len(data))
                        num += 1
                    nfile = it_start + 1
                    continue
                elif line[0] == 'E':
                    #print(num)
                    # num = int(line[1])
                    if num > 0:  # Selection of relevant particles/vertices in the last event
                        # print(mpx,mpy)
                        selection = set()
                        data[num - 1] = {'params': params, 'v': dict(), 'l': []}
                        for photon in holder['l']:
                            # SIgnal which vertex give charged leptons
                            outg_a = photon[-1]
                            data[num - 1]['l'].append(photon)
                            selection.add(outg_a)
                        for vertex in selection:
                            # select only the vertices that give charged leptons
                            data[num - 1]['v'][vertex] = holder['v'][vertex]
                    # print(data)
                    holder = {'v': dict(), 'l': []}
                    i += 1
                    if (num % 500) == 0:
                        print(f'RUNNING: {base_out} ' + f'Event {num}')
                        print(len(data))
                    if num == nfile * batch:
                        with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                            json.dump(data, file)
                        print(f'Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}')
                        del data
                        data = dict()
                        nfile += 1
                    if num == corte_sup:
                        final_part_active = False
                        break
                    num += 1
                elif line[0] == 'U':
                    params = line[1:]
                    if params[0] == 'GEV':
                        p_scaler = 1
                    else:
                        p_scaler = 1 / 1000
                    if params[1] == 'MM':
                        d_scaler = 1
                    else:
                        d_scaler = 10
                    # print(p_scaler)
                elif line[0] == 'V':
                    outg = int(line[1])
                    info = *[float(x) for x in line[3:6]], int(line[8])  # x,y,z,number of outgoing
                    holder['v'][outg] = list(info)
                    # print(outg)
                elif line[0] == 'P':
                    pdg = line[2]
                    if (abs(int(pdg)) in cleptons) and (line[8] == '1'):
                        info = pdg, *[float(x) for x in line[3:8]], outg  # px,py,pz,E,m,vertex from where it comes
                        holder['l'].append(list(info))
                    codes.append(pdg)
            df.close()

            if final_part_active:
                # Event selection for the last event
                selection = set()
                data[num - 1] = {'params': params, 'v': dict(), 'l': []}
                for photon in holder['l']:
                    # SIgnal which vertex give charged leptons
                    outg_a = photon[-1]
                    data[num - 1]['l'].append(photon)
                    selection.add(outg_a)
                for vertex in selection:
                    # select only the vertices that give charged leptons
                    data[num - 1]['v'][vertex] = holder['v'][vertex]

                # print(data[num])
                # print(data.keys())

                with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                    json.dump(data, file)

                print(f'RUNNING: {base_out} ' + f'info saved in {file_out.replace(".json", f"-{nfile}.json")}')