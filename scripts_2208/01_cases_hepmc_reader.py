import json
import numpy as np
from pathlib import Path
import gc
import glob
import re
import psutil
import tracemalloc
import sys
tracemalloc.start()

# Particle Parameters
neutralinos = [9900016, 9900014, 9900012, 1000023]
neutrinos = [12, 14, 16, 1000022]

# print(active_z, active_r)

destiny = "./data/clean/"
types = ["ZH","WH","TTH"]
tevs = [13]

for type in types[:]:
    for tev in tevs[:]:
        for file_in in sorted(glob.glob(f"./data/raw/run_{type}*{tev}.hepmc"))[:]:
            # Programming Parameters

            base_out = re.search(f'({type}.+)\.', file_in).group(1)
            file_out = destiny + f'recollection_photons-{base_out}.json'

            it = 0
            i = 0
            limit = 2

            it_start = 0
            batch = 10000
            corte_inf = it_start * batch
            corte_sup = corte_inf + batch * 20
            final_part_active = True

            # Action
            df = open(file_in, "r")
            Path(destiny).mkdir(exist_ok=True, parents=True)

            while it < 2:
                df.readline()
                it += 1

            # Initializing values
            data = dict()
            num = 0
            p_scaler = None
            d_scaler = None
            for sentence in df:
            #while i<(limit+20):
                #sentence = df.readline()
                #print(sentence)
                line = sentence.split()
                if num <= corte_inf:
                    holder = {'v':dict(),'a':[],'n5':dict()}
                    tpx = 0
                    tpy = 0
                    if line[0] == 'E':
                        if (num % 500) == 0:
                            print(f'RUNNING: {base_out} ' + f'Event {num}')
                            print(0)
                        num += 1
                    nfile = it_start + 1
                    continue
                elif line[0] == 'E':
                    # num = int(line[1])
                    if num > 0: #Selection of relevant particles/vertices in the last event
                        #print(mpx,mpy)
                        selection = set()
                        data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']}
                        for n5_k, n5_v in holder['n5'].items():
                            #print(n5_k , n5_i)
                            selection.add(n5_k)
                            selection.add(n5_v[-1])
                        for photon in holder['a']:
                            # select only the photons that come from a n5 vertex
                            outg_a = photon[-1]
                            data[num - 1]['a'].append(photon)
                            if outg_a in selection:
                                x, y, z = [d_scaler * ix for ix in holder['v'][outg_a][0:3]]
                                #print(x,y,z)
                                r = np.sqrt(x**2 + y**2)
                            selection.add(outg_a)
                        for vertex in selection:
                            # select only the vertices that have a neutralino as incoming
                            data[num-1]['v'][vertex] = holder['v'][vertex]
                    #print(data)
                    holder = {'v':dict(),'a':[],'n5':dict()}
                    i += 1
                    if (num % 500) == 0:
                        print(f'RUNNING: {base_out} ' + f'Event {num}')
                        print(len(data))
                        print(str(psutil.virtual_memory().percent) + " %")
                    if num == nfile * batch:
                        with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                            json.dump(data, file)
                        print(f'Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}')
                        del data
                        gc.collect()

                        snapshot = tracemalloc.take_snapshot()
                        top_stats = snapshot.statistics('lineno')
                        print("[ Top 10 ]")
                        for stat in top_stats[:10]:
                            print(stat)
                        #(gc.get_objects())
                        #sys.exit()

                        data = dict()
                        holder = {'v': dict(), 'a': [], 'n5': dict()}
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
                        p_scaler = 1/1000
                    if params[1] == 'MM':
                        d_scaler = 1
                    else:
                        d_scaler = 10
                    #print(p_scaler)
                elif line[0] == 'V':
                    outg = int(line[1])
                    info = *[float(x) for x in line[3:6]], int(line[8]) # x,y,z,number of outgoing
                    holder['v'][outg] = list(info)
                    #print(outg)
                elif line[0] == 'P':
                    pid = line[1]
                    pdg = line[2]
                    #Extracting the MET of the event
                    in_vertex = int(line[11])

                    if (abs(int(pdg)) == 22) and (in_vertex == 0):
                        # id = int(line[1])
                        # px, py, pz, E, m = [float(x) for x in line[3:8]]
                        info = int(pid), *[float(x) for x in line[3:8]], outg # px,py,pz,E,m,vertex from where it comes
                        holder['a'].append(list(info))
                    elif abs(int(pdg)) in neutralinos:
                        info = *[float(x) for x in line[3:8]], outg # px,py,pz,E,m,out_vertex
                        holder['n5'][in_vertex] = list(info)
            df.close()

            if final_part_active:
                # Event selection for the last event
                selection = set()
                data[num - 1] = {'params': params, 'v': dict(), 'a': [], 'n5': holder['n5']}
                for n5_k, n5_v in holder['n5'].items():
                    # print(n5_k , n5_i)
                    selection.add(n5_k)
                    selection.add(n5_v[-1])
                for photon in holder['a']:
                    # select only the photons that come from a n5 vertex
                    outg_a = photon[-1]
                    data[num - 1]['a'].append(photon)
                    if outg_a in selection:
                        x, y, z = [d_scaler * ix for ix in holder['v'][outg_a][0:3]]
                        # print(x,y,z)
                        r = np.sqrt(x ** 2 + y ** 2)
                    selection.add(outg_a)
                for vertex in selection:
                    # select only the vertices that have a neutralino as incoming
                    data[num - 1]['v'][vertex] = holder['v'][vertex]

                #print(data[num])
                #print(data.keys())

                with open(file_out.replace('.json', f'-{nfile}.json'), 'w') as file:
                    json.dump(data, file)

                print(f'FINAL {base_out}: Saved til {num - 1} in {file_out.replace(".json", f"-{nfile}.json")}\n')
