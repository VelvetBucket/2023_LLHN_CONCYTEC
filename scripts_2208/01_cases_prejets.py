from pathlib import Path
import sys
import glob
import re

bosons = [22]
leptons = [11, 12, 13, 14, 16, 18]
heavy_n = [9900016,9900014,9900012]
vetos = bosons + leptons + heavy_n

destiny = "./data/raw/"
types = ['ZH', "WH", "TTH"]
tevs = [13]

Path(destiny).mkdir(exist_ok=True, parents=True)

for tev in tevs[:]:
    for type in types[:]:

        for file_in in sorted(glob.glob(f"./data/raw/run_{type}*{tev}.hepmc")):

            #print(file_in)
            base_out = re.search(f'({type}.+)\.', file_in).group(1)
            file_out = destiny + "prejets_" + base_out + ".txt"
            #print(file_out)

            df = open(file_in, "r")
            prej = open(file_out, 'w')
            i= 0
            it = 0
            while it < 2:
                df.readline()
                it += 1

            #while i<3 :
            #    sentence = df.readline()
            for sentence in df:
                line = sentence.split()
                if line[0] == "E":
                    prej.write(f"Ev{i} px py pz E\n")
                    print(f"{base_out} Event {i}")
                    i+=1
                elif line[0] == "P":
                    if (abs(int(line[2])) not in vetos) and (line[8] == '1'):
                        data = ' '.join(line[3:7])
                        #print(data)
                        prej.write(f'P {data}\n')

            df.close()
            prej.close()
