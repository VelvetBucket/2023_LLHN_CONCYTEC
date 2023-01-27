from pathlib import Path
import pandas as pd
import sys
import os
import glob
import re

types = ['ZH', "WH", "TTH"]
tevs = [13]

for tev in tevs[:]:
    for type in types[:]:

        root = "/home/cristian/Desktop/HEP_Jones/paper_2023"
        origin = root + f"/scripts_2208/data/bins/{tev}/{type}/"
        destiny = root + f"/scripts_2208/data/bins/{tev}/{type}/"

        Path(destiny).mkdir(exist_ok=True, parents=True)
        os.system(f'rm {destiny}*.root')
        os.system(f'rm {destiny}*.pickle')

        for in_file in [f for f in sorted(glob.glob(origin + f'*.hepmc')) if 'pos' in f][:]:
            #print(in_file)
            #continue
            out_file = in_file.replace('.hepmc','.root').replace(origin, destiny)
            os.system('cd /home/cristian/Programs/MG5_aMC_v2_9_2/Delphes && ./DelphesHepMC '
                      f'{root}/Delphes_cards/delphes_card_LLHNscanV4test.tcl {out_file} {in_file}')
            #print(zbns)