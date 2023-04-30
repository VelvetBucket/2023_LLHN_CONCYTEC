from pathlib import Path
import pandas as pd
import sys
import os
import glob
import re

destiny_base = './data/clean'
types = ['ZH', "WH", "TTH"]
tevs = [13]

root = sys.argv[1]
origin = root + f"/scripts_2208/data/raw/"
destiny = root + f"/scripts_2208/data/clean/"

Path(destiny).mkdir(exist_ok=True, parents=True)
os.system(f'cd {destiny} && find . -name \*.root -type f -delete')

for tev in tevs[:]:
    for type in types[:]:

        for in_file in sorted(glob.glob(origin + f"complete_{type}*{tev}.hepmc"))[:]:

            out_file = in_file.replace('.hepmc','.root').replace(origin, destiny)
            os.system(f'cd {sys.argv[2]} && ./DelphesHepMC2 '
                      f'{root}/Delphes_cards/delphes_card_LLHNscanV5.tcl {out_file} {in_file}')
