from pathlib import Path
import sys
import os
import glob
from multiprocessing import Pool


def main(in_file):
    out_file = in_file.replace('.hepmc', '.root').replace(origin, destiny)
    os.system(f'cd {sys.argv[2]} && ./DelphesHepMC2 '
                f'{root}/Delphes_cards/delphes_card_LLHNscanV5.tcl {out_file} {in_file}')
    return
destiny_base = './data/clean'
types = ['ZH', "WH", "TTH"]
tevs = [13]

root = sys.argv[1]
origin = root + f"/scripts_2208/data/raw/"
destiny = root + f"/scripts_2208/data/clean/"

Path(destiny).mkdir(exist_ok=True, parents=True)
os.system(f'cd {destiny} && find . -name \*.root -type f -delete')

allcases = []
for typex in types[:]:
    for tevx in tevs[:]:
        for file_inx in sorted(glob.glob(origin + f"complete_{typex}*{tevx}.hepmc"))[:]:
            allcases.append(file_inx)

if __name__ == '__main__':
    with Pool(1) as pool:
        pool.map(main, allcases)


