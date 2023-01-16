import sys
sys.path.insert(0, "/home/cristian/Programs/fastjet-3.4.0/lib/python3.8/site-packages/")

from pathlib import Path
import glob
from fastjet import *
import pandas as pd
import matplotlib.pyplot as plt
import re
from my_funcs import my_arctan

destiny = "./data/clean/"

outputs=["Event","id", "px", "py", "pz", "pt", "eta", "phi", "E"]
types = ['ZH', "WH", "TTH"]

tevs = [13]

for tev in tevs[:]:
    for type in types[:]:

        for file_in in sorted(glob.glob(f"./data/raw/prejets_{type}*{tev}.txt")):

            jet_list = []
            base_out = re.search(f'({type}.+)\.', file_in).group(1)
            file_out = destiny + "jets-" + base_out + ".pickle"

            Path(destiny).mkdir(exist_ok=True, parents=True)
            prej = open(file_in, "r")
            #jets = open(destiny + file_out, 'w')

            R = 0.4
            jetdef = JetDefinition(antikt_algorithm, R, E_scheme, Best)
            i = 0

            #jets.write(f'INPUT: {file_in}\n')
            #jets.write(f"Clustering with {jetdef.description()}\n\n")
            #while i<4 :
                #sentence = prej.readline()
            for sentence in prej:
                line = sentence.split()
                if "Ev" in line[0]:
                    if i > 0:
                        # print(inputs)
                        cluster = ClusterSequence(inputs, jetdef)
                        inc_jets = sorted_by_pt(cluster.inclusive_jets(20.0))
                        for ix, jet in enumerate(inc_jets):
                            jet_list.append([i-1,ix, jet.px(), jet.py(), jet.pz(), jet.pt(), jet.eta(), jet.phi(), jet.E()])
                            #print(jet.phi())
                    print(f"{base_out} Event {i}")

                    i+=1
                    inputs = []
                    #print(inputs)
                elif line[0] == "P":
                    x = 0
                    inputs.append(PseudoJet(*[float(x) for x in line[1:]]))

            #FINAL EVENT
            cluster = ClusterSequence(inputs, jetdef)
            inc_jets = sorted_by_pt(cluster.inclusive_jets(20.0))
            for ix, jet in enumerate(inc_jets):
                jet_list.append([i-1, ix, jet.px(), jet.py(), jet.pz(), jet.pt(), jet.eta(), jet.phi(), jet.E()])
                #print(jet.phi())
                #print(f"{type}_{card}_{tev} Event {i }")

            prej.close()
            jets_df=pd.DataFrame(jet_list, columns=outputs)
            jets_df = jets_df.set_index(["Event","id"])
            #print(jets_df['pt'].min())

            plt.hist(jets_df.pt)
            #plt.show()
            #plt.savefig(destiny + f"{.png")
            plt.close()
            jets_df.to_pickle(file_out)