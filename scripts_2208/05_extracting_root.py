import sys
import os
import pandas as pd
import glob
import ROOT
from pathlib import Path
import numpy as np
import re

print('ROOT FIRST ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
print('DELPHES CLASSES   :',ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"'))
print('EXRROT TREE READER:',ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"'))

types = ['ZH', "WH", "TTH"]
tevs = [13]

for type in types[:]:
        for tev in tevs[:]:

            origin = f"./data/clean/"
            destiny = f"./data/clean/"

            for input_file in sorted(glob.glob(origin + f"*.root"))[:]:

                out_file = input_file.replace('.root','_photons.pickle')

                # Create chain of root trees
                chain = ROOT.TChain("Delphes")
                chain.Add(input_file)

                # Create object of class ExRootTreeReader
                treeReader = ROOT.ExRootTreeReader(chain)
                numberOfEntries = treeReader.GetEntries()

                met = treeReader.UseBranch("MissingET")
                branchPhoton = treeReader.UseBranch("Photon")
                branchJet = treeReader.UseBranch("Jet")
                branchElectron = treeReader.UseBranch("Electron")
                branchMuon = treeReader.UseBranch("Muon")

                # Loop over all events
                photons = []
                jets = []
                leptons = []
                print(f"Number of Entries: {numberOfEntries}")
                for entry in range(numberOfEntries):
                    # Load selected branches with data from specified event
                    treeReader.ReadEntry(entry)
                    miss = met[0].MET
                    #print(entry0)

                    #print(branchPhoton, branchElectron, branchMuon)
                    for ph in branchPhoton:
                        #print(ph.PT, ph.Eta)
                        if ph.PT > 10 and (abs(ph.Eta) < 1.37 or 1.52 < abs(ph.Eta) < 2.37):
                            #print(ph.Eta)
                            photons.append({"N": entry, "E":ph.E, "pt":ph.PT, "eta":ph.Eta, 'phi': ph.Phi,
                                            'z_origin': ph.ZOrigin, 'rel_tof': ph.RelativeT,'MET': miss})

                    for jet in branchJet:
                        if jet.PT > 25:
                            y = np.log((jet.PT * np.sinh(jet.Eta) + np.sqrt(jet.Mass**2 +
                                (jet.PT * np.cosh(jet.Eta))**2)) / (np.sqrt(jet.Mass**2 + jet.PT**2)))
                            if abs(y) < 4.4:
                                jets.append({"N": entry, "pt": jet.PT, "eta": jet.Eta, 'phi': jet.Phi})

                    for e in branchElectron:
                        if e.PT > 10 and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.47):
                            leptons.append({"N": entry, 'pdg': 11, "pt":e.PT,
                                            "eta":e.Eta, 'phi': e.Phi, 'mass': 0.000511})

                    for mu in branchMuon:
                        if mu.PT > 10 and abs(mu.Eta) < 2.7:
                            leptons.append({"N": entry, 'pdg': 13, "pt": mu.PT,
                                            "eta": mu.Eta, 'phi': mu.Phi, 'mass': 0.10566})

                #input_file.close()
                chain.Clear()

                df = pd.DataFrame(photons)
                df_jets = pd.DataFrame(jets)
                df_leps = pd.DataFrame(leptons)

                if (df.shape[0] == 0) or (df_leps.shape[0] == 0):
                    print(df.shape,df_leps.shape)
                    continue
                if df_jets.shape[0] == 0:
                    df_jets = pd.DataFrame(columns=["N", "pt", "eta", 'phi','M', 'MET'])
                    #print(df_jets)

                df = df.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df.groupby('N', as_index=False).cumcount()
                df['id'] = g
                df = df.set_index(['N', 'id'])
                print(f'{100 * df.index.unique(0).size / numberOfEntries:2f} %')
                df.to_pickle(out_file)

                df_jets = df_jets.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df_jets.groupby('N', as_index=False).cumcount()
                df_jets['id'] = g
                df_jets = df_jets.set_index(['N', 'id'])
                df_jets.to_pickle(out_file.replace('_photons','_jets'))

                df_leps = df_leps.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df_leps.groupby('N', as_index=False).cumcount()
                df_leps['id'] = g
                df_leps = df_leps.set_index(['N', 'id'])
                df_leps.to_pickle(out_file.replace('_photons', '_leptons'))
                print(df)
