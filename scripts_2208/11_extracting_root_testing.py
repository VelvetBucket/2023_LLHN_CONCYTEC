import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ROOT
from pathlib import Path
import numpy as np

print('ROOT FIRST ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
#print('ROOT SECON ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
print('DELPHES CLASSES   :',ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"'))
print('EXRROT TREE READER:',ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"'))

types = ['ZH', "WH", "TTH"]
tevs = [13]

for type in types[:]:
        for tev in tevs[:]:

            origin = f"./data/bins/{tev}/{type}/"
            destiny = f"./data/bins/{tev}/{type}/"
            destiny_im = f"./cases/{tev}/{type}/ims/only_Delphes/"

            Path(destiny_im).mkdir(exist_ok=True, parents=True)

            for input_file in sorted(glob.glob(origin + f"*.root"))[:]:

                out_file = input_file.replace('.root','_photons.pickle')

                # Create chain of root trees
                chain = ROOT.TChain("Delphes")
                chain.Add(input_file)
    
                # Create object of class ExRootTreeReader
                treeReader = ROOT.ExRootTreeReader(chain)
                numberOfEntries = treeReader.GetEntries()
    
                # Get pointers to branches used in this analysis
                met = treeReader.UseBranch("MissingET")
                branchPhoton = treeReader.UseBranch("Photon")
                branchJet = treeReader.UseBranch("Jet")
                branchElectron = treeReader.UseBranch("Electron")
                branchMuon = treeReader.UseBranch("Muon")
                branchETE = treeReader.UseBranch("InitialElectrons")
                branchECAL = treeReader.UseBranch("ElectronFilter")
                branchEF = treeReader.UseBranch("ElectronIso")
                branchECalIso = treeReader.UseBranch("ElectronCaloIso")
                branchEIso = treeReader.UseBranch("ElectronEff")

                # Loop over all events
                photons = []
                jets = []
                leptons = []
                lens = []
                print(f"\n{input_file}\nNumber of Entries: {numberOfEntries}")
                for entry in range(numberOfEntries):
                    # Load selected branches with data from specified event
                    treeReader.ReadEntry(entry)
                    miss = met[0].MET

                    branchETEe = []
                    branchEFe = []
                    branchECALe = []
                    branchECalIsoe = []
                    branchEIsoe = []

                    for maybeE in branchETE:
                        if maybeE.PT > 10:
                            branchETEe.append(maybeE)
                    for maybeE in branchECAL:
                        if maybeE.PT > 10:
                            branchECALe.append(maybeE)
                    for maybeE in branchEF:
                        if maybeE.PT > 10:
                            branchEFe.append(maybeE)
                    for maybeE in branchECalIso:
                        if maybeE.PT > 10:
                            branchECalIsoe.append(maybeE)
                    for maybeE in branchEIso:
                        if maybeE.PT > 10:
                            branchEIsoe.append(maybeE)
                    lens.append({'N':entry, "InitialElectrons": len(branchETEe), 'ElectronFilter': len(branchECALe),
                           'ElectronIso': len(branchEFe), 'ElectronCaloIso': len(branchECalIsoe),
                            'ElectronEff': len(branchEIsoe)})
                    #print(branchPhoton, branchElectron, branchMuon)
                    for ph in branchPhoton:
                        #print(ph.PT, ph.Eta)
                        if ph.PT > 10 and (abs(ph.Eta) < 1.37 or 1.52 < abs(ph.Eta) < 2.37):
                            #print(ph.Eta)
                            photons.append({"N": entry, "E":ph.E, "pt":ph.PT, "eta":ph.Eta, 'phi': ph.Phi,
                                            'rel_tof': ph.T, 'MET': miss})

                    for jet in branchJet:
                        if jet.PT > 25:
                            y = np.log((jet.PT * np.sinh(jet.Eta) + np.sqrt(jet.Mass**2 +
                                (jet.PT * np.cosh(jet.Eta))**2)) / (np.sqrt(jet.Mass**2 + jet.PT**2)))
                            if abs(y) < 4.4:
                                jets.append({"N": entry, "pt": jet.PT, "eta": jet.Eta, 'phi': jet.Phi,
                                             'M': jet.Mass, 'MET': miss})
                    #and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.47)
                    for e in branchElectron:
                        if e.PT > 10:
                            leptons.append({"N": entry, 'pdg': 11, "pt":e.PT,
                                            "eta":e.Eta, 'phi': e.Phi, 'MET': miss})
                    # and abs(mu.Eta) < 2.7
                    for mu in branchMuon:
                        if mu.PT > 10:
                            leptons.append({"N": entry, 'pdg': 13, "pt": mu.PT,
                                            "eta": mu.Eta, 'phi': mu.Phi, 'MET': miss})

                df = pd.DataFrame(photons)
                df_jets = pd.DataFrame(jets)
                df_leps = pd.DataFrame(leptons)
                df_lens = pd.DataFrame(lens)
                if (df.shape[0] == 0) or (df_leps.shape[0] == 0):
                    #print(df.shape,df_leps.shape)
                    continue
                if df_jets.shape[0] == 0:
                    df_jets = pd.DataFrame(columns=["N", "pt", "eta", 'phi','M', 'MET'])
                    #print(df_jets)

                df = df.sort_values(by=['N', 'pt'], ascending=[True, False])
                g = df.groupby('N', as_index=False).cumcount() + 1
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
                #print(df_leps)

                df_lens = df_lens.set_index(['N'])
                df_lens.to_pickle(out_file.replace('_photons', '_succesion2'))
                #print(df_lens)
                if 'All' in input_file:
                    nbins = 50

                    #plt.scatter(np.abs(df.eta),df.rel_tof)
                    #plt.savefig(destiny_im + f'reltof_all.jpg')
                    #plt.show()
                    #plt.close()
                    #sys.exit()

                    plt.hist(df.eta, bins=nbins)
                    plt.savefig(destiny_im + f'eta_all.jpg')
                    #plt.show()
                    plt.close()

                    plt.hist(df.pt, bins=nbins)
                    plt.savefig(destiny_im + f'PT_all.jpg')
                    #plt.show()
                    plt.close()

                    plt.hist(df.loc[pd.IndexSlice[:,1],'MET'], bins=nbins)
                    plt.savefig(destiny_im + f'MET_all.jpg')
                    #plt.show()
                    plt.close()

