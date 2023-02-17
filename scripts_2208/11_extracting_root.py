import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import ROOT
from pathlib import Path
import numpy as np
import itertools
import re

print('ROOT FIRST ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
#print('ROOT SECON ATTEMPT:',ROOT.gSystem.Load("libDelphes"))
print('DELPHES CLASSES   :',ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"'))
print('EXRROT TREE READER:',ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"'))

types = ['ZH', "WH", "TTH"]
tevs = [13]
ph_stages = ['Initial','ECal','Isolation','CaloIsolation','Efficiency']
stages = ['Initial','Tracking','Isolation','CaloIsolation','Efficiency']
xleptons = ['Electron','Muon']

for type in types[:1]:
        for tev in tevs[:]:

            origin = f"./data/bins/{tev}/{type}/"
            destiny = f"./data/bins/{tev}/{type}/"
            destiny_im = f"./cases/{tev}/{type}/ims/only_Delphes/"

            Path(destiny_im).mkdir(exist_ok=True, parents=True)

            files_in = sorted(glob.glob(origin + f"*.root"))
            '''
            bases= sorted(list(set([re.search(f'/.*({type}.+-df.+)_', x).group(1) for x in files_in])))
            for base_out in bases[:]:
                preDelphes_file = 'df_photon_smeared_'+ base_out +'.pickle'
            '''

            for input_file in sorted(glob.glob(origin + f"*.root"))[:]:

                if 'All' not in input_file:
                    name_base = re.search(f'/.*({type}.+)-df', input_file).group(1)
                    dataset = re.search(f'/.*df(.+?)_', input_file).group(1)
                    tsign = re.search(f'/.*_(.+?)\.', input_file).group(1)
                    z = re.search(f'/.*_z(.+?)_', input_file).group(1)
                    t = re.search(f'/.*_t(.+?)_', input_file).group(1)
                    preDelphes_file = f'./data/clean/df_photon_smeared_{dataset}-{name_base}_{tsign}.pickle'
                    preDelphes = pd.read_pickle(preDelphes_file)
                    #print(preDelphes.loc[[299]])
                    preDelphes = preDelphes[(preDelphes.z_binned == int(z)) & (preDelphes.t_binned == int(t))]
                    preDelphes = sorted(preDelphes.index.get_level_values(0).unique())
                    #print(preDelphes)
                print(input_file)
                out_file = input_file.replace('.root','_photons.pickle')

                # Create chain of root trees
                chain = ROOT.TChain("Delphes")
                chain.Add(input_file)

                # Create object of class ExRootTreeReader
                treeReader = ROOT.ExRootTreeReader(chain)
                numberOfEntries = treeReader.GetEntries()
                # if 'All' not in input_file:
                #     if numberOfEntries == len(preDelphes):
                #         continue
                #     else:
                #         print(numberOfEntries, preDelphes)
                #         sys.exit()
                # Get pointers to branches used in this analysis
                met = treeReader.UseBranch("MissingET")
                branchPhoton = treeReader.UseBranch("Photon")
                branchJet = treeReader.UseBranch("Jet")
                branchElectron = treeReader.UseBranch("Electron")
                branchMuon = treeReader.UseBranch("Muon")

                ##### TESTING BRANCHES
                branchDict = {(l, stage): treeReader.UseBranch(stage + l) for l in xleptons for stage in stages}
                for stage in ph_stages:
                    branchDict[('Photon',stage)] = treeReader.UseBranch(stage + 'Photon')
                stagesList = []
                # Loop over all events
                photons = []
                jets = []
                leptons = []

                print(f"Number of Entries: {numberOfEntries}")
                for entry0 in range(numberOfEntries):

                    # Load selected branches with data from specified event
                    treeReader.ReadEntry(entry0)
                    miss = met[0].MET
                    #print(entry0)
                    if 'All' in input_file:
                        entry = entry0
                    else:
                        entry = preDelphes[entry0]
                        #sys.exit()

                    valuesDict = {key: 0 for key in branchDict.keys()}
                    #print(valuesDict)
                    valuesDict['N'] = entry
                    #print(valuesDict)

                    for key, branch in branchDict.items():
                        lep = key[0]
                        stage = key[1]
                        for e in branch:
                            if lep == 'Electron' and e.PT > 10 and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.47):
                                valuesDict[key] += 1
                            elif lep == 'Muon' and e.PT > 10 and abs(e.Eta) < 2.7:
                                valuesDict[key] += 1
                            elif lep == 'Photon' and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.37):
                                if stage == 'Initial':
                                    if abs(e.PID) == 22 and e.PT > 10:
                                        valuesDict[key] += 1
                                elif stage == 'ECal':
                                    if e.ET > 10:
                                        valuesDict[key] += 1
                                elif e.PT > 10:
                                    valuesDict[key] += 1
                    stagesList.append(valuesDict)
                    #print(stagesList)
                    #sys.exit()
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

                    for e in branchElectron:
                        if e.PT > 10 and (abs(e.Eta) < 1.37 or 1.52 < abs(e.Eta) < 2.47):
                            leptons.append({"N": entry, 'pdg': 11, "pt":e.PT,
                                            "eta":e.Eta, 'phi': e.Phi, 'mass': 0.000511, 'MET': miss})

                    for mu in branchMuon:
                        if mu.PT > 10 and abs(mu.Eta) < 2.7:
                            leptons.append({"N": entry, 'pdg': 13, "pt": mu.PT,
                                            "eta": mu.Eta, 'phi': mu.Phi, 'mass': 0.10566, 'MET': miss})

                chain.Clear()

                if len(stagesList) > 0:
                    pre_df = pd.DataFrame(stagesList).set_index('N')
                    pre_df.columns = pd.MultiIndex.from_tuples(pre_df.columns)
                    print(pre_df)
                    pre_df.to_pickle(out_file.replace('_photons', '_DelphesCuts'))
                #sys.exit()

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
                #print(df_leps)

                if 'All' in input_file:
                    nbins = 50

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

