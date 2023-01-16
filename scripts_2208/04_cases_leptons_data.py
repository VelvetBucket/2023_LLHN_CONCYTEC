import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from my_funcs import my_arctan
import sys
import glob
import re

#CMSdet_radius = 1.29 # meters
#CMSdet_semilength = 2.935
ATLASdet_radius= 1.4
ATLASdet_semilength = 2.9

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s


def plot_config(figsize,xtitle,ytitle,xsize, ysize, xtsize, ytsize):
    plt.figure(figsize=figsize)
    plt.ylabel(ytitle, fontsize=ysize)
    plt.xlabel(xtitle, fontsize=xsize)
    plt.xticks(fontsize=xtsize)
    plt.yticks(fontsize=ytsize)
    return

def pipeline(detec_radius, detec_semilength, detec_name):

    mwpairs = set(re.search(f'({type}.+)\-', x).group(1) for x in
                  glob.glob(f'./data/clean/recollection_leptons-{type}*{tev}-*.json'))

    for base_out in sorted(list(mwpairs))[:]:

        pts = []
        pzs = []
        nus = []

        counter = 0
        dicts = []

        for file_in in sorted(glob.glob(f'./data/clean/recollection_leptons-{base_out}-*.json'))[:]:

            print(file_in)
            try:
                del data
            except UnboundLocalError:
                file_in

            with open(file_in, 'r') as file:
                data = json.load(file)
            #print(len(data.keys()))
            for event in list(data.keys())[:]:
                print(f'RUNNING: {base_out} - {detec_name} - Event {event}')
                holder = data[event]
                params = holder['params']
                # Defining scaler according to parameters units
                if params[0] == 'GEV':
                    p_scaler = 1  # GeV to GeV
                elif params[0] == 'MEV':
                    p_scaler = 1 / 1000  # MeV to GeV
                else:
                    print(params[0])
                    continue

                if params[1] == 'MM':
                    d_scaler = 1  # mm to mm
                elif params[1] == 'CM':
                    d_scaler = 10  # cm to mm
                else:
                    print(params[1])
                    continue

                # Adjusting detector boundaries
                r_detec = detec_radius * 1000  # m to mm
                z_detec = detec_semilength * 1000

                # Define our holder for pairs:
                pt_dum = 0
                ix = 0
                for lepton in holder['l']:
                    info = dict()
                    info['Event'] = int(event)

                    vertex = str(lepton[-1])
                    pdg = lepton[0]
                    px, py, pz = [p_scaler*ix for ix in lepton[1:4]]
                    x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                    mass = lepton[-2] * p_scaler
                    r = np.sqrt(x ** 2 + y ** 2)
                    # Calculating transverse momentum
                    pt = np.sqrt(px ** 2 + py ** 2)
                    Et = np.sqrt(mass ** 2 + pt ** 2)
                    E = np.sqrt(mass ** 2 + pt ** 2 + pz ** 2)
                    if r >= (r_detec) or abs(z) >= (z_detec):
                         continue
                    elif pt < 0.1:
                        continue

                    # print(mass_ph)
                    info['id'] = ix
                    info['pdg'] = int(pdg)
                    info['r'] = r / r_detec
                    info['z'] = z / z_detec
                    info['px'] = px
                    info['py'] = py
                    info['pt'] = pt
                    info['pz'] = pz
                    info['ET'] = Et
                    info['E'] = E

                    ix += 1
                    counter += 1

                    phi = my_arctan(py, px)

                    theta = np.arctan2(pt, pz)
                    nu = -np.log(np.tan(theta / 2))

                    pts.append(pt)
                    pzs.append(pz)
                    nus.append(nu)

                    info['eta']=nu
                    info['phi']=phi

                    dicts.append(info)

        print(f'Detected leptons in {detec_name}: {counter}')

        dicts = pd.DataFrame(dicts)
        dicts = dicts.sort_values(by=['Event','pt'],ascending=[True,False])
        g = dicts.groupby('Event', as_index=False).cumcount()
        dicts['id'] = g
        dicts = dicts.set_index(['Event','id'])
        #print(dicts[np.abs(dicts.pdg)==11][['pt']])

        dicts.to_pickle(destiny_info+f'lepton_df-{base_out}.pickle')
        print('df saved!')

    return

types = ['ZH', "WH", "TTH"]
tevs = [13]

for type in types[:]:
    for tev in tevs[:]:

            destiny_info = './data/clean/'

            pipeline(ATLASdet_radius,ATLASdet_semilength,'ATLAS')