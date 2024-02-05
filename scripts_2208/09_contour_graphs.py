import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from math import floor, ceil
import scipy.ndimage
import sys

k_factors = {'ZH':1.491,'WH':1.253,'TTH':1.15}

deltas = ['01','15']
events_up = {6.84578:['1'],3.84578:['2+'],6.84844:['1','2+']}
sigmas_up = {0.042:['1'],0.022:['2+'],0.041:['1','2+']}

for delta in deltas[1:]:
    print(delta)
    origin = f"./data/matrices_{delta}GeV/"
    destiny = f"./data/"
    names = list(sorted(glob.glob(origin + f"bin_*.json")))
    #for n_up, channels in events_up.items():
    for s_up, channels in sigmas_up.items():
        print(channels)
        ## Opening the files and assigning mass and alpha as tag
        values=[]
        for name in names[:]:
            mass = float(re.search(f'/.*M(\d+,?\d+|\d+)_', name).group(1).replace(',','.'))
            alpha = float(re.search(f'/.*Alpha(\d+,?\d+|\d+)_', name).group(1).replace(',','.'))
            proccess = re.search(f'/.*_13-(.*).json', name).group(1)
            #print(mass, alpha)
            with open(name, 'r') as file:
                info = json.load(file)
            info = [np.asarray(info[ch])[-1,-1,-1] for ch in channels]
            #print(info)
            #sys.exit()
            values.append([(mass,alpha),(proccess,sum(info))])

        # Grouping them by same mass and alpha
        points = {}
        for value in values:
            points.setdefault(value[0], []).append(value[-1])
        print(points)

        pre_params = [set(i for i,j in points.keys()), set(j for i,j in points.keys())]
        masses, alphas = [sorted(list(x)) for x in pre_params]
        print(masses, alphas)

        # Keeping only the ones that have three elements
        points = {key: [k_factors[proc[0]]*proc[1] for proc in val] for key, val in points.items() if len(val) == 3}
        points = {key: val for key, val in points.items()}# if key[0] >2 and key[1] < 8}

        #print(points)
        #sys.exit()
        # Sum all the channels
        points = {key: sum(val) for key, val in points.items()}

        # Get the branching ratio
        not_points = [key for key, val in points.items() if val == 0]
        #points = {key: 100 * n_up/(val/0.2) for key, val in points.items() if val > 0 }#and key[0]%1==0 and key[1]%1==0}
        points = {key: 100 * s_up/(val/(139 * 0.2)) for key, val in points.items() if val > 0 }#and key[0]%1==0 and key[1]%1==0}
        data = [[*key,val] for key, val in points.items()]

        pd.DataFrame(data).to_csv(destiny + f'datapoints_{delta}GeV-{"_".join(channels)}-Sigma.dat',index=False)
        print(data)
#sys.exit()
#
# print(f'points considered: {len(points)}')
# print(f'points not considered: {len(not_points)}')
#
# zoom = 1
# x,y = np.meshgrid(masses,alphas)
# z = np.full((len(alphas),len(masses)),max(points.values()))#max(points.values()))
# print(len(masses),len(alphas),z.shape)
#
# for xi, mass in enumerate(masses):
#     for yi, alpha in enumerate(alphas):
#         if (mass, alpha) in points:
#             val = points[(mass, alpha)]
#             #if val <= 500.:
#             if val > 0.:
#                 z[yi, xi] = val
#
# #print(z[9,1])
# #z = 10.**scipy.ndimage.zoom(np.log10(z),zoom)
# #print(z)
# color_min = floor(np.log10(min([x for x in points.values() if x > 0])))
# color_max = ceil(np.log10(max(points.values())))
#
# levels = 10. ** np.arange(color_min,color_max+1)
# #levels = 10. ** np.array([-1,0,1,2,5,19])
# plt.contourf(x,y,z,levels=levels,locator=ticker.LogLocator())
# plt.colorbar()
#
# not_px = [zoom*x for x,y in not_points]
# not_py = [zoom*y for x,y in not_points]
# px = [zoom*x for x,y in points.keys()]
# py = [zoom*y for x,y in points.keys()]
#
# #plt.scatter(px, py, color='pink')
# #plt.scatter(not_px, not_py, color='orange')
#
# plt.xlabel('MASS')
# plt.ylabel('ALPHA')
# plt.xlim(1,10)
# plt.ylim(1,10)
# plt.show()