import sys
import matplotlib.pyplot as plt
import glob
import re
import json
import numpy as np
from itertools import groupby
import contextlib

def extract(filenames):
    with contextlib.ExitStack() as stack:
        # This allows us to access the context of multiple jsons in a single "with"
        files = [stack.enter_context(open(fname,'r')) for fname in filenames]
        meats = [json.load(meat) for meat in files]
    numbers = [np.asarray(nums)[-1,-1,-1] for c in meats for nums in c.values()]
    return sum(numbers)


origin = f"./data/matrices/"
destiny = f"./data/"
files = list(reversed(sorted(glob.glob(origin + f"bin_matrices-MN*.json"))))
times = list(set([re.search(f'/.*T(.+)_', a).group(1) for a in files]))
times = map(lambda a: str(a).replace('.',','), sorted([float(x.replace(',','.')) for x in times]))
times = list(times)
#print(times)
benchs = [[file for file in files if time in file] for time in times]
events = list(map(extract,benchs))

print(benchs)

sigmaBR = [100*0.041/(event/(139 * 0.2)) for event in events]
alphaBR = [100*5.83/(event/0.2) for event in events]
alphaprimeBR = [100*6.84/(event/0.2) for event in events]

float_times = [float(x.replace(',','.')) for x in times]

plt.plot(float_times, sigmaBR, color='r', label= 'sigma 95')
plt.plot(float_times, alphaBR, color='b', label= 'alpha')
plt.plot(float_times, alphaprimeBR, color='g', label= 'alpha prima')
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.3,100)
plt.xlim(0.3,100)
plt.ylabel('BR(H -> NLSP NLSP) %')
plt.title('NLSP: 60GeV - LSP: 0.5GeV')
plt.legend()
plt.show()
