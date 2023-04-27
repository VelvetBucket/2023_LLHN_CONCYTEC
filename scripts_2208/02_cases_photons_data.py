import re
import json
import numpy as np
from pathlib import Path
import pandas as pd
from my_funcs import my_arctan
import sys
import glob

ATLASdet_radius= 1.5
ATLASdet_semilength = 3.512

mass_conversion = 1.78266192*10**(-27)	#GeV to kg
p_conversion = 5.344286*10**(-19)	#GeV to kg.m/s
c_speed = 299792458	#m/s


def pipeline(detec_radius, detec_semilength, detec_name):

    mwpairs = set(re.search(f'({type}.+)\-', x).group(1) for x in
                glob.glob(f'./data/clean/recollection_photons-{type}*{tev}-*.json'))

    for base_out in sorted(list(mwpairs))[:]:

        z_prompts=[]
        z_origin = []
        pts = []
        pzs = []
        tofs = []
        tofs_b = []
        tofs_e = []
        p_tofs = []
        rel_tofs = []
        nus = []
        tns = []
        tphs = []
        trs = []
        tzs = []
        counter = 0
        dicts = []

        for file_in in sorted(glob.glob(f'./data/clean/recollection_photons-{base_out}-*.json'))[:]:

            print(file_in)
            try:
                del data
            except UnboundLocalError:
                file_in

            with open(file_in, 'r') as file:
                data = json.load(file)
            print(len(data.keys()))

            for event in list(data.keys())[:]:
                if (int(event) % 500) == 0:
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

                ix = 1

                for photon in holder['a']:
                    info = dict()
                    info['Event'] = int(event)

                    vertex = str(photon[-1])
                    pid = photon[0]
                    px, py, pz = [p_scaler*ix for ix in photon[1:4]]
                    x, y, z = [d_scaler*ix for ix in holder['v'][vertex][0:3]]
                    mass_ph = photon[-2] * p_scaler
                    r = np.sqrt(x ** 2 + y ** 2)
                    # Calculating transverse momentum
                    pt = np.sqrt(px ** 2 + py ** 2)
                    Et = np.sqrt(mass_ph ** 2 + pt ** 2)
                    E = np.sqrt(mass_ph ** 2 + pt ** 2 + pz ** 2)

                    # print(mass_ph)
                    info['pid'] = pid
                    info['r'] = r / r_detec
                    info['z'] = z / z_detec
                    info['px'] = px
                    info['py'] = py
                    info['pt'] = pt
                    info['pz'] = pz
                    info['ET'] = Et
                    info['E'] = E
                    ix += 1

                    if pt < 10.0:
                        continue
                    elif r >= (r_detec) or abs(z) >= (z_detec):
                         continue

                    # Calculating the z_origin of each photon
                    v_z = np.array([0, 0, 1])  # point in the z axis
                    d_z = np.array([0, 0, 1])  # z axis vector

                    v_ph = np.array([x, y, z])
                    d_ph = np.array([px, py, pz])

                    n = np.cross(d_z, d_ph)

                    n_ph = np.cross(d_ph, n)

                    c_z = v_z + (((v_ph - v_z) @ n_ph) / (d_z @ n_ph)) * d_z

                    # Calculating the time of flight
                    try:
                        # TIme of the neutralino
                        vertex_n = str(holder['n5'][vertex][-1])
                        mass_n = holder['n5'][vertex][-2] * p_scaler
                        # print(mass_n)
                        px_n, py_n, pz_n = [p_scaler*ix for ix in holder['n5'][vertex][0:3]]
                        x_n, y_n, z_n = [d_scaler*ix for ix in holder['v'][vertex_n][0:3]]
                        # print(vertex_n)
                        dist_n = np.sqrt((x - x_n) ** 2 + (y - y_n) ** 2 + (z - z_n) ** 2)
                        p_n = np.sqrt(px_n ** 2 + py_n ** 2 + pz_n ** 2)

                        prev_n = (p_n * p_conversion) / (mass_n * mass_conversion)
                        v_n = (prev_n / np.sqrt(1 + (prev_n / c_speed) ** 2)) * 1000  # m/s to mm/s
                        t_n = dist_n / v_n  # s
                        t_n = t_n * (10 ** 9)  # ns
                        tns.append(t_n)
                        #print(z)
                        ic = 0
                    except KeyError:
                        t_n = 0.0
                        #x= y = z = r = 0.0
                        ic = 1
                        z_prompts.append(z)
                        #print(z)
                    #print(t_n)
                    # Now, time of the photon
                    vx = (c_speed * px / np.linalg.norm(d_ph)) * 1000  # mm/s
                    vy = (c_speed * py / np.linalg.norm(d_ph)) * 1000  # mm/s
                    vz = (c_speed * pz / np.linalg.norm(d_ph)) * 1000  # mm/s

                    tr = (-(x * vx + y * vy) + np.sqrt(
                        (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * (r_detec ** 2 - r ** 2))) / (
                             (vx ** 2 + vy ** 2))

                    if tr < 0:
                        tr = (-(x * vx + y * vy) - np.sqrt(
                            (x * vx + y * vy) ** 2 + (vx ** 2 + vy ** 2) * ((r_detec) ** 2 - r ** 2))) / (
                                 (vx ** 2 + vy ** 2))

                    tz = (np.sign(vz) * z_detec - z) / vz

                    # Now we see which is the impact time
                    if tr < tz:
                        # rf = r_detec
                        rf = r_detec
                        zf = z + vz * tr
                        t_ph = tr * (10 ** 9)

                        x_final = x + vx * tr
                        y_final = y + vy * tr

                    elif tz < tr:
                        rf = np.sqrt((y + vy * tz) ** 2 + (x + vx * tz) ** 2)
                        zf = np.sign(vz) * z_detec
                        t_ph = tz * (10 ** 9)

                        x_final = x + vx * tz
                        y_final = y + vy * tz

                    else:
                        rf = r_detec
                        zf = np.sign(vz) * z_detec
                        t_ph = tz * (10 ** 9)

                        x_final = x + vx * tz
                        y_final = y + vy * tz

                    tof = t_ph + t_n

                    prompt_tof = (10**9)*np.sqrt(rf**2+zf**2)/(c_speed*1000)
                    rel_tof = tof - prompt_tof
                    phi = my_arctan(y_final, x_final)

                    theta = np.arctan2(rf, zf)
                    nu = -np.log(np.tan(theta / 2))

                    counter += 1

                    z_origin.append(c_z[-1])
                    pts.append(pt)
                    pzs.append(pz)
                    tofs.append(tof)
                    if abs(nu) < abs(-np.log(np.tan(np.arctan2(r_detec, z_detec) / 2))):
                        tofs_b.append(tof)
                    else:
                        tofs_e.append(tof)
                    p_tofs.append(prompt_tof)
                    rel_tofs.append(rel_tof)
                    nus.append(nu)
                    #print(t_n)
                    tphs.append(t_ph)
                    trs.append(tr * (10 ** 9))
                    tzs.append(tz * (10 ** 9))

                    info['eta']=nu
                    info['phi']=phi
                    info['z_origin'] = abs(c_z[-1])
                    info['rel_tof'] = rel_tof

                    dicts.append(info)

        print(f'Detected photons in {detec_name}: {counter}')

        dicts = pd.DataFrame(dicts)
        print(dicts)
        #sys.exit()
        #print(dicts.shape)
        #print(dicts.z.max(), dicts.z.min(), dicts.r.max())

        dicts.to_pickle(destiny_info+f'photon_df-{base_out}.pickle')
        print('df saved!')
        # Grafiquitos
    return


types = ["ZH","WH","TTH"]
tevs = [13]

for type in types[:]:
    for tev in tevs[:]:

        destiny_info = './data/clean/'

        pipeline(ATLASdet_radius,ATLASdet_semilength,'ATLAS')