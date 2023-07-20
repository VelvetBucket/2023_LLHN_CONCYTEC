import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

cross_sec = {13:{'VBF': {50: 0.231009, 30: 0.233562, 10: 0.234305}},
             8: {'VBF': {50: 1.12885, 30: 1.14555, 10: 1.13663},'GF': {50: 7.4004, 30: 7.4165, 10: 7.409}}} # pb
l = {8: 20.3,13: 139} #fb-1
br_nn = 0.21
br_np = {8: {50: 0.719,30: 0.935,10: 0.960}, 13:{50: 0.799, 30: 0.938, 10: 0.960}}

def isolation(phs, surr, obs, same=False, dR=0.2):
    phs_list = []
    for ix in phs.index.get_level_values(0).unique()[:]:
        event_ph = phs.loc[ix]
        try:
            event_surr = surr.loc[ix]
            # print(event_surr)
            for index_ph, row_ph in event_ph.iterrows():
                # print(row_ph)
                cone = 0
                for index_d, row_d in event_surr.iterrows():
                    dr = np.sqrt((row_d.phi - row_ph.phi) ** 2 + (row_d.eta - row_ph.eta) ** 2)
                    if same and (index_ph == index_d):
                        dr = dR*1000
                    if dr < dR:
                        cone += row_d[obs]
                phs_list.append(cone)
        except KeyError:
            phs_list.extend([0]*len(event_ph))

    return phs_list

def my_arctan(y,x):

    arctan = np.arctan2(y,x)
    if not isinstance(x, (pd.Series, np.ndarray)):
        arctan = np.asarray(arctan)
    arctan[arctan < 0] += 2*np.pi
    return arctan

def get_scale(tev,type,mass,nevents=100000):
    return ((cross_sec[tev][type][mass]*1000)*l[tev]*br_nn*(br_np[tev][mass])**2)/nevents

def format_exponent(ax, axis='y', size=13):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-2, 2))

    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
        x_pos = 0.0
        y_pos = 1.0
        horizontalalignment = 'left'
        verticalalignment = 'bottom'
    else:
        ax_axis = ax.xaxis
        x_pos = 1.0
        y_pos = -0.05
        horizontalalignment = 'right'
        verticalalignment = 'top'

    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()

    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = np.float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' % expo
    else:
        offset_text = "   "
    # Turn off the offset text that's calculated automatically
    ax_axis.offsetText.set_visible(False)

    # Add in a text box at the top of the y axis
    ax.text(x_pos, y_pos, offset_text, fontsize=size, transform=ax.transAxes,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment)

    return ax
