# import this to have global figure config
# like fonts, export format, page width, dpi

#               v  pt to inch conversion
textwidth = (1/72.27) * 369.0
#                      \the\textwidth in pt

import matplotlib.pyplot as pl

# match report font
# nvm, looks shit
# pl.rcParams['font.family'] = 'serif'
# pl.rcParams['font.serif'].insert(0, 'Palatino')


data_dir = 'plot_data'
show = True
pagewidth = 2 * textwidth   # to make it look nicer...
fig_dir = './report_figs/'
fig_format = 'pdf'
dpi = 100

# confidence band width
sigs = 3
confidence_band_alpha = 0.1
