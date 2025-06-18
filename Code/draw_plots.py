"""
Main options and functions to draw graphs.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from Code.inequality import dststat_age

from Code.utils import path_data_inputs, path_graphs, path_tables, path_results

# Plot options
# Full list: https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
plt.style.use('default')
# Width of lines and size of markers for scatter plots
plt.rc('lines', **{'linewidth': 2, 'markersize': 8})
# Save a gray color to use manually the scatter plots
scatter_color = '#4c4c4c'
# Save colors manually for heatmaps
heatmap_c_slides = 'Greys'
heatmap_c_paper = 'Blues'
# Format ticks
plt.rc('xtick', **{'direction': 'in', 'top': True})
plt.rc('ytick', **{'direction': 'in', 'right': True})
# Margins of x-axis
plt.rcParams['axes.xmargin'] = 0
# Legend
plt.rc('legend', **{'loc': 'best', 'fancybox': True, 'facecolor': 'none'})
# Specify colors for plots
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
                  '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#EB811B', '#800000', '#1a801a', '#708090', '#b6c26b', '#377eb8', '#bd476a',
             '#654321', '#41bfbb', '#d62728', '#e377c2', '#9467bd']
# Set colors by country
colors_c = dict()
colors_c['USA'] = '#000000'
colors_c['CHN'] = colors[0]
colors_c['IND'] = colors[1]
colors_c['DEU'] = colors[2]
colors_c['JPN'] = colors[3]
colors_c['AUS'] = colors[7]
colors_c['ESP'] = '#b3d3ff'  # Add a light blue
colors_c['GBR'] = colors[6]
colors_c['ITA'] = colors[4]
colors_c['NLD'] = colors[8]
colors_c['SWE'] = colors[9]
colors_c['CAN'] = '#0ecc1c'  # Add a light green
colors_c['FRA'] = colors[5]
colors_c['World'] = '#4a7cb3'

plt.rcParams['axes.prop_cycle'] = cycler('color', colors)
# Colors to match the colors used with the Metropolis theme:
#  - #EB811B: orange from the Metropolis theme
#  - #800000: custom red defined in the preamble (RGB: 128, 0, 0)
#  - #1a801a: custom green defined in the preamble (RGB: 26, 128, 26)
#  - #708090: custom gray defined in the preamble (RGB: 112,128,144)
#  - #b76bc2: custom purple defined in the preamble (RGB: 152, 78, 163)
#  - #377eb8: custom blue defined in the preamble (RGB: 55, 126, 184)
#  - #654321: custom brown
#  - #41bfbb: custom blue/green
#  - then some of the default colors for graphs with many lines

# Set colors for High/Medium/Low fertility scenarios
c_low = colors[2]
c_high = colors[1]
c_med = colors[0]
# Set line styles
ls_low = '--'
ls_high = '--'
ls_med = '-'

# Set colors for graphs of decomposition
c_Delta = 'k'
c_Delta_pi = colors[0]
c_Delta_a = colors[1]
c_Delta_h = colors[2]
c_Delta_pidata = 'k'
# Set line styles
ls_Delta = '-'
ls_Delta_pi = '-'
ls_Delta_a = '--'
ls_Delta_h = '--'
ls_Delta_pidata = ':'


# Default padding in plt.tight_layout()
padding = 0.1
w_padding = 0.5
h_padding = 0.5

# Additional options specific for figures for slides
# Note: You need to have installed the 'Fira Sans' font on your computer
# Instructions:  download the .ttf files and paste them in /Library/Fonts/ (for MacOS)
# Download link: https://fonts.google.com/specimen/Fira+Sans?selection.family=Fira+Sans


def figure_slides(figsize=[6.4, 4.8], ncols_subplot=1):
    """"
    Sets parameters for figures to be used in slides. Run before matplotlib commands e.g. plot.

    """
    if ncols_subplot <= 1:
        fontsize_main = 13
        fontsize_legend = 13
    elif ncols_subplot >= 2:
        fontsize_main = 20
        fontsize_legend = 18
    # Size
    plt.rcParams['figure.figsize'] = figsize
    # Font options
    font = {'family': 'sans-serif',
            'weight': 'regular',
            'size': fontsize_main}
    plt.rc('font', **font)
    plt.rc('text', usetex=False)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'sans:italic'
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'  #'Fira Sans'
    plt.rcParams['axes.titleweight'] = 'regular'
    plt.rcParams['axes.labelweight'] = 'regular'
    plt.rcParams['figure.titleweight'] = 'regular'
    # Legend
    plt.rc('legend', **{'fontsize': fontsize_legend, 'borderaxespad': 2, 'edgecolor': 'black'})
    # Resolution of saved figures and transparent background
    plt.rc('savefig', **{'dpi': 250, 'transparent': True})

def figure_paper(ncols_subplot=0, figsize=[6.4, 4.8],):
    """"
    Sets parameters for figures to be used in the paper. Run before matplotlib commands e.g. plot.

    """
    if ncols_subplot <= 1:
        fontsize_main = 14
        fontsize_legend = 14
        border = 1
    elif ncols_subplot == 2:
        fontsize_main = 22
        fontsize_legend = 20
        border = 0.5
    else:
        # Larger font for graphs used in a subplot of 3 or more
        fontsize_main = 28
        fontsize_legend = 28
        border = 0.2

    # Size
    plt.rcParams['figure.figsize'] = figsize
    # Font options
    font = {'family': 'serif',
            'weight': 'regular',
            'size': fontsize_main}
    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
    # Legend
    plt.rc('legend', **{'fontsize': fontsize_legend, 'borderaxespad': border, 'edgecolor': 'none'})
    # Resolution of saved figures and non-transparent background
    plt.rc('savefig', **{'dpi': 150, 'transparent': False})





def figure_theme(figsize = [6.4, 4.8], type = "paper",  ncols_subplot = 0, fontsize_main = None , fontsize_legend = None, border = None ):

    """"
    Sets parameters for figures to be used in the paper. Run before matplotlib commands e.g. plot.

    """
    # Check `type` argument
    types = ["slides", "paper"]
    if type not in types:
        raise ValueError("Invalid `type` argument. Expected one of: %s" % types)
    # Size
    plt.rcParams['figure.figsize'] = figsize

    if type == "slides":

        if fontsize_main is None:
            if ncols_subplot <= 1:
                fontsize_main = 13
                fontsize_legend = 13
            elif ncols_subplot >= 2:
                fontsize_main = 20
                fontsize_legend = 18
        
        if fontsize_legend is None:
            if ncols_subplot <= 1:
                fontsize_legend = 13
            elif ncols_subplot >= 2:
                fontsize_main = 20
            
        # Font options
        font = {'family': 'serif',
                'weight': 'regular',
                'size': fontsize_main}
        plt.rc('font', **font)
        
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
        # Legend
        plt.rc('legend', **{'fontsize': fontsize_legend, 'borderaxespad': 2, 'edgecolor': 'none'})

        # plt.rc('text', usetex=False)
        # plt.rcParams['mathtext.fontset'] = 'custom'
        # plt.rcParams['mathtext.it'] = 'sans:italic'
        # plt.rcParams['font.sans-serif'] = 'Fira Sans'  #'Fira Sans'
        # plt.rcParams['axes.titleweight'] = 'regular'
        # plt.rcParams['axes.labelweight'] = 'regular'
        # plt.rcParams['figure.titleweight'] = 'regular'
        # Legend
        #plt.rc('legend', **{'fontsize': fontsize_legend, 'borderaxespad': 2, 'edgecolor': 'black'})
        # Resolution of saved figures and transparent background
        plt.rc('savefig', **{'dpi': 250, 'transparent': True})

    else:

        if fontsize_main is None:
            if ncols_subplot <= 1:
                fontsize_main = 14
            elif ncols_subplot == 2:
                fontsize_main = 22
            else:# Larger font for graphs used in a subplot of 3 or more
                fontsize_main = 28
                

        if fontsize_legend is None:
            if ncols_subplot <= 1:
                fontsize_legend = 14
            elif ncols_subplot == 2:
                fontsize_legend = 20
            else:# Larger font for graphs used in a subplot of 3 or more
                fontsize_legend = 28
        
        if border is None:
            if ncols_subplot <= 1:
                border = 1
            elif ncols_subplot == 2:
                border = 0.5
            else:# Larger font for graphs used in a subplot of 3 or more
                border = 0.2



        # Font options
        font = {'family': 'serif',
                'weight': 'regular',
                'size': fontsize_main}
        plt.rc('font', **font)
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{mathpazo}')
        # Legend
        plt.rc('legend', **{'fontsize': fontsize_legend, 'borderaxespad': border, 'edgecolor': 'none'})
        # Resolution of saved figures and non-transparent background
        plt.rc('savefig', **{'dpi': 150, 'transparent': False})

        return plt.figure(figsize = figsize)
    
