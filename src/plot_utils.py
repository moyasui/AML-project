### This is a template for all python plotting scripts to have the same structure
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # for nice plots


cmt = 1 / 2.54 # cm to inch
# use pastel seaborn pallete
sns.set_palette('pastel') # https://seaborn.pydata.org/tutorial/color_palettes.html
# use white background
sns.set_style("white") # https://seaborn.pydata.org/tutorial/aesthetics.html




# Set all fonts to be equal to tex
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["text.usetex"] = True

# Saving parameters
plt.rcParams["savefig.dpi"] = 300

# Figure options, set tight layout
plt.rc("figure", autolayout=True)

# Font sizes
plt.rc("axes", titlesize=20, labelsize=18)
plt.rc("legend", fontsize=16, shadow=True)

# Tick parameters
_ticks_default_parameters = {
    "labelsize": 14
}
plt.rc("xtick", **_ticks_default_parameters)
plt.rc("ytick", **_ticks_default_parameters)

# Line options
plt.rc("lines", linewidth=2)


def make_figs_path(filename):
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != "AdvancedMachineLearning":
        root_path = root_path.parent

    figs_path = root_path / pl.Path("Analysis/figs")

    if not figs_path.exists():
        return None
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    figs_path /= filename

    return str(figs_path)


def save(filename):
    if filename:
        filename = make_figs_path(filename)
        plt.gcf().set_size_inches(22 * cmt, 18 * cmt)
        plt.tight_layout()
        plt.savefig(filename)