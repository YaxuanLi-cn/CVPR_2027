"""Shared matplotlib style of the supplementary figures (print, single CVPR column = 3.3 in).
Colours: the first three slots of a palette validated for colour-vision deficiency (blue, orange, aqua), assigned by entity:
  refined stage / primary series -> BLUE, coarse stage / second series -> ORANGE, third series -> AQUA, context -> GREY.
Text and axes use ink colours, never a series colour."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#c9c8c2'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3de'
COL = 3.3      # column width in inches
FIG = '/root/rasr-pairuav/docs/paper/supp/figures'
def use():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
                         'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.8, 'axes.linewidth': 0.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                         'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.4,
                         'axes.axisbelow': True, 'lines.linewidth': 1.3, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round', 'legend.frameon': False, 'legend.handlelength': 1.4, 'legend.borderaxespad': 0.2, 'figure.facecolor': 'white',
                         'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.titlepad': 3, 'axes.labelpad': 2})
def save(fig, name):
    import os; os.makedirs(FIG, exist_ok=True)
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight', pad_inches=0.02); fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight', pad_inches=0.02); plt.close(fig); print('saved', name)
