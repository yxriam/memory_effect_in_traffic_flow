import matplotlib.pyplot as plt
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
def nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'figure.dpi': 130, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False, 'lines.linewidth': 1.2})
def panel_label(ax, s, x=-0.18, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left')
