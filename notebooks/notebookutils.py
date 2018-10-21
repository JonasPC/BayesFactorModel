# utility functions for notebooks

import os
import pandas as pd
import seaborn as sbn


def root_dir():
    os.chdir("..")
    print('now in dir: ', os.getcwd())


def corr_plot(df, save_path=None):
    """Shows correlation plot

    Parameters
    ==========
    df : (object) pd.DataFrame
        df on which to find correlations
    save_path: (str)
        default = None - path to save figure. usually 'figs//some_file_name'
        The '.png' is automatically added.

    Returns
    =======
    Display of correlation plot
    """

    corr = df.corr()
    cmap = sbn.diverging_palette(220, 10, as_cmap=True)
    sbn_plot = sbn.heatmap(corr, cmap=cmap, center=0,
                           square=True, linewidths=.5)

    if save_path is not None:
        fig = sbn_plot.get_figure()
        fig.savefig(save_path + '.png')
