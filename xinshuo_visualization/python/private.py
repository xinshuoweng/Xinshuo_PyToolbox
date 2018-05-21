# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes private functions for internal use only in visualization
import numpy as np
import matplotlib.pyplot as plt

from xinshuo_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists_or_creatable, isfile

def get_fig_ax_helper(fig=None, ax=None):
    if fig is None: fig = plt.gcf()
    if ax is None: ax = plt.gca()   
    return fig, ax

def save_vis_close_helper(fig=None, ax=None, vis=False, save_path=None, debug=True, transparent=True, closefig=True):
    # save and visualization
    if save_path is not None:
        if debug: mkdir_if_missing(save_path)
        fig.savefig(save_path, dpi=80, transparent=transparent)
    if vis: plt.show()
    if closefig:
        plt.close(fig)
        return None, None
    else: return fig, ax

def autopct_generator(upper_percentage_to_draw):
    '''
    this function generate a autopct when draw a pie chart
    '''
    def inner_autopct(pct):
        return ('%.2f' % pct) if pct > upper_percentage_to_draw else ''
    return inner_autopct

def fixOverLappingText(text):
    # if undetected overlaps reduce sigFigures to 1
    sigFigures = 2
    positions = [(round(item.get_position()[1],sigFigures), item) for item in text]

    overLapping = Counter((item[0] for item in positions))
    overLapping = [key for key, value in overLapping.items() if value >= 2]

    for key in overLapping:
        textObjects = [text for position, text in positions if position == key]

        if textObjects:

            # If bigger font size scale will need increasing
            scale = 0.1

            spacings = np.linspace(0,scale*len(textObjects),len(textObjects))

            for shift, textObject in zip(spacings,textObjects):
                textObject.set_y(key + shift)