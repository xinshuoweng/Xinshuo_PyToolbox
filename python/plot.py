# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file define a set of functions related to matplotlib
from collections import Counter
import numpy as np

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
