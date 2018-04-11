# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu

# this file generate a synthetic image with various objects

# from __future__ import print_function
import numpy as np
from scipy.misc import imresize

from xinshuo_io import load_image

class synthetic_image_data(object):
	def __init__(self, img, debug=True):
		'''
		generate synthetic data on a given image, the image should be an numpy array
		'''
		if isstring(img):
			img = load_image(img, mode='numpy', debug=debug)

		if debug:
			assert isuintimage(img), 'the input image is not correct'

		self.debug = debug
		self.img = img
		self.width = img.shape[1]
		self.height = img.shape[0]

		self.__original = np.copy(img)

	def get_image(self):
		return self.img

	def clean(self):
		self.img = np.copy(self.__original)

	def add_line(self, center_location, height=3, width=100, intensity=255):
		'''
		add a line to the synthetic data

		parameters:
			center_location:		(x, y)
			height:					integer
			width:					integer
			intensity:				[0, 1]
		'''
		if self.debug:
			assert isinteger(height) and isinteger(width), 'the input height and width should be integer'
			assert height > 0 and width > 0, 'the input height and width should be larger than 0'
			assert isinteger(intensity) and intensity >= 0 and intensity <= 255, 'the input line intensity is not correct: %d' % intensity
			assert is2dpts(center_location), 'the center location of the line is not correct'

		x, y = center_location[0], center_location[1]
		left_bound, right_bound = int(x - width/2), int(x + width/2)
		lower_bound, upper_bound = int(y - height/2), int(y + height/2)
		self.img[lower_bound:upper_bound+1, left_bound:right_bound+1, :] = intensity

	def add_background(self, back_img, patch_location, mode='resize'):
		'''
		add random background to a specific path in the image

		parameters:
			back_img:				integer image
			patch_location:			[x_left, y_lower, x_right, y_upper], a location of the patch
			mode:					trun, keep, resize
		'''
		if isstring(back_img):
			back_img = load_image(back_img, mode='numpy', debug=self.debug)

		if self.debug:
			assert isuintimage(back_img), 'the input background image is not correct'
			assert patch_location[0] >= 0 and patch_location[1] >= 0, 'the input patch location is not correct'
			assert patch_location[2] <= self.width and patch_location[3] <= self.height, 'the input patch location is not correct'
			assert patch_location[2] >= patch_location[0] and patch_location[3] >= patch_location[1], 'the input patch location is not correct'
			assert mode == 'trim' or mode == 'resize', 'the input mode is not correct'

		if mode == 'resize':
			back_img = imresize(back_img, (patch_location[3] - patch_location[1], patch_location[2] - patch_location[0], 3))
			self.img[patch_location[1]:patch_location[3], patch_location[0]:patch_location[2], :] = np.copy(back_img)