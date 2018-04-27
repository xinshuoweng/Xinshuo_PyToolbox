# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# this file includes classes of linear filters ready applied on the images
import numpy as np
# import math, time, copy
# import matplotlib.pyplot as plt
# from numpy.testing import assert_almost_equal
# from math import radians as rad

# from private import safe_bbox, safe_center_bbox, bboxcheck_TLBR, bboxcheck_TLWH

class linear_filter(object):
	def __init__(self, filter_size, warning=True, debug=True):
		'''
		generate synthetic data on a given image, the image should be an numpy array
		'''
		if debug: isimsize(filter_size), 'the filter size is not correct'
		self.debug = debug
		self.warning = warning
		self.filter_size = filter_size

	def get_image(self):
		return self.img
