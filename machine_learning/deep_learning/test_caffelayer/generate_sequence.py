#!/usr/bin/env python
# -*- coding: utf-8 -*- 

# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu
import numpy as np
import os, sys
import time, math
# import json
# import argparse
# import pprint
# from numpy.testing import assert_allclose, assert_array_almost_equal, assert_almost_equal
# import random
from copy import deepcopy
from PIL import Image

# self contained modules
import __init__paths__
import caffe
from preprocess import unpreprocess_image_caffe, preprocess_image_caffe
from file_io import mkdir_if_missing
from visualize import visualize_save_image
from check import *

def generate_sequence(img_path, net, save_path, debug=True, vis=True):
	if not is_path_exists(save_path):
		mkdir_if_missing(save_path)
	if debug:
		assert is_path_exists(img_path) and isfile(img_path), 'input image path is not correct'

	mkdir_if_missing(os.path.join(save_path, 'images'))
	mkdir_if_missing(os.path.join(save_path, 'activations'))
	mkdir_if_missing(os.path.join(save_path, 'features'))
	img = Image.open(img_path)
	rot_list = range(0, 360)	# rotate 360 degree

	index = 0
	for rot_tmp in rot_list:
		print('processing RE-Pooling visualization %d/%d' % (index+1, len(rot_list)))
		img_rot = np.array(img.rotate(rot_tmp).convert('L')).astype('float32') / 255.
		save_path_tmp = os.path.join(save_path, 'images', 'image_%03d.jpg' % rot_tmp)
		visualize_save_image(image=img_rot, vis=vis, save=True, save_path=save_path_tmp, debug=debug)		
		inputdata = preprocess_image_caffe([img_rot], debug=debug)	# process as caffe input
		
		net.blobs['data'].data[...] = inputdata
		net.forward()
		activation = deepcopy(net.blobs['activations'].data)
		feature = deepcopy(net.blobs['features'].data)

		activationlist = unpreprocess_image_caffe(activation, debug=debug)
		test = activationlist[0]

		save_path_tmp = os.path.join(save_path, 'activations', 'activations_%03d.jpg' % rot_tmp)
		visualize_save_image(image=activationlist[0], vis=vis, save=True, save_path=save_path_tmp, debug=debug)

		featurelist = unpreprocess_image_caffe(feature, debug=debug)
		save_path_tmp = os.path.join(save_path, 'features', 'features_%03d.jpg' % rot_tmp)
		visualize_save_image(image=featurelist[0], vis=vis, save=True, save_path=save_path_tmp, debug=debug)
		index += 1

def main():
	img_path = '0.jpg'
	generate_type = 'RE_Pooling'

	caffe.set_mode_gpu()
	caffe.set_device(1)	
	net = caffe.Net('%s.prototxt'%generate_type, caffe.TEST)
	

	generate_sequence(img_path, net, save_path='output', debug=False, vis=False)

if __name__ == '__main__':
	main()